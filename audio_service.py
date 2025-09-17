#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频处理服务模块
提供语音识别(ASR)和语音合成(TTS)功能
基于百度云API和火山引擎API实现
"""

import requests
import base64
import tempfile
import os
from pydub import AudioSegment
import soundfile as sf
import numpy as np

# 导入配置
from config import (BAIDU_API_KEY, BAIDU_SECRET_KEY, DEFAULT_VOICE_SPEED, DEFAULT_VOICE_PITCH, 
                   DEFAULT_VOICE_VOLUME, DEFAULT_VOICE_PERSON, VOLCANO_APP_ID, VOLCANO_ACCESS_TOKEN, VOLCANO_SECRET_KEY)

class AudioService:
    """音频处理服务类"""
    
    @staticmethod
    def get_baidu_access_token():
        """获取百度云access_token"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": BAIDU_API_KEY,
            "client_secret": BAIDU_SECRET_KEY
        }
        response = requests.post(url, params=params)
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            print(f"获取百度access_token失败: {response.text}")
            return None
    
    @staticmethod
    def convert_to_wav(audio_file_path):
        """转换音频为WAV格式"""
        try:
            # 获取文件扩展名
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            
            # 如果已经是wav格式且采样率正确，直接返回
            if file_ext == ".wav":
                try:
                    data, samplerate = sf.read(audio_file_path)
                    if samplerate == 16000:
                        return audio_file_path
                except Exception:
                    pass
            
            # 创建临时文件
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # 使用pydub转换
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav_file, format="wav")
            
            return temp_wav_file
        except Exception as e:
            print(f"音频转换失败: {e}")
            return audio_file_path
    
    @classmethod
    def asr_recognize(cls, audio_file_path):
        """语音识别(ASR)"""
        token = cls.get_baidu_access_token()
        if not token:
            return "无法获取百度云访问令牌"
        
        # 转换为wav格式并设置正确的采样率
        temp_wav_file = cls.convert_to_wav(audio_file_path)
        with open(temp_wav_file, "rb") as f:
            audio_data = f.read()
        
        # 如果是临时创建的文件，使用后删除
        if temp_wav_file != audio_file_path:
            try:
                os.unlink(temp_wav_file)
            except:
                pass
        
        url = "https://vop.baidu.com/server_api"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "format": "wav",
            "rate": 16000,
            "channel": 1,
            "token": token,
            "cuid": "123456PYTHON",
            "len": len(audio_data),
            "speech": base64.b64encode(audio_data).decode('utf-8'),
            "dev_pid": 1737  # 英语模型
        }
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if result.get("err_no") == 0 and result.get("result"):
                return result["result"][0]
            else:
                print(f"语音识别失败: {result}")
                return "语音识别失败"
        else:
            print(f"语音识别请求失败: {response.text}")
            return "语音识别请求失败"
    
    @classmethod
    def asr_recognize_from_numpy(cls, audio_data, sample_rate):
        """从NumPy数组进行语音识别"""
        # 保存到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, sample_rate)
        temp_file.close()
        
        # 进行语音识别
        result = cls.asr_recognize(temp_file.name)
        
        # 删除临时文件
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        return result
    
    @classmethod
    def tts_synthesize(cls, text, spd=DEFAULT_VOICE_SPEED, pit=DEFAULT_VOICE_PITCH, vol=DEFAULT_VOICE_VOLUME, per=DEFAULT_VOICE_PERSON, tts_engine="baidu"):
        """语音合成(TTS)"""
        if tts_engine == "volcano":
            return cls.volcano_tts_synthesize(text)
        
        # 默认使用百度云TTS
        token = cls.get_baidu_access_token()
        if not token:
            return None
        
        url = "https://tsn.baidu.com/text2audio"
        params = {
            "tex": text,
            "tok": token,
            "cuid": "123456PYTHON",
            "ctp": 1,
            "lan": "zh",
            "spd": spd,  # 语速
            "pit": pit,  # 音调
            "vol": vol,  # 音量
            "per": per   # 发音人，0女1男3度逍遥4度丫丫
        }
        
        response = requests.post(url, params=params)
        if response.headers.get("Content-Type", "").startswith("audio/"):
            # 创建临时文件保存音频
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            print(f"语音合成失败: {response.text}")
            return None
            
    @classmethod
    def volcano_tts_synthesize(cls, text):
        """使用火山引擎进行语音合成"""
        import json
        import time
        import uuid
        import hmac
        import hashlib
        
        # 火山引擎TTS API地址
        url = f"https://openspeech.bytedance.com/api/v1/tts?token={VOLCANO_ACCESS_TOKEN}"
        
        # 生成唯一请求ID
        req_id = str(uuid.uuid4())
        
        # 准备请求体
        request_data = {
            "app": {
                "appid": VOLCANO_APP_ID,
                "token": "",  # 使用签名认证，token为空
                "cluster": "volcano_tts"  # 根据文档使用正确的cluster值
            },
            "user": {
                "uid": "user_" + str(int(time.time()))
            },
            "audio": {
                "voice_type": "BV700_streaming",  # 使用文档中的示例音色
                "encoding": "mp3",
                "rate": 24000,
                "speed_ratio": 1.0,  # 语速，范围：0.2~3.0
                "volume_ratio": 1.0,  # 音量，范围：0.1~3.0
                "pitch_ratio": 1.0,   # 音调，范围：0.1~3.0
                "language": "cn"      # 指定语言
            },
            "request": {
                "reqid": req_id,
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "silence_duration": 125
            }
        }
        
        # 设置请求头 - 使用正确的Bearer Token格式（注意是分号而不是空格）
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer;{VOLCANO_ACCESS_TOKEN}"
        }
        
        try:
            response = requests.post(url, headers=headers, json=request_data)
            if response.status_code == 200:
                result = response.json()
                # 根据文档，成功状态码是3000
                if result.get("code") == 3000 and result.get("data"):
                    # 解码Base64音频数据
                    audio_data = base64.b64decode(result["data"])
                    
                    # 创建临时文件保存音频
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    temp_file.write(audio_data)
                    temp_file.close()
                    return temp_file.name
                else:
                    print(f"火山引擎语音合成失败: {result}")
            else:
                print(f"火山引擎语音合成请求失败: {response.text}")
        except Exception as e:
            print(f"火山引擎语音合成异常: {e}")
            
        return None

# 测试代码
if __name__ == "__main__":
    # 测试TTS
    audio_file = AudioService.tts_synthesize("这是一个测试，测试百度语音合成接口。")
    if audio_file:
        print(f"语音合成成功，文件保存在: {audio_file}")
    else:
        print("语音合成失败")
    
    # 如果有音频文件，可以测试ASR
    if audio_file:
        text = AudioService.asr_recognize(audio_file)
        print(f"语音识别结果: {text}")