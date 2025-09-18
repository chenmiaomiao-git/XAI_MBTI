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
import hmac
import hashlib
import time
import json
import uuid
import re
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
                except Exception as e:
                    print(f"检查WAV文件时出错: {e}")
                    # 继续尝试转换
            
            # 创建临时文件
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # 使用pydub转换
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav_file, format="wav")
            
            return temp_wav_file
        except Exception as e:
            print(f"音频转换失败: {e}")
            # 返回None而不是原文件路径，让调用者知道转换失败
            return None
    
    @classmethod
    def asr_recognize(cls, audio_file_path, language="English"):
        """语音识别(ASR)
        
        Args:
            audio_file_path: 音频文件路径
            language: 语言选择，支持 "Chinese", "English", "Japanese", "French"
        """
        token = cls.get_baidu_access_token()
        if not token:
            return "Unable to get Baidu access token"
        
        # 转换为wav格式并设置正确的采样率
        temp_wav_file = cls.convert_to_wav(audio_file_path)
        if temp_wav_file is None:
            return "音频格式转换失败"
            
        try:
            with open(temp_wav_file, "rb") as f:
                audio_data = f.read()
        except Exception as e:
            print(f"读取音频文件失败: {e}")
            return "读取音频文件失败"
        
        # 如果是临时创建的文件，使用后删除
        if temp_wav_file != audio_file_path:
            try:
                os.unlink(temp_wav_file)
            except Exception as e:
                print(f"删除临时文件失败: {e}")
        
        # 根据语言选择设置不同的语音识别模型
        dev_pid = 1737  # 默认英语模型
        if language == "Chinese":
            dev_pid = 1537  # 普通话模型
        elif language == "English":
            dev_pid = 1737  # 英语模型
        elif language == "Japanese":
            dev_pid = 1737  # 使用英语模型（百度API不直接支持日语，可以考虑其他服务）
        elif language == "French":
            dev_pid = 1737  # 使用英语模型（百度API不直接支持法语，可以考虑其他服务）
        
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
            "dev_pid": dev_pid
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
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
        except requests.exceptions.Timeout:
            print("语音识别请求超时")
            return "语音识别请求超时"
        except requests.exceptions.RequestException as e:
            print(f"语音识别网络请求异常: {e}")
            return "语音识别网络请求失败"
    
    @classmethod
    def asr_recognize_from_numpy(cls, audio_data, sample_rate, language="English"):
        """从NumPy数组进行语音识别
        
        Args:
            audio_data: 音频数据NumPy数组
            sample_rate: 采样率
            language: 语言选择，支持 "Chinese", "English", "Japanese", "French"
        """
        # 保存到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_file.close()
            
            # 进行语音识别
            result = cls.asr_recognize(temp_file.name, language=language)
            
            return result
        except Exception as e:
            print(f"处理NumPy音频数据时出错: {e}")
            return "音频数据处理失败"
        finally:
            # 确保临时文件被删除
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                 print(f"删除临时文件失败: {e}")
    
    @classmethod
    def tts_synthesize(cls, text, spd=DEFAULT_VOICE_SPEED, pit=DEFAULT_VOICE_PITCH, vol=DEFAULT_VOICE_VOLUME, per=DEFAULT_VOICE_PERSON, tts_engine="baidu", language="English", voice_type=None, speed=1.0, pitch=1.0):
        """语音合成(TTS)
        
        Args:
            text: 要合成的文本
            spd: 语速 (百度云参数)
            pit: 音调 (百度云参数)
            vol: 音量 (百度云参数)
            per: 发音人 (百度云参数)
            tts_engine: TTS引擎，支持 "baidu" 和 "volcano"
            language: 语言选择，支持 "Chinese", "English", "Japanese"
            voice_type: 火山引擎音色类型，如 "BV700_streaming"(灿灿), "BV701_streaming"(擎苍), "BV702_streaming"(Stefan)
            speed: 语速 (火山引擎参数，范围：0.5~2.0)
            pitch: 音调 (火山引擎参数，范围：0.5~2.0)
        """
        if tts_engine == "volcano":
            return cls.volcano_tts_synthesize(text, language=language, voice_type=voice_type, speed=speed, pitch=pitch)
        
        # 默认使用百度云TTS
        token = cls.get_baidu_access_token()
        if not token:
            return None
        
        # 根据语言选择设置不同的语言代码
        lan = "en"  # 默认英语
        if language == "Chinese":
            lan = "zh"
        elif language == "English":
            lan = "en"
        # 百度API不支持日语和法语，使用英语作为后备
        # 注意：百度API不支持jp作为语言代码
        elif language == "Japanese" or language == "French":
            lan = "en"
        
        url = "https://tsn.baidu.com/text2audio"
        params = {
            "tex": text,
            "tok": token,
            "cuid": "123456PYTHON",
            "ctp": 1,
            "lan": lan,
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
    def volcano_tts_synthesize(cls, text, language="English", voice_type=None, speed=1.0, pitch=1.0):
        """使用火山引擎进行语音合成
        
        Args:
            text: 要合成的文本
            language: 语言选择，支持 "Chinese", "English", "Japanese"（不支持"French"）
            voice_type: 音色类型，如果为None则根据语言自动选择
            speed: 语速，范围：0.5~2.0
            pitch: 音调，范围：0.5~2.0
        """
        import json
        import time
        import uuid
        import hmac
        import hashlib
        
        # 火山引擎TTS API地址 - 在URL中包含token
        url = f"https://openspeech.bytedance.com/api/v1/tts?token={VOLCANO_ACCESS_TOKEN}"
        
        # 生成唯一请求ID
        req_id = str(uuid.uuid4())
        
        # 根据语言选择设置不同的语言代码
        lang_code = "en"  # 默认英语
        if language == "Chinese":
            lang_code = "zh"  # 火山引擎使用zh而不是cn
        elif language == "English":
            lang_code = "en"
        elif language == "Japanese":
            lang_code = "ja"  # 火山引擎使用ja而不是jp
        else:
            # 不支持其他语言，使用英语作为后备
            lang_code = "en"
            
        # 根据语言选择合适的音色
        if voice_type is None:
            if language == "Chinese":
                voice_type = "BV700_streaming"  # 灿灿
            elif language == "English":
                voice_type = "BV702_streaming"  # Stefan
            elif language == "Japanese":
                voice_type = "BV700_streaming"  # 灿灿（支持日语）
            else:
                voice_type = "BV700_streaming"  # 默认使用灿灿
        
        # 确保文本不为空且进行格式处理
        if not text or text.strip() == "":
            print("火山引擎语音合成错误: 文本内容为空")
            return None
        
        # 处理文本，确保符合火山引擎要求
        processed_text = text.strip()
        
        # 移除可能导致问题的特殊字符
        import re
        # 简化文本处理，移除控制字符和特殊字符
        processed_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', processed_text)  # 移除控制字符和不可见字符
            
        # 准备请求体 - 根据火山引擎最新API文档格式
        request_data = {
            "app": {
                "appid": VOLCANO_APP_ID,  # 使用配置文件中的appid
                "token": "",  # 使用签名认证，token为空
                "cluster": "volcano_tts"  # 根据文档使用正确的cluster值
            },
            "user": {
                "uid": "user_" + str(int(time.time()))  # 动态生成用户ID
            },
            "audio": {
                "voice_type": voice_type,  # 使用传入的音色参数
                "encoding": "mp3",
                "rate": 24000,
                "speed_ratio": speed,  # 使用传入的语速参数，范围：0.2~3.0
                "volume_ratio": 1.0,  # 音量，范围：0.1~3.0
                "pitch_ratio": pitch,   # 使用传入的音调参数，范围：0.1~3.0
                "language": lang_code  # 使用根据language参数设置的语言代码
            },
            "request": {
                "reqid": req_id,
                "text": processed_text,  # 使用处理后的文本
                "text_type": "plain",
                "operation": "query",
                "silence_duration": 125
            }
        }
        
        # 针对中文特殊处理，使用百度云TTS而非火山引擎
        if lang_code == "zh":
            # 使用百度云TTS合成中文
            return AudioService.tts_synthesize(text, spd=int(speed*5), pit=int(pitch*5), vol=15, per=0, tts_engine="baidu", language="Chinese")
        
        # 设置请求头 - 使用正确的Bearer Token格式（注意是分号而不是空格）
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer;{VOLCANO_ACCESS_TOKEN}"
        }
        
        # 在函数开始时生成时间戳，确保整个处理过程中使用相同的时间戳
        timestamp = int(time.time())
        # 使用项目的静态目录保存音频文件
        static_dir = os.path.join("/data/LLM-SFT/CCYTest/XAI_MBTI/static")
        os.makedirs(static_dir, exist_ok=True)
        audio_file_path = os.path.join(static_dir, f"tts_output_{timestamp}.mp3")
        print(f"保存音频数据到: {audio_file_path}")
        
        # 打印请求信息以便调试
        print(f"火山引擎TTS请求: 语言={lang_code}, 文本长度={len(processed_text)}")
        # print(f"请求数据: {json.dumps(request_data, ensure_ascii=False)}")
        # print(f"请求头: {headers}")
        
        try:
            response = requests.post(url, headers=headers, json=request_data)
            print(f"火山引擎响应状态码: {response.status_code}")
            # 不打印响应头信息，避免长段数据
            
            if response.status_code == 200:
                # 检查响应内容类型
                content_type = response.headers.get("Content-Type", "")
                print(f"响应内容类型: {content_type}")
                
                # 首先尝试解析为JSON
                try:
                    result = response.json()
                    # 不打印完整的响应JSON，避免长段数据
                    print(f"火山引擎响应JSON状态码: {result.get('code')}")
                    
                    # 使用预先生成的时间戳和文件路径
                    
                    # 检查是否成功
                    if result.get("code") == 0 or result.get("code") == 3000:  # 火山引擎成功状态码
                        # 从data字段获取base64编码的音频数据
                        audio_base64 = result.get("data", "")
                        if audio_base64:
                            # 解码base64数据
                            audio_data = base64.b64decode(audio_base64)
                            
                            # 直接保存为音频文件，不做额外处理
                            try:
                                print(f"保存音频数据到: {audio_file_path}")
                                with open(audio_file_path, "wb") as f:
                                    f.write(audio_data)
                                if os.path.exists(audio_file_path):
                                    return audio_file_path
                                else:
                                    print(f"错误: 文件保存失败: {audio_file_path}")
                            except Exception as save_error:
                                print(f"保存音频文件时发生错误: {save_error}")
                        else:
                            # 尝试从payload字段获取音频数据
                            payload = result.get("payload", {})
                            if isinstance(payload, dict) and "audio_data" in payload:
                                audio_base64 = payload.get("audio_data", "")
                                if audio_base64:
                                    # 解码base64数据并直接保存
                                    audio_data = base64.b64decode(audio_base64)
                                    try:
                                        with open(audio_file_path, "wb") as f:
                                            f.write(audio_data)
                                        if os.path.exists(audio_file_path):
                                            return audio_file_path
                                        else:
                                            print(f"错误: payload文件保存失败: {audio_file_path}")
                                    except Exception as save_error:
                                        print(f"保存payload音频文件时发生错误: {save_error}")
                            print("火山引擎响应中没有找到音频数据")
                    else:
                        print(f"火山引擎语音合成失败: {result.get('message', '未知错误')}")
                except ValueError as e:
                    print(f"解析JSON响应失败: {e}")
                    # 如果不是JSON格式，可能是直接返回的音频数据
                    if content_type.startswith("audio/"):
                        # 保存为音频文件 - 使用预先生成的文件路径
                        with open(audio_file_path, "wb") as f:
                            f.write(response.content)
                        print(f"已保存直接音频内容到静态文件: {audio_file_path}")
                        return audio_file_path
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