#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本
用于测试音频服务和聊天服务
"""

import os
import time
from audio_service import AudioService
from chat_service import ChatService

def test_tts():
    """测试语音合成"""
    print("测试语音合成...")
    text = "这是一个测试，测试百度语音合成接口。"
    audio_file = AudioService.tts_synthesize(text)
    if audio_file and os.path.exists(audio_file):
        print(f"语音合成成功，文件保存在: {audio_file}")
        return audio_file
    else:
        print("语音合成失败")
        return None

def test_asr(audio_file):
    """测试语音识别"""
    if not audio_file or not os.path.exists(audio_file):
        print("没有可用的音频文件进行测试")
        return
    
    print("测试语音识别...")
    text = AudioService.asr_recognize(audio_file)
    print(f"语音识别结果: {text}")
    return text

def test_chat():
    """测试聊天服务"""
    print("测试聊天服务...")
    message = "你好，请介绍一下你自己"
    print(f"发送消息: {message}")
    
    # 测试ESTJ模型
    print("\n使用ESTJ模型:")
    response = ChatService.send_chat_request(message, lora_path="estj")
    print(f"回复: {response.get('reply', '无回复')}")
    
    # 测试INFP模型
    print("\n使用INFP模型:")
    response = ChatService.send_chat_request(message, lora_path="infp")
    print(f"回复: {response.get('reply', '无回复')}")

def main():
    """主函数"""
    print("=== 开始测试服务 ===\n")
    
    # 测试TTS
    audio_file = test_tts()
    
    # 等待一下，确保文件写入完成
    if audio_file:
        time.sleep(1)
    
    # 测试ASR
    test_asr(audio_file)
    
    # 测试聊天服务
    test_chat()
    
    # 清理临时文件
    if audio_file and os.path.exists(audio_file):
        try:
            os.unlink(audio_file)
            print(f"\n已删除临时文件: {audio_file}")
        except:
            pass
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()