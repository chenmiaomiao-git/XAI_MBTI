#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聊天服务模块
处理与API的交互，发送聊天请求并处理响应
"""

import requests
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# 导入配置
from config import API_BASE_URL, CHAT_ENDPOINT, CHAT_STREAM_ENDPOINT, DEFAULT_LORA_PATH, DEFAULT_PROMPT_CHOICE, DEFAULT_FORMAT_CHOICE

class ChatService:
    """聊天服务类"""
    
    @staticmethod
    def send_chat_request(message, history=None, lora_path=DEFAULT_LORA_PATH, prompt_choice=DEFAULT_PROMPT_CHOICE, promptFormat_choice=DEFAULT_FORMAT_CHOICE, use_stream=False):
        """发送聊天请求
        
        Args:
            message: 用户消息
            history: 聊天历史记录
            lora_path: 模型路径，可选值："estj", "infp", "base_1", "base_2"
            prompt_choice: 提示词选择，可选值："assist_estj", "assist_infp", "null"
            promptFormat_choice: 格式选择，可选值："ordinary", "custom"
            use_stream: 是否使用流式API
            
        Returns:
            聊天响应对象
        """
        if history is None:
            history = []
        
        payload = {
            "message": message,
            "history": history,
            "lora_path": lora_path,
            "prompt_choice": prompt_choice,
            "promptFormat_choice": promptFormat_choice,
            "session_id": datetime.now().strftime("%Y%m%d%H%M%S")
        }
        
        endpoint = CHAT_STREAM_ENDPOINT if use_stream else CHAT_ENDPOINT
        
        try:
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API请求失败: {response.text}")
                return {
                    "reply": "API请求失败，请稍后再试", 
                    "history": history,
                    "lora_path": lora_path,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": payload["session_id"]
                }
        except Exception as e:
            print(f"发送聊天请求异常: {e}")
            return {
                "reply": "发送请求时发生错误，请检查网络连接", 
                "history": history,
                "lora_path": lora_path,
                "timestamp": datetime.now().isoformat(),
                "session_id": payload["session_id"]
            }
    
    @staticmethod
    def get_available_models():
        """获取可用的模型列表"""
        try:
            response = requests.get(f"{API_BASE_URL}/model_templates")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取模型列表失败: {response.text}")
                return {"estj": "ESTJ人格模型", "infp": "INFP人格模型", "base_1": "基础模型1", "base_2": "基础模型2"}
        except Exception as e:
            print(f"获取模型列表异常: {e}")
            return {"estj": "ESTJ人格模型", "infp": "INFP人格模型", "base_1": "基础模型1", "base_2": "基础模型2"}
    
    @staticmethod
    def get_prompt_templates():
        """获取提示词模板列表"""
        try:
            response = requests.get(f"{API_BASE_URL}/prompt_templates")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取提示词模板失败: {response.text}")
                return {"assist_estj": "ESTJ助手", "assist_infp": "INFP助手", "base": "基础提示词"}
        except Exception as e:
            print(f"获取提示词模板异常: {e}")
            return {"assist_estj": "ESTJ助手", "assist_infp": "INFP助手", "base": "基础提示词"}
    
    @staticmethod
    def get_promptFormat_templates():
        """获取格式模板列表"""
        try:
            response = requests.get(f"{API_BASE_URL}/promptFormat_templates")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取格式模板失败: {response.text}")
                return {"ordinary": "普通格式", "custom": "自定义格式"}
        except Exception as e:
            print(f"获取格式模板异常: {e}")
            return {"ordinary": "普通格式", "custom": "自定义格式"}

# 测试代码
if __name__ == "__main__":
    # 测试聊天请求
    response = ChatService.send_chat_request("你好，请介绍一下你自己")
    print(f"聊天响应: {response['reply']}")
    
    # 测试获取模型列表
    models = ChatService.get_available_models()
    print(f"可用模型: {models}")
    
    # 测试获取提示词模板
    prompts = ChatService.get_prompt_templates()
    print(f"提示词模板: {prompts}")
    
    # 测试获取格式模板
    formats = ChatService.get_promptFormat_templates()
    print(f"格式模板: {formats}")