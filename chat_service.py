#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat service module
Handles API interactions, sends chat requests and processes responses
"""

import requests
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Import configuration
from config import API_BASE_URL, CHAT_ENDPOINT, CHAT_STREAM_ENDPOINT, DEFAULT_LORA_PATH, DEFAULT_PROMPT_CHOICE, DEFAULT_FORMAT_CHOICE

class ChatService:
    """Chat service class"""
    
    @staticmethod
    def send_chat_request(message, history=None, lora_path=DEFAULT_LORA_PATH, prompt_choice=DEFAULT_PROMPT_CHOICE, promptFormat_choice=DEFAULT_FORMAT_CHOICE, use_stream=False):
        """Send chat request
        
        Args:
            message: User message
            history: Chat history
            lora_path: Model path, options: "estj", "infp", "base_1", "base_2"
            prompt_choice: Prompt selection, options: "assist_estj", "assist_infp", "null"
            promptFormat_choice: Format selection, options: "ordinary", "custom"
            use_stream: Whether to use streaming API
            
        Returns:
            Chat response object
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
                print(f"API request failed: {response.text}")
                return {
                    "reply": "API request failed, please try again later", 
                    "history": history,
                    "lora_path": lora_path,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": payload["session_id"]
                }
        except Exception as e:
            print(f"Chat request exception: {e}")
            return {
                "reply": "Error occurred while sending request, please check network connection", 
                "history": history,
                "lora_path": lora_path,
                "timestamp": datetime.now().isoformat(),
                "session_id": payload["session_id"]
            }
    
    @staticmethod
    def get_available_models():
        """Get available model list"""
        try:
            response = requests.get(f"{API_BASE_URL}/model_templates")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get model list: {response.text}")
                return {"estj": "ESTJ Personality Model", "infp": "INFP Personality Model", "base_1": "Base Model 1", "base_2": "Base Model 2"}
        except Exception as e:
            print(f"Get model list exception: {e}")
            return {"estj": "ESTJ Personality Model", "infp": "INFP Personality Model", "base_1": "Base Model 1", "base_2": "Base Model 2"}
    
    @staticmethod
    def get_prompt_templates():
        """Get prompt template list"""
        try:
            response = requests.get(f"{API_BASE_URL}/prompt_templates")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get prompt templates: {response.text}")
                return {"assist_estj": "ESTJ Assistant", "assist_infp": "INFP Assistant", "base": "Base Prompt"}
        except Exception as e:
            print(f"Get prompt templates exception: {e}")
            return {"assist_estj": "ESTJ Assistant", "assist_infp": "INFP Assistant", "base": "Base Prompt"}
    
    @staticmethod
    def get_promptFormat_templates():
        """Get format template list"""
        try:
            response = requests.get(f"{API_BASE_URL}/promptFormat_templates")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get format templates: {response.text}")
                return {"ordinary": "Ordinary Format", "custom": "Custom Format"}
        except Exception as e:
            print(f"Get format templates exception: {e}")
            return {"ordinary": "Ordinary Format", "custom": "Custom Format"}

# Test code
if __name__ == "__main__":
    # Test chat request
    response = ChatService.send_chat_request("Hello, please introduce yourself")
    print(f"Chat response: {response['reply']}")
    
    # Test get model list
    models = ChatService.get_available_models()
    print(f"Available models: {models}")
    
    # Test get prompt templates
    prompts = ChatService.get_prompt_templates()
    print(f"Prompt templates: {prompts}")
    
    # Test get format templates
    formats = ChatService.get_promptFormat_templates()
    print(f"Format templates: {formats}")