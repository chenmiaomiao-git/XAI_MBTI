#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置文件
存储API密钥和其他配置信息
"""

# 导入环境变量
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# API地址配置
API_BASE_URL = os.getenv("API_BASE_URL", "http://114.215.186.130:8717")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
CHAT_STREAM_ENDPOINT = f"{API_BASE_URL}/chat_stream"

# 百度云API配置
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "5Q9Gb2NsS9bQImjeXubMtEz0")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY", "MZJ2qIMcKM7puTqyOAhzeO8aLBJyA2UA")

# 火山引擎API配置
VOLCANO_APP_ID = os.getenv("VOLCANO_APP_ID", "7620208835")
VOLCANO_ACCESS_TOKEN = os.getenv("VOLCANO_ACCESS_TOKEN", "cXjr83Fe-tbZbIXHJbThejcLP_uBz-2r")
VOLCANO_SECRET_KEY = os.getenv("VOLCANO_SECRET_KEY", "3WMZx4sWEMZjhz-vnAaPjeilSC5laG1v")

# 模型配置
DEFAULT_LORA_PATH = "estj"
DEFAULT_PROMPT_CHOICE = "assist_estj"
DEFAULT_FORMAT_CHOICE = "ordinary"

# 语音配置
DEFAULT_VOICE_SPEED = 5  # 语速
DEFAULT_VOICE_PITCH = 5  # 音调
DEFAULT_VOICE_VOLUME = 5  # 音量
DEFAULT_VOICE_PERSON = 0  # 发音人，0女1男3度逍遥4度丫丫

# 界面配置
APP_TITLE = "XAI-MBTI"
APP_SUBTITLE = "智能对话系统 - 基于MBTI人格的AI助手"
APP_THEME = "soft"
APP_PRIMARY_COLOR = "blue"