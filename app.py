
# ============================================================================
# 导入必要的库和模块
# ============================================================================

# 导入Gradio库，用于创建Web界面
import gradio as gr
# 导入操作系统相关功能
import os
# 导入临时文件处理模块，用于处理音频文件
import tempfile
# 导入数值计算库
import numpy as np
# 导入音频文件处理库
import soundfile as sf
# 导入日期时间处理模块
from datetime import datetime
# 导入类型注解支持
from typing import List, Tuple, Optional, Dict, Any
# 导入时间处理模块
import time

# 导入自定义服务模块
# AudioService: 处理语音识别和语音合成功能
from audio_service import AudioService
# ChatService: 处理聊天请求和响应功能
from chat_service import ChatService

# 注意：以下功能已经移动到相应的服务模块中
# - 语音识别和合成功能已移动到AudioService模块
# - 发送聊天请求功能已移动到ChatService模块

# ============================================================================
# 音频处理和聊天功能函数
# ============================================================================

# 处理音频输入函数
def handle_audio(audio, history, lora_path, prompt_choice, promptFormat_choice, tts_choice):
    """
    处理用户的音频输入，将其转换为文本并生成回复
    
    参数:
        audio: 用户的音频输入，可以是文件路径或音频数据元组
        history: 当前的聊天历史记录
        lora_path: 选择的模型路径
        prompt_choice: 选择的提示模板
        promptFormat_choice: 选择的提示格式
        tts_choice: 选择的语音合成风格
        
    返回:
        chatbot_output: 更新后的聊天界面历史
        chat_history_output: 更新后的聊天历史数据
        audio_output: 生成的回复音频路径
        "": 清空输入框
    """
    # 调用handle_chat函数处理音频输入，传入None作为文本消息参数
    chatbot_output, chat_history_output, audio_output, _ = handle_chat(None, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, audio)
    return chatbot_output, chat_history_output, audio_output, ""

# 处理聊天提交函数 - 应用程序的核心功能
def handle_chat(message, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, audio=None):
    """
    处理用户的聊天输入（文本或音频），生成回复并合成语音
    
    参数:
        message: 用户的文本消息，如果是音频输入则为None
        history: 当前的聊天历史记录
        lora_path: 选择的模型路径（如'estj', 'infp'等）
        prompt_choice: 选择的提示模板（如'assist_estj', 'assist_infp'等）
        promptFormat_choice: 选择的提示格式（如'ordinary', 'custom'）
        tts_choice: 选择的语音合成风格（如'Standard Voice', 'Soft Female Voice'等）
        audio: 可选的音频输入，可以是文件路径或音频数据元组
        
    返回:
        history_for_display: 更新后的聊天界面历史
        chat_history + [(message, reply)]: 更新后的聊天历史数据
        audio_path: 生成的回复音频路径
        "": 清空输入框
    """
    # 如果提供了音频，先进行语音识别处理
    if audio is not None:
        try:
            # 检查音频格式并进行相应处理
            if isinstance(audio, str):
                # 如果音频是字符串路径，直接使用该路径
                audio_file_path = audio
                # 创建临时文件用于后续处理
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                # 复制音频文件到临时文件
                import shutil
                shutil.copy(audio_file_path, temp_audio.name)
                temp_audio.close()
            elif isinstance(audio, tuple) and len(audio) >= 2:
                # 如果音频是元组格式(采样率, 数据)，保存到临时文件
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(temp_audio.name, audio[1], audio[0])  # 写入采样率和音频数据
                temp_audio.close()
            else:
                # 不支持的音频格式，返回错误信息
                print(f"音频格式错误: {type(audio)}, 内容: {audio}")
                return history + [("<audio controls style='display:none;'></audio>", "录音格式错误，请重试")], history, None, ""
            
            # 保存一个永久副本用于前端播放
            timestamp = int(time.time())  # 使用时间戳确保文件名唯一
            audio_filename = f"audio_input_{timestamp}.wav"
            audio_path = os.path.join("static", audio_filename)
            
            # 确保static目录存在，用于存储音频文件
            os.makedirs("static", exist_ok=True)
            
            # 复制音频文件到静态目录
            import shutil
            shutil.copy(temp_audio.name, audio_path)
            
            # 调用语音识别服务将音频转换为文本
            message = AudioService.asr_recognize(temp_audio.name)
            os.unlink(temp_audio.name)  # 删除临时文件，释放空间
            
            # 检查语音识别结果，如果失败则返回错误信息
            if not message or message == "语音识别失败" or message == "语音识别请求失败":
                return history + [(f"<audio src='/{audio_path}' controls style='display:none;'></audio>", "语音识别失败，请重试")], history, None, ""
        except Exception as e:
            # 捕获并处理音频处理过程中的任何异常
            print(f"处理录音时出错: {e}")
            return history + [("<audio controls style='display:none;'></audio>", f"处理录音时出错: {str(e)}")], history, None, ""
    
    # 如果消息为空，不进行处理直接返回
    if not message or message.strip() == "":
        return history, history, None, ""
    
    # 创建历史记录的副本用于显示
    history_for_display = history.copy()
    
    # 根据输入类型更新历史记录
    if audio is not None:
        # 如果是语音输入，添加带有隐藏音频控件的消息
        audio_message = f"{message} <audio src='/{audio_path}' controls style='display:none;'></audio>"
        history_for_display.append((audio_message, None))
    else:
        # 如果是文本输入，直接添加消息
        history_for_display.append((message, None))
    
    # 在用户消息后添加英文回复请求（不显示在前端，但传给模型）
    message_with_english_request = message + " Please answer in English."
    
    # 准备聊天历史数据，过滤掉没有回复的消息
    chat_history = [(h[0], h[1]) for h in history if h[1] is not None]
    
    # 调用聊天服务发送请求并获取回复
    response = ChatService.send_chat_request(message_with_english_request, chat_history, lora_path, prompt_choice, promptFormat_choice)
    
    # 从响应中获取回复文本，如果没有则使用默认错误消息
    reply = response.get("reply", "Sorry, the server did not return a valid reply")
    
    # 根据选择的TTS风格进行语音合成
    if tts_choice == "Standard Voice":
        # 使用标准女声配置
        audio_path = AudioService.tts_synthesize(reply)
    elif tts_choice == "Soft Female Voice":
        # 使用温柔女声配置（度丫丫音色，降低语速，提高音调）
        audio_path = AudioService.tts_synthesize(reply, spd=4, pit=6, per=4)
    elif tts_choice == "Energetic Male Voice":
        # 使用活力男声配置（度逍遥音色，提高语速和音调）
        audio_path = AudioService.tts_synthesize(reply, spd=6, pit=6, per=3)
    elif tts_choice == "Volcano Engine TTS":
        # 使用火山引擎TTS
        audio_path = AudioService.tts_synthesize(reply, tts_engine="volcano")
    else:
        # 默认使用标准配置
        audio_path = AudioService.tts_synthesize(reply)
    
    # 更新显示历史，将最后一条用户消息与AI回复配对
    history_for_display[-1] = (message, reply)
    
    # 返回更新后的历史、聊天记录、音频路径和空字符串（清空输入框）
    return history_for_display, chat_history + [(message, reply)], audio_path, ""

# 处理文件上传函数
def handle_upload(file, history, lora_path, prompt_choice, promptFormat_choice, tts_choice):
    """
    处理用户上传的音频文件，将其转换为文本并生成回复
    
    参数:
        file: 用户上传的音频文件对象
        history: 当前的聊天历史记录
        lora_path: 选择的模型路径
        prompt_choice: 选择的提示模板
        promptFormat_choice: 选择的提示格式
        tts_choice: 选择的语音合成风格
        
    返回:
        history_for_display: 更新后的聊天界面历史
        chat_history + [(message, reply)]: 更新后的聊天历史数据
        audio_path: 生成的回复音频路径
        "": 清空输入框
    """
    # 检查文件是否为空
    if file is None:
        return history, history
    
    # 调用语音识别服务将上传的音频文件转换为文本
    message = AudioService.asr_recognize(file.name)
    
    # 检查语音识别结果，如果失败则返回错误信息
    if not message or message == "语音识别失败" or message == "语音识别请求失败":
        return history + [("<audio src='" + file.name + "' controls style='display:none;'></audio>", "Speech recognition failed, please try again")], history
    
    # 创建历史记录的副本用于显示
    history_for_display = history.copy()
    # 添加用户消息到历史记录
    history_for_display.append((message, None))
    
    # 在用户消息后添加英文回复请求（不显示在前端，但传给模型）
    message_with_english_request = message + " Please answer in English."
    
    # 准备聊天历史数据，过滤掉没有回复的消息
    chat_history = [(h[0], h[1]) for h in history if h[1] is not None]
    # 调用聊天服务发送请求并获取回复
    response = ChatService.send_chat_request(message_with_english_request, chat_history, lora_path, prompt_choice, promptFormat_choice)
    
    # 从响应中获取回复文本，如果没有则使用默认错误消息
    reply = response.get("reply", "抱歉，服务器没有返回有效回复")
    
    # 根据选择的TTS风格进行语音合成
    # 注意：这里的TTS选择标签与handle_chat函数中的不同，使用中文标签
    if tts_choice == "标准语音":
        # 使用标准女声配置
        audio_path = AudioService.tts_synthesize(reply)
    elif tts_choice == "温柔女声":
        # 使用温柔女声配置（度丫丫音色，降低语速，提高音调）
        audio_path = AudioService.tts_synthesize(reply, spd=4, pit=6, per=4)
    elif tts_choice == "活力男声":
        # 使用活力男声配置（度逍遥音色，提高语速和音调）
        audio_path = AudioService.tts_synthesize(reply, spd=6, pit=6, per=3)
    else:
        # 默认使用标准配置
        audio_path = AudioService.tts_synthesize(reply)
    
    # 更新显示历史，将最后一条用户消息与AI回复配对
    history_for_display[-1] = (message, reply)
    
    # 返回更新后的历史、聊天记录、音频路径和空字符串（清空输入框）
    return history_for_display, chat_history + [(message, reply)], audio_path, ""

# ============================================================================
# UI界面创建部分
# ============================================================================

# 导入应用程序配置
from config import APP_TITLE, APP_SUBTITLE, APP_THEME, APP_PRIMARY_COLOR

# 创建Gradio界面函数
def create_interface():
    """
    创建应用程序的Gradio Web界面
    
    配置整个应用的UI布局、组件和事件处理
    
    返回:
        demo: 配置好的Gradio Blocks界面对象
    """
    # 使用Gradio Blocks创建自定义界面，设置主题和CSS样式
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=APP_PRIMARY_COLOR), css=".container { max-width: 800px; margin: auto; }") as demo:
        # 添加应用标题和副标题
        gr.Markdown(
            f"""# {APP_TITLE}
            {APP_SUBTITLE}
            """
        )
        
        # 创建主界面布局，使用行和列组织组件
        with gr.Row():
            # 左侧主要聊天区域（占4/5宽度）
            with gr.Column(scale=4):
                # 聊天机器人组件，用于显示对话历史
                chatbot = gr.Chatbot(
                    [],  # 初始为空列表
                    elem_id="chatbot",  # HTML元素ID，用于CSS和JavaScript选择
                    height=500,  # 设置高度
                    avatar_images=("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png", "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"),  # 用户和机器人头像
                )
                
                # 消息输入区域布局
                with gr.Row():
                    # 文本输入框（占大部分宽度）
                    with gr.Column(scale=12):
                        msg = gr.Textbox(
                            show_label=False,  # 不显示标签
                            placeholder="Enter message...",  # 占位文本
                            container=False  # 不使用容器样式
                        )
                    
                    # 上传按钮（小列，固定宽度）
                    with gr.Column(scale=1, min_width=50):
                        upload_btn = gr.UploadButton("➕", file_types=["audio/*"], size="sm")  # 仅接受音频文件
                    
                    # 发送按钮（小列，固定宽度）
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button("Send", variant="primary", size="sm")  # 主要按钮样式
                        
                # 添加JavaScript代码，用于音频录制控制和UI增强
                mic_js = """
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    // 录音状态提示元素，用于显示录音时间
                    const statusEl = document.getElementById('recording_status');
                    let isRecording = false;
                    let timer = null;
                    
                    // 获取音频输入组件
                    const audioInput = document.getElementById('mic_input');
                    if (!audioInput) return; // 确保音频组件存在，否则退出
                    
                    // 监听录音按钮点击事件（开始和停止按钮）
                    const recordButtons = audioInput.querySelectorAll('.record, .stop');
                    recordButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            // 检测是否是开始录音按钮
                            if (this.classList.contains('record')) {
                                // 开始录音状态
                                isRecording = true;
                                // 开始录音：创建计时器显示录音时长
                                let seconds = 0;
                                timer = setInterval(() => {
                                    // 格式化分钟和秒钟，确保两位数显示
                                    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
                                    const secs = (seconds % 60).toString().padStart(2, '0');
                                    // 更新录音状态显示
                                    statusEl.innerHTML = `<span style="color: red;">Recording ${mins}:${secs}</span>`;
                                    seconds++;
                                }, 1000);
                            } else if (this.classList.contains('stop')) {
                                // 停止录音状态
                                isRecording = false;
                                // 清除计时器并重置状态显示
                                clearInterval(timer);
                                statusEl.innerHTML = '';
                                
                                // 等待一小段时间后自动提交录音（给系统处理音频的时间）
                                setTimeout(() => {
                                    // 查找提交按钮并触发点击事件
                                    const submitBtn = document.querySelector('button[variant="primary"]');
                                    if (submitBtn) {
                                        submitBtn.click();
                                    } else {
                                        // 备用方法：查找包含'Send'文本的按钮
                                        const allButtons = document.querySelectorAll('button');
                                        for (const btn of allButtons) {
                                            if (btn.textContent.includes('Send')) {
                                                btn.click();
                                                break;
                                            }
                                        }
                                    }
                                }, 1000);
                            }
                        });
                    });
                    }
                    
                    // 语音消息播放优化：定期检查并美化音频元素
                    setInterval(() => {
                        // 查找聊天区域中的所有音频元素
                        const audioElements = document.querySelectorAll('#chatbot audio');
                        audioElements.forEach(audio => {
                            // 为未处理的音频元素添加包装器
                            if (!audio.parentNode.classList.contains('audio-wrapped')) {
                                // 创建包装div并设置样式
                                const wrapper = document.createElement('div');
                                wrapper.className = 'audio-wrapped';
                                wrapper.style.marginTop = '8px';
                                // 插入包装器并移动音频元素
                                audio.parentNode.insertBefore(wrapper, audio);
                                wrapper.appendChild(audio);
                                // 强制显示音频控件（覆盖可能的隐藏设置）
                                audio.style.display = 'block';
                            }
                        });
                    }, 1000);
                    });
                    
                    // 定期检查并设置新添加的语音消息，添加自定义UI
                    setInterval(function() {
                        try {
                            // 查找所有非用户消息（机器人回复）
                            const messages = document.querySelectorAll('.message:not(.user)');
                            messages.forEach(function(message) {
                                // 只处理尚未处理过的消息
                                if (!message.hasAttribute('data-audio-processed')) {
                                    // 查找消息中的音频元素
                                    const audioElements = message.querySelectorAll('audio');
                                    audioElements.forEach(function(audio) {
                                        // 确保音频有源
                                        if (audio.src) {
                                            // 创建美化的语音条元素
                                            const audioMessage = document.createElement('div');
                                            audioMessage.className = 'audio-message';
                                            audioMessage.setAttribute('data-audio-url', audio.src);
                                            // 设置语音条的HTML内容，包含波形动画
                                            audioMessage.innerHTML = `
                                                <div style="display: flex; align-items: center; background: #e6f7ff; padding: 5px 10px; border-radius: 15px; width: fit-content; cursor: pointer;">
                                                    <span style="margin-right: 5px;">🔊</span>
                                                    <div style="width: 50px; height: 20px; background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMjAiPjxwYXRoIGQ9Ik0wLDEwIHE1LC04IDEwLDAgdDEwLDAgMTAsMCAxMCwwIDEwLDAgMTAsMCAxMCwwIDEwLDAgMTAsMCkiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzAwN2JmZiIgc3Ryb2tlLXdpZHRoPSIyIj48YW5pbWF0ZVRyYW5zZm9ybSBhdHRyaWJ1dGVOYW1lPSJkIiBhdHRyaWJ1dGVUeXBlPSJYTUwiIHR5cGU9InRyYW5zbGF0ZSIgdmFsdWVzPSJNMCwxMCBxNSwtOCAxMCwwIHQxMCwwIDEwLDAgMTAsMCAxMCwwIDEwLDAgMTAsMCAxMCwwIDEwLDApO00wLDEwIHE1LDggMTAsMCB0MTAsLTggMTAsOCAxMCwtOCAxMCw4IDEwLC04IDEwLDggMTAsLTggMTAsOCkiIGR1cj0iMC44cyIgcmVwZWF0Q291bnQ9ImluZGVmaW5pdGUiLz48L3BhdGg+PC9zdmc+') center center no-repeat;"></div>
                                                </div>
                                            `;
                                            
                                            // 将语音条添加到消息中
                                            message.appendChild(audioMessage);
                                            // 标记消息已处理，避免重复处理
                                            message.setAttribute('data-audio-processed', 'true');
                                        }
                                    });
                                }
                            });
                        } catch (error) {
                            // 捕获并记录处理过程中的错误
                            console.error('Failed to process voice message:', error);
                        }
                    }, 1000); // 每秒检查一次
                });
                </script>
                """
                # 插入JavaScript代码到页面
                gr.HTML(mic_js)
                
                # 语音输入区域
                with gr.Row():
                    with gr.Column(scale=1):
                        # 语音输入组件，配置Gradio原生录音功能
                        # 设置元素ID为mic_input，与JavaScript中的选择器匹配
                        audio_input = gr.Audio(
                            label="Voice Input",  # 组件标签
                            elem_id="mic_input",  # HTML元素ID
                            visible=True,  # 可见性
                            autoplay=True,  # 自动播放
                            show_download_button=False,  # 隐藏下载按钮
                            show_share_button=False,  # 隐藏分享按钮
                            interactive=True,  # 允许用户交互
                            type="filepath",  # 返回文件路径而非音频数据
                            sources=["microphone"]  # 仅允许麦克风输入
                        )
                        # 录音状态显示区域，由JavaScript更新
                        recording_status = gr.HTML("", elem_id="recording_status", visible=True)
                
                # 语音回复播放组件
                audio_output = gr.Audio(label="Voice Reply", visible=True, autoplay=True)  # 自动播放AI回复的语音
                
            # 模型和提示选择区域 - 配置AI个性和语音风格
            with gr.Column(scale=1):
                # MBTI性格模型选择 - 决定AI回复的性格特征
                lora_path = gr.Radio(
                    ["estj", "infp", "base_1", "base_2"],  # 不同MBTI性格类型模型
                    label="Model Selection",  # 标签
                    value="estj"  # 默认选择ESTJ性格模型
                )
                
                # 提示模板选择 - 决定AI回复的风格和语气
                prompt_choice = gr.Radio(
                    ["assist_estj", "assist_infp", "null"],  # 不同提示模板
                    label="Prompt Selection",  # 标签
                    value="assist_estj"  # 默认使用ESTJ助手提示
                )
                
                # 格式选择 - 控制提示格式，默认隐藏
                promptFormat_choice = gr.Radio(
                    ["ordinary", "custom"],  # 普通格式或自定义格式
                    label="Format Selection",  # 标签
                    value="ordinary",  # 默认使用普通格式
                    visible=False  # 在UI中隐藏此选项
                )
                
                # 语音合成风格选择 - 控制AI回复的语音特征
                tts_choice = gr.Radio(
                    ["Standard Voice", "Soft Female Voice", "Energetic Male Voice", "Volcano Engine TTS"],  # 不同语音风格
                    label="Voice Style Selection",  # 标签
                    value="Standard Voice"  # 默认使用标准语音
                )
                
                # 清除聊天按钮 - 重置对话历史
                clear_btn = gr.Button("Clear Chat")
            
            # 存储聊天历史的状态
            chat_history = gr.State([])
            
            # 确保音频输入与聊天函数关联（在定义所有变量后）
            audio_input.change(fn=handle_chat, inputs=[audio_input, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice], outputs=[chatbot, chat_history, audio_output, audio_input])
            
            # 添加JavaScript代码，用于浏览器本地TTS
            js_code = """
                function browserTTS(text) {
                    if ('speechSynthesis' in window) {
                        // 创建语音合成实例
                        const utterance = new SpeechSynthesisUtterance(text);
                        // 设置语音参数
                        utterance.rate = 1.0;  // 语速
                        utterance.pitch = 1.0; // Pitch
                        utterance.volume = 1.0; // Volume
                        // Get Chinese voice
                        const voices = window.speechSynthesis.getVoices();
                        const chineseVoice = voices.find(voice => voice.lang.includes('zh'));
                        if (chineseVoice) {
                            utterance.voice = chineseVoice;
                        }
                        // Play voice
                        window.speechSynthesis.speak(utterance);
                        return true;
                    }
                    return false;
                }
                
                // Listen for chat reply updates and add avatars
                document.addEventListener('DOMContentLoaded', function() {
                    // Regularly check for new chat messages
                    setInterval(function() {
                        // TTS processing
                        const ttsChoice = document.querySelector('input[name="tts_choice"]:checked');
                        if (ttsChoice && ttsChoice.value === "Browser Local TTS") {
                            const chatMessages = document.querySelectorAll('.message:not(.user)');
                            const lastMessage = chatMessages[chatMessages.length - 1];
                            if (lastMessage && !lastMessage.hasAttribute('data-tts-processed')) {
                                const messageText = lastMessage.textContent.trim();
                                if (messageText) {
                                    browserTTS(messageText);
                                    lastMessage.setAttribute('data-tts-processed', 'true');
                                }
                            }
                        }
                        
                        // Add avatars to all messages
                        addAvatarsToMessages();
                    }, 1000);
                    
                    // Function to add avatars
                    function addAvatarsToMessages() {
                        // Get all message elements
                        const userMessages = document.querySelectorAll('.message.user');
                        const botMessages = document.querySelectorAll('.message.bot');
                        
                        // 为用户消息添加头像
                        userMessages.forEach(msg => {
                            if (!msg.hasAttribute('data-avatar-added')) {
                                const avatar = document.createElement('div');
                                avatar.className = 'user-avatar';
                                avatar.style.cssText = 'position:absolute;left:-40px;top:0;width:35px;height:35px;background-image:url("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png");background-size:cover;border-radius:50%;';
                                
                                // 确保消息容器是相对定位
                                msg.style.position = 'relative';
                                msg.style.marginLeft = '40px';
                                msg.style.marginBottom = '10px';
                                
                                // 插入头像
                                msg.insertBefore(avatar, msg.firstChild);
                                msg.setAttribute('data-avatar-added', 'true');
                            }
                        });
                        
                        // 为机器人消息添加头像
                        botMessages.forEach(msg => {
                            if (!msg.hasAttribute('data-avatar-added')) {
                                const avatar = document.createElement('div');
                                avatar.className = 'bot-avatar';
                                avatar.style.cssText = 'position:absolute;left:-40px;top:0;width:35px;height:35px;background-image:url("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png");background-size:cover;border-radius:50%;';
                                
                                // 确保消息容器是相对定位
                                msg.style.position = 'relative';
                                msg.style.marginLeft = '40px';
                                msg.style.marginBottom = '10px';
                                
                                // 插入头像
                                msg.insertBefore(avatar, msg.firstChild);
                                msg.setAttribute('data-avatar-added', 'true');
                            }
                        });
                    }
                });
                """
            gr.HTML("<script>" + js_code + "</script>", visible=False)
        
        # 事件处理部分 - 定义UI组件的交互行为
        # 提交按钮点击事件 - 处理文本输入并生成回复
        submit_btn.click(
            fn=handle_chat,  # 处理函数
            # 输入参数列表
            inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="submit"  # API端点名称
        )
        
        # 文本框回车提交事件 - 与提交按钮功能相同
        msg.submit(
            fn=handle_chat,  # 处理函数
            # 输入参数列表
            inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg]
        )
        
        # 音频输入处理事件（录音完成后自动处理）
        audio_input.change(
            fn=handle_audio,  # 处理函数
            # 输入参数列表
            inputs=[audio_input, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="audio"  # API端点名称
        )
        
        # 麦克风录音停止事件 - 录音结束后自动处理语音输入
        audio_input.stop_recording(
            fn=handle_chat,  # 处理函数
            # 确保参数顺序与handle_chat函数定义一致
            inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, audio_input],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="mic_recording"  # API端点名称
        )
        
        # 文件上传事件 - 处理上传的音频文件
        upload_btn.upload(
            fn=handle_upload,  # 处理函数
            # 输入参数列表
            inputs=[upload_btn, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="upload"  # API端点名称
        )
        
        # 音频输入已经在上面的audio_input.change事件中处理
        
        # 清除聊天按钮事件 - 重置对话历史
        clear_btn.click(
            fn=lambda: ([], []),  # 返回两个空列表，分别用于清空chatbot和chat_history
            inputs=None,  # 不需要输入参数
            outputs=[chatbot, chat_history],  # 清空这两个组件
            api_name="clear"  # API端点名称
        )
        
        # 自定义CSS - 设置应用程序的视觉样式和布局
        gr.Markdown("""
        <style>
        /* 设置整体容器字体 - 使用现代无衬线字体系列 */
        .gradio-container {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }
        
        /* 聊天框主容器样式 - 添加圆角和阴影效果 */
        #chatbot {
            border-radius: 10px;                /* 圆角边框 */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* 轻微阴影效果 */
            padding-left: 10px;                /* 为头像留出空间 */
        }
        
        /* 调整聊天容器内部布局 - 增加内边距 */
        .chatbot {
            padding: 15px;                      /* 内部填充 */
        }
        
        /* 确保消息容器有足够空间显示头像 */
        .message-wrap {
            position: relative;                 /* 相对定位 */
            padding-left: 10px;                /* 左侧填充 */
        }
        
        /* 用户消息气泡样式 - 使用浅蓝色背景 */
        #chatbot .user {
            background-color: #f0f4f9;          /* 浅蓝色背景 */
            border-radius: 10px;                /* 圆角边框 */
            padding: 10px 15px;                 /* 内部填充 */
            margin-bottom: 10px;                /* 底部外边距 */
            margin-left: 40px !important;       /* 左侧留出头像空间 */
            position: relative !important;      /* 相对定位 */
        }
        
        /* 用户头像样式 - 使用表情符号作为头像 */
        #chatbot .user::before {
            content: '' !important;              /* 伪元素内容 */
            position: absolute !important;      /* 绝对定位 */
            left: -40px !important;             /* 左侧位置 */
            top: 0 !important;                  /* 顶部位置 */
            width: 35px !important;             /* 头像宽度 */
            height: 35px !important;            /* 头像高度 */
            background-image: url('https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png') !important; /* 用户头像图片 */
            background-size: cover !important;   /* 背景图片覆盖 */
            border-radius: 50% !important;      /* 圆形头像 */
            z-index: 100 !important;            /* 层级 */
        }
        
        /* 机器人消息气泡样式 - 使用浅蓝色背景 */
        #chatbot .bot {
            background-color: #e6f7ff;          /* 浅蓝色背景 */
            border-radius: 10px;                /* 圆角边框 */
            padding: 10px 15px;                 /* 内部填充 */
            margin-bottom: 10px;                /* 底部外边距 */
            margin-left: 40px !important;       /* 左侧留出头像空间 */
            position: relative !important;      /* 相对定位 */
        }
        
        /* 机器人头像样式 - 使用机器人表情符号作为头像 */
        #chatbot .bot::before {
            content: '' !important;              /* 伪元素内容 */
            position: absolute !important;      /* 绝对定位 */
            left: -40px !important;             /* 左侧位置 */
            top: 0 !important;                  /* 顶部位置 */
            width: 35px !important;             /* 头像宽度 */
            height: 35px !important;            /* 头像高度 */
            background-image: url('https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png') !important; /* 机器人头像图片 */
            background-size: cover !important;   /* 背景图片覆盖 */
            border-radius: 50% !important;      /* 圆形头像 */
            z-index: 100 !important;            /* 层级 */
        }
        
        /* 优化音频输入组件显示 - 移除边框并添加上边距 */
        #mic_input {
            margin-top: 10px;                   /* 顶部外边距 */
            border: none;                       /* 移除边框 */
        }
        
        /* 隐藏Gradio原生的录音按钮（仅保留音频播放控件） */
        #mic_input .controls > button:not(.play-button) {
            display: none !important;           /* 隐藏非播放按钮 */
        }
        
        /* 确保录音按钮对用户不可见但JS可访问 - 用于自定义录音控制 */
        #mic_input .record, #mic_input .stop {
            opacity: 0 !important;              /* 透明度为0 */
            position: absolute !important;      /* 绝对定位 */
            pointer-events: all !important;     /* 允许JS点击事件 */
            width: 1px !important;              /* 最小宽度 */
            height: 1px !important;             /* 最小高度 */
            overflow: hidden !important;        /* 隐藏溢出部分 */
            z-index: -1 !important;             /* 负层级确保在视觉上隐藏 */
        }
        
        /* 美化麦克风按钮 - 使用绿色背景和圆角 */
        #mic_button {
            background-color: #4CAF50;          /* 绿色背景 */
            color: white;                       /* 白色文字 */
            border: none;                       /* 无边框 */
            border-radius: 20px;                /* 圆角边框 */
            padding: 8px 16px;                  /* 内部填充 */
            cursor: pointer;                    /* 鼠标指针样式 */
            width: 100%;                        /* 宽度100% */
            transition: all 0.3s ease;          /* 平滑过渡效果 */
        }
        
        /* 麦克风按钮悬停效果 - 轻微放大并添加阴影 */
        #mic_button:hover {
            transform: scale(1.05);              /* 放大效果 */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* 轻微阴影 */
        }
        
        /* 录音状态显示样式 - 居中显示录音时间 */
        #recording_status {
            margin-top: 5px;                    /* 顶部外边距 */
            text-align: center;                 /* 文字居中 */
            font-size: 12px;                    /* 小字体 */
        }
        
        /* 音频消息样式 - 添加上边距和鼠标指针样式 */
        .audio-message {
            margin-top: 10px;                   /* 顶部外边距 */
            cursor: pointer;                    /* 鼠标指针样式 */
        }
        
        /* 音频消息悬停效果 - 轻微透明 */
        .audio-message:hover {
            opacity: 0.8;                        /* 轻微透明 */
        }
        
        /* 麦克风按钮过渡效果 - 平滑动画 */
        #mic_button {
            transition: all 0.3s ease;          /* 平滑过渡效果 */
        }
        
        /* 麦克风按钮悬停放大效果 */
        #mic_button:hover {
            transform: scale(1.1);              /* 放大效果 */
        }
        </style>
        """)
    
    return demo

# 主函数 - 应用程序入口点
def main():
    import argparse
    
    # 创建命令行参数解析器 - 用于配置服务器启动参数
    parser = argparse.ArgumentParser(description='启动MBTI聊天应用')
    # 添加端口参数，默认为8003
    parser.add_argument('--server_port', type=int, default=8003, help='服务器端口号')
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建Gradio界面
    demo = create_interface()
    # 启动Web服务器
    # share=True: 生成可公开访问的临时URL
    # server_name="0.0.0.0": 监听所有网络接口
    # server_port: 使用指定的端口号
    # allowed_paths=["static"]: 允许访问static目录中的文件（用于音频文件）
    demo.launch(share=True, server_name="0.0.0.0", server_port=args.server_port, favicon_path=None, allowed_paths=["static"])

# 程序入口点 - 当脚本直接运行时执行main函数
if __name__ == "__main__":
    main()