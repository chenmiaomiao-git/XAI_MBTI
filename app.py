
import gradio as gr
import os
import tempfile
import numpy as np
import soundfile as sf
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import time

# 导入自定义服务模块
from audio_service import AudioService
from chat_service import ChatService

# 这些函数已经移动到AudioService和ChatService模块中

# 发送聊天请求函数已移动到ChatService模块

# 处理音频输入
def handle_audio(audio, history, lora_path, prompt_choice, promptFormat_choice, tts_choice):
    # 调用handle_chat函数处理音频输入
    chatbot_output, chat_history_output, audio_output, _ = handle_chat(None, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, audio)
    return chatbot_output, chat_history_output, audio_output, ""

# 处理聊天提交
def handle_chat(message, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, audio=None):
    # 如果提供了音频，先进行语音识别
    if audio is not None:
        try:
            # 检查音频格式
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
                sf.write(temp_audio.name, audio[1], audio[0])
                temp_audio.close()
            else:
                # 不支持的音频格式
                print(f"音频格式错误: {type(audio)}, 内容: {audio}")
                return history + [("<audio controls style='display:none;'></audio>", "录音格式错误，请重试")], history, None, ""
            
            # 保存一个永久副本用于前端播放
            timestamp = int(time.time())
            audio_filename = f"audio_input_{timestamp}.wav"
            audio_path = os.path.join("static", audio_filename)
            
            # 确保static目录存在
            os.makedirs("static", exist_ok=True)
            
            # 复制音频文件
            import shutil
            shutil.copy(temp_audio.name, audio_path)
            
            # 语音识别
            message = AudioService.asr_recognize(temp_audio.name)
            os.unlink(temp_audio.name)  # 删除临时文件
            
            if not message or message == "语音识别失败" or message == "语音识别请求失败":
                return history + [(f"<audio src='/{audio_path}' controls style='display:none;'></audio>", "语音识别失败，请重试")], history, None, ""
        except Exception as e:
            print(f"处理录音时出错: {e}")
            return history + [("<audio controls style='display:none;'></audio>", f"处理录音时出错: {str(e)}")], history, None, ""
    
    # 如果消息为空，不处理
    if not message or message.strip() == "":
        return history, history, None, ""
    
    # 更新历史记录，添加用户消息
    history_for_display = history.copy()
    
    # 如果是语音输入，添加音频控件（不显示[音频文件]标签）
    if audio is not None:
        # 创建带有音频控件的消息，只显示识别出的文本和隐藏的音频控件
        audio_message = f"{message} <audio src='/{audio_path}' controls style='display:none;'></audio>"
        history_for_display.append((audio_message, None))
    else:
        history_for_display.append((message, None))
    
    # 在用户消息后添加英文回复请求（不显示在前端，但传给模型）
    message_with_english_request = message + " Please answer in English."
    
    # 发送聊天请求
    chat_history = [(h[0], h[1]) for h in history if h[1] is not None]
    response = ChatService.send_chat_request(message_with_english_request, chat_history, lora_path, prompt_choice, promptFormat_choice)
    
    # 获取回复
    reply = response.get("reply", "Sorry, the server did not return a valid reply")
    
    # 根据TTS选择进行语音合成
    if tts_choice == "Standard Voice":
        # 使用标准女声配置
        audio_path = AudioService.tts_synthesize(reply)
    elif tts_choice == "Soft Female Voice":
        # 使用温柔女声配置（度丫丫音色，降低语速，提高音调）
        audio_path = AudioService.tts_synthesize(reply, spd=4, pit=6, per=4)
    elif tts_choice == "Energetic Male Voice":
        # 使用活力男声配置（度逍遥音色，提高语速和音调）
        audio_path = AudioService.tts_synthesize(reply, spd=6, pit=6, per=3)
    else:
        # 默认使用标准配置
        audio_path = AudioService.tts_synthesize(reply)
    
    # 更新显示历史
    history_for_display[-1] = (message, reply)
    
    # 返回更新后的历史、音频路径和空字符串（清空输入框）
    return history_for_display, chat_history + [(message, reply)], audio_path, ""

# 处理文件上传
def handle_upload(file, history, lora_path, prompt_choice, promptFormat_choice, tts_choice):
    if file is None:
        return history, history
    
    # 语音识别
    message = AudioService.asr_recognize(file.name)
    
    if not message or message == "语音识别失败" or message == "语音识别请求失败":
        return history + [("<audio src='" + file.name + "' controls style='display:none;'></audio>", "Speech recognition failed, please try again")], history
    
    # 更新历史记录，添加用户消息（不显示[音频文件]标签）
    history_for_display = history.copy()
    history_for_display.append((message, None))
    
    # 在用户消息后添加英文回复请求（不显示在前端，但传给模型）
    message_with_english_request = message + " Please answer in English."
    
    # 发送聊天请求
    chat_history = [(h[0], h[1]) for h in history if h[1] is not None]
    response = ChatService.send_chat_request(message_with_english_request, chat_history, lora_path, prompt_choice, promptFormat_choice)
    
    # 获取回复
    reply = response.get("reply", "抱歉，服务器没有返回有效回复")
    
    # 根据TTS选择进行语音合成
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
    
    # 更新历史记录（不显示[音频文件]标签）
    history_for_display[-1] = (message, reply)
    
    # 返回更新后的历史、音频路径和空字符串（清空输入框）
    return history_for_display, chat_history + [(message, reply)], audio_path, ""

# 导入配置
from config import APP_TITLE, APP_SUBTITLE, APP_THEME, APP_PRIMARY_COLOR

# 创建Gradio界面
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=APP_PRIMARY_COLOR), css=".container { max-width: 800px; margin: auto; }") as demo:
        gr.Markdown(
            f"""# {APP_TITLE}
            {APP_SUBTITLE}
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=500,
                    avatar_images=("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png", "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"),
                )
                
                with gr.Row():
                    with gr.Column(scale=12):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter message...",
                            container=False
                        )
                    
                    with gr.Column(scale=1, min_width=50):
                        upload_btn = gr.UploadButton("➕", file_types=["audio/*"], size="sm")
                    
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button("Send", variant="primary", size="sm")
                        
                # 添加JavaScript代码，用于音频录制控制
                mic_js = """
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    // 录音状态提示
                    const statusEl = document.getElementById('recording_status');
                    let isRecording = false;
                    let timer = null;
                    
                    // 监听音频输入组件的变化
                    const audioInput = document.getElementById('mic_input');
                    if (!audioInput) return; // 确保音频组件存在
                    
                    // 监听录音按钮点击
                    const recordButtons = audioInput.querySelectorAll('.record, .stop');
                    recordButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            // 检测是否是开始录音按钮
                            if (this.classList.contains('record')) {
                                isRecording = true;
                                // 开始录音：更新计时
                                let seconds = 0;
                                timer = setInterval(() => {
                                    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
                                    const secs = (seconds % 60).toString().padStart(2, '0');
                                    statusEl.innerHTML = `<span style="color: red;">Recording ${mins}:${secs}</span>`;
                                    seconds++;
                                }, 1000);
                            } else if (this.classList.contains('stop')) {
                                // 停止录音
                                isRecording = false;
                                clearInterval(timer);
                                statusEl.innerHTML = '';
                                
                                // 等待一小段时间后自动提交录音
                                setTimeout(() => {
                                    // 查找提交按钮并触发点击
                                    const submitBtn = document.querySelector('button[variant="primary"]');
                                    if (submitBtn) {
                                        submitBtn.click();
                                    } else {
                                        // 备用方法：查找Send按钮
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
                    
                        // 语音消息播放优化
                        setInterval(() => {
                            const audioElements = document.querySelectorAll('#chatbot audio');
                            audioElements.forEach(audio => {
                                if (!audio.parentNode.classList.contains('audio-wrapped')) {
                                    const wrapper = document.createElement('div');
                                    wrapper.className = 'audio-wrapped';
                                    wrapper.style.marginTop = '8px';
                                    audio.parentNode.insertBefore(wrapper, audio);
                                    wrapper.appendChild(audio);
                                    // 强制显示音频控件
                                    audio.style.display = 'block';
                                }
                            });
                        }, 1000);
                    });
                    
                    // 定期检查并设置新添加的语音消息
                    setInterval(function() {
                        try {
                            const messages = document.querySelectorAll('.message:not(.user)');
                            messages.forEach(function(message) {
                                if (!message.hasAttribute('data-audio-processed')) {
                                    const audioElements = message.querySelectorAll('audio');
                                    audioElements.forEach(function(audio) {
                                        if (audio.src) {
                                            // 创建语音条元素
                                            const audioMessage = document.createElement('div');
                                            audioMessage.className = 'audio-message';
                                            audioMessage.setAttribute('data-audio-url', audio.src);
                                            audioMessage.innerHTML = `
                                                <div style="display: flex; align-items: center; background: #e6f7ff; padding: 5px 10px; border-radius: 15px; width: fit-content; cursor: pointer;">
                                                    <span style="margin-right: 5px;">🔊</span>
                                                    <div style="width: 50px; height: 20px; background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMjAiPjxwYXRoIGQ9Ik0wLDEwIHE1LC04IDEwLDAgdDEwLDAgMTAsMCAxMCwwIDEwLDAgMTAsMCAxMCwwIDEwLDAgMTAsMCkiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzAwN2JmZiIgc3Ryb2tlLXdpZHRoPSIyIj48YW5pbWF0ZVRyYW5zZm9ybSBhdHRyaWJ1dGVOYW1lPSJkIiBhdHRyaWJ1dGVUeXBlPSJYTUwiIHR5cGU9InRyYW5zbGF0ZSIgdmFsdWVzPSJNMCwxMCBxNSwtOCAxMCwwIHQxMCwwIDEwLDAgMTAsMCAxMCwwIDEwLDAgMTAsMCAxMCwwIDEwLDApO00wLDEwIHE1LDggMTAsMCB0MTAsLTggMTAsOCAxMCwtOCAxMCw4IDEwLC04IDEwLDggMTAsLTggMTAsOCkiIGR1cj0iMC44cyIgcmVwZWF0Q291bnQ9ImluZGVmaW5pdGUiLz48L3BhdGg+PC9zdmc+') center center no-repeat;"></div>
                                                </div>
                                            `;
                                            
                                            // 将语音条添加到消息中
                                            message.appendChild(audioMessage);
                                            message.setAttribute('data-audio-processed', 'true');
                                        }
                                    });
                                }
                            });
                        } catch (error) {
                            console.error('Failed to process voice message:', error);
                        }
                    }, 1000);
                });
                </script>
                """
                gr.HTML(mic_js)
                
                # 语音输入区域
                with gr.Row():
                    with gr.Column(scale=1):
                        # 语音输入卡片，启用Gradio原生录音功能，只显示录音选项
                        # 注意：确保这个元素ID为mic_input，与JavaScript中的选择器匹配
                        audio_input = gr.Audio(label="Voice Input", elem_id="mic_input", visible=True, autoplay=True, show_download_button=False, show_share_button=False, interactive=True, type="filepath", sources=["microphone"])
                        recording_status = gr.HTML("", elem_id="recording_status", visible=True)
                
                audio_output = gr.Audio(label="Voice Reply", visible=True, autoplay=True)
                
            # 模型和提示选择
            with gr.Column(scale=1):
                lora_path = gr.Radio(
                    ["estj", "infp", "base_1", "base_2"],
                    label="Model Selection",
                    value="estj"
                )
                
                prompt_choice = gr.Radio(
                    ["assist_estj", "assist_infp", "null"],
                    label="Prompt Selection",
                    value="assist_estj"
                )
                
                # 格式选择默认使用ordinary且隐藏前端显示
                promptFormat_choice = gr.Radio(
                    ["ordinary", "custom"],
                    label="Format Selection",
                    value="ordinary",
                    visible=False
                )
                
                # 增加TTS选择
                tts_choice = gr.Radio(
                    ["Standard Voice", "Soft Female Voice", "Energetic Male Voice"],
                    label="Voice Style Selection",
                    value="Standard Voice"
                )
                
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
        
        # 事件处理
        submit_btn.click(
            handle_chat,
            [msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            [chatbot, chat_history, audio_output, msg],
            api_name="submit"
        )
        
        msg.submit(
            handle_chat,
            [msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            [chatbot, chat_history, audio_output, msg]
        )
        
        # 音频输入处理（录音完成后自动处理）
        audio_input.change(
            fn=handle_audio,
            inputs=[audio_input, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="audio"
        )
        
        # 麦克风音频输入处理（录音停止后自动处理）
        audio_input.stop_recording(
            fn=handle_chat,
            # 确保参数顺序与handle_chat函数定义一致
            inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, audio_input],
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="mic_recording"
        )
        
        upload_btn.upload(
            handle_upload,
            [upload_btn, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice],
            [chatbot, chat_history, audio_output, msg],
            api_name="upload"
        )
        
        # 音频输入已经在上面的audio_input.change事件中处理
        
        clear_btn.click(
            lambda: ([], []),
            None,
            [chatbot, chat_history],
            api_name="clear"
        )
        
        # 自定义CSS
        gr.Markdown("""
        <style>
        .gradio-container {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }
        
        #chatbot {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding-left: 10px; /* 为头像留出空间 */
        }
        
        /* 调整聊天容器内部布局 */
        .chatbot {
            padding: 15px;
        }
        
        /* 确保消息容器有足够空间显示头像 */
        .message-wrap {
            position: relative;
            padding-left: 10px;
        }
        
        /* 添加头像和气泡样式 - 使用更精确的选择器 */
        #chatbot .user {
            background-color: #f0f4f9;
            border-radius: 10px;
            padding: 10px 15px;
            margin-bottom: 10px;
            margin-left: 40px !important;
            position: relative !important;
        }
        
        #chatbot .user::before {
            content: '' !important;
            position: absolute !important;
            left: -40px !important;
            top: 0 !important;
            width: 35px !important;
            height: 35px !important;
            background-image: url('https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png') !important;
            background-size: cover !important;
            border-radius: 50% !important;
            z-index: 100 !important;
        }
        
        #chatbot .bot {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 10px 15px;
            margin-bottom: 10px;
            margin-left: 40px !important;
            position: relative !important;
        }
        
        #chatbot .bot::before {
            content: '' !important;
            position: absolute !important;
            left: -40px !important;
            top: 0 !important;
            width: 35px !important;
            height: 35px !important;
            background-image: url('https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png') !important;
            background-size: cover !important;
            border-radius: 50% !important;
            z-index: 100 !important;
        }
        
        /* 优化音频输入组件显示 */
        #mic_input {
            margin-top: 10px;
            border: none;
        }
        
        /* 隐藏Gradio原生的录音按钮（仅保留音频播放控件） */
        #mic_input .controls > button:not(.play-button) {
            display: none !important;
        }
        
        /* 确保录音按钮对用户不可见但JS可访问 */
        #mic_input .record, #mic_input .stop {
            opacity: 0 !important;
            position: absolute !important;
            pointer-events: all !important; /* 允许JS点击事件 */
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
            z-index: -1 !important; /* 确保在视觉上隐藏 */
        }
        
        /* 美化麦克风按钮 */
        #mic_button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 8px 16px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        #mic_button:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        #recording_status {
            margin-top: 5px;
            text-align: center;
            font-size: 12px;
        }
        
        .audio-message {
            margin-top: 10px;
            cursor: pointer;
        }
        
        .audio-message:hover {
            opacity: 0.8;
        }
        
        #mic_button {
            transition: all 0.3s ease;
        }
        
        #mic_button:hover {
            transform: scale(1.1);
        }
        </style>
        """)
    
    return demo

# 主函数
def main():
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='启动MBTI聊天应用')
    parser.add_argument('--server_port', type=int, default=8003, help='服务器端口号')
    args = parser.parse_args()
    
    demo = create_interface()
    # 添加静态文件目录配置，用于访问录音文件
    demo.launch(share=True, server_name="0.0.0.0", server_port=args.server_port, favicon_path=None, allowed_paths=["static"])

if __name__ == "__main__":
    main()