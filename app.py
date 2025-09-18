
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

# 音频文件检查函数
def check_audio_file(file_path):
    """Check if audio file is valid and return detailed information"""
    if not file_path:
        return {"valid": False, "error": "File path is empty"}
    
    try:
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File does not exist", "path": file_path}
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {"valid": False, "error": "File size is 0", "path": file_path, "size": 0}
        
        # Try to read file header to check if it's a valid MP3 file
        with open(file_path, "rb") as f:
            header = f.read(4)
            # MP3文件可能的头部格式：
            # - ID3标签开头
            # - MPEG帧头开头（各种变体）
            is_mp3 = (
                header.startswith(b"ID3") or  # ID3 tag
                (header[0] == 0xff and (header[1] & 0xe0) == 0xe0)  # MPEG frame header: first 11 bits are all 1
            )
            
            if not is_mp3:
                # For TTS services like Volcano Engine, other valid audio formats may be returned
                # If file size is reasonable, we consider it valid
                if file_size > 1000:  # At least 1KB of audio data
                    print(f"Audio file validation: File header is not standard MP3 format but file size is reasonable, considered valid: {header.hex()}")
                    is_mp3 = True
                else:
                    return {"valid": False, "error": "Not a valid MP3 file", "path": file_path, "size": file_size, "header": header.hex()}
        
        return {"valid": True, "path": file_path, "size": file_size}
    except Exception as e:
        return {"valid": False, "error": str(e), "path": file_path}

# 测试音频文件可访问性函数
def test_audio_accessibility(file_path):
    """Test if audio file can be accessed by web server"""
    if not file_path or not os.path.exists(file_path):
        return {"accessible": False, "error": "File does not exist"}
    
    try:
        # Check file permissions
        file_stat = os.stat(file_path)
        file_mode = oct(file_stat.st_mode)[-3:]
        
        # Check if file is readable
        readable = os.access(file_path, os.R_OK)
        
        # Get absolute path of file
        abs_path = os.path.abspath(file_path)
        
        # Check if file is in temporary directory
        temp_dir = tempfile.gettempdir()
        in_temp_dir = abs_path.startswith(temp_dir)
        
        # Create a symbolic link to static directory to make file accessible via web (for testing only)
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        if not os.path.exists(static_dir):
            os.makedirs(static_dir, exist_ok=True)
        
        link_path = os.path.join(static_dir, os.path.basename(file_path))
        try:
            # If link already exists, delete it first
            if os.path.exists(link_path):
                os.remove(link_path)
            # Create hard link (not symbolic link, as some systems may restrict symbolic links)
            os.link(file_path, link_path)
            link_created = True
        except Exception as e:
            link_created = False
            link_error = str(e)
        
        return {
            "accessible": readable,
            "path": file_path,
            "abs_path": abs_path,
            "permissions": file_mode,
            "in_temp_dir": in_temp_dir,
            "link_created": link_created,
            "link_path": link_path if link_created else None,
            "link_error": link_error if not link_created else None
        }
    except Exception as e:
        return {"accessible": False, "error": str(e), "path": file_path}

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
def handle_audio(audio, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, asr_language_choice):
    """
    处理用户的音频输入，将其转换为文本并生成回复
    
    参数:
        audio: 用户的音频输入，可以是文件路径或音频数据元组
        history: 当前的聊天历史记录
        lora_path: 选择的模型路径
        prompt_choice: 选择的提示模板
        promptFormat_choice: 选择的提示格式
        tts_choice: 选择的语音合成风格
        reply_language_choice: 选择的回复语言
        asr_language_choice: 选择的ASR识别语言
        
    返回:
        chatbot_output: 更新后的聊天界面历史
        chat_history_output: 更新后的聊天历史数据
        audio_output: 生成的回复音频路径
        "": 清空输入框
    """
    # 如果是音频输入，先根据选择的语言进行语音识别
    asr_result = None
    if isinstance(audio, tuple):
        # 如果是录音数据，使用语音识别服务将其转换为文本
        audio_data, sample_rate = audio
        # 根据选择的ASR语言设置语音识别语言
        asr_result = AudioService.asr_recognize_from_numpy(audio_data, sample_rate, language=asr_language_choice)
        audio = None  # 清空音频输入，因为已经转换为文本
    
    # 调用handle_chat函数处理音频输入，传入识别结果作为文本消息参数
    chatbot_output, chat_history_output, audio_output, _ = handle_chat(asr_result, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, asr_language_choice, audio)
    return chatbot_output, chat_history_output, audio_output, ""

# 处理聊天提交函数 - 应用程序的核心功能
def handle_chat(message, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, asr_language_choice=None, audio=None):
    """
    处理用户的聊天输入（文本或音频），生成回复并合成语音
    
    参数:
        message: 用户的文本消息，如果是音频输入则为None
        history: 当前的聊天历史记录
        lora_path: 选择的模型路径（如'estj', 'infp'等）
        prompt_choice: 选择的提示模板（如'assist_estj', 'assist_infp'等）
        promptFormat_choice: 选择的提示格式（如'ordinary', 'custom'）
        tts_choice: 选择的语音合成风格（如'Standard Voice', 'Soft Female Voice'等）
        reply_language_choice: 选择的回复语言
        asr_language_choice: 选择的ASR识别语言（可选）
        audio: 可选的音频输入，可以是文件路径或音频数据元组
        
    返回:
        history_for_display: 更新后的聊天界面历史
        chat_history + [(message, reply)]: 更新后的聊天历史数据
        audio_path: 生成的回复音频路径
        "": 清空输入框
    """
    audio_path = None  # 初始化音频路径变量
    
    # 如果已经有识别结果（从handle_audio传入），直接使用
    if message is not None:
        # 使用已经识别的文本，不需要重新处理音频
        pass
    # 如果提供了音频但没有识别结果，进行语音识别处理
    elif audio is not None:
        try:
            # 检查音频格式并进行相应处理
            if isinstance(audio, str):
                # 如果音频是字符串路径，验证它是文件而不是目录
                if not os.path.isfile(audio):
                    print(f"Invalid audio file path: {audio}")
                    return history + [("<audio controls style='display:none;'></audio>", "Invalid audio file")], history, None, ""
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
                print(f"Audio format error: {type(audio)}, content: {audio}")
                return history + [("<audio controls style='display:none;'></audio>", "Recording format error, please try again")], history, None, ""
            
            # 保存一个永久副本用于前端播放
            timestamp = int(time.time())  # 使用时间戳确保文件名唯一
            audio_filename = f"audio_input_{timestamp}.wav"
            audio_path = os.path.join("static", audio_filename)
            
            # 确保static目录存在，用于存储音频文件
            os.makedirs("static", exist_ok=True)
            
            # 复制音频文件到静态目录
            import shutil
            shutil.copy(temp_audio.name, audio_path)
            
            # 调用语音识别服务将音频转换为文本，使用ASR语言选择
            asr_lang = asr_language_choice if asr_language_choice else reply_language_choice
            message = AudioService.asr_recognize(temp_audio.name, language=asr_lang)
            os.unlink(temp_audio.name)  # 删除临时文件，释放空间
            
            # 检查语音识别结果，如果失败则返回错误信息
            # 检查所有可能的错误返回值
            error_messages = [
                "Speech recognition failed", 
                "Speech recognition request failed", 
                "Speech recognition request timeout",
                "Speech recognition network request failed",
                "Audio format conversion failed",
                "Failed to read audio file",
                "Audio data processing failed",
                "Unable to get Baidu access token"
            ]
            
            if not message or message in error_messages:
                # 确保audio_path已定义，如果未定义则使用空字符串
                audio_display = f"<audio src='/{audio_path}' controls style='display:none;'></audio>" if audio_path else "<audio controls style='display:none;'></audio>"
                error_msg = message if message else "Speech recognition failed"
                return history + [(audio_display, f"{error_msg}, please try again")], history, None, ""
        except Exception as e:
            # 捕获并处理音频处理过程中的任何异常
            print(f"Error processing recording: {e}")
            # 确保audio_path已定义，如果未定义则使用空字符串
            audio_display = f"<audio src='/{audio_path}' controls style='display:none;'></audio>" if audio_path else "<audio controls style='display:none;'></audio>"
            return history + [(audio_display, f"Error processing recording: {str(e)}")], history, None, ""
    
    # 如果消息为空，不进行处理直接返回
    if not message or message.strip() == "":
        return history, history, None, ""
    
    # 创建历史记录的副本用于显示
    history_for_display = history.copy()
    
    # 根据输入类型更新历史记录
    if audio is not None and audio_path is not None:
        # 如果是语音输入，添加带有隐藏音频控件的消息
        audio_message = f"{message} <audio src='/{audio_path}' controls style='display:none;'></audio>"
        history_for_display.append((audio_message, None))
    else:
        # 如果是文本输入，直接添加消息
        history_for_display.append((message, None))
    
    # 根据选择的回复语言添加相应的请求（不显示在前端，但传给模型）
    if reply_language_choice == "Chinese":
        message_with_language_request = message + " "
    elif reply_language_choice == "English":
        message_with_language_request = message + " Please answer in English"
    elif reply_language_choice == "Japanese":
        message_with_language_request = message + " 日本語で答えてください"
    else:
        # 默认使用英文
        message_with_language_request = message + " Please answer in English"
    
    # 准备聊天历史数据，过滤掉没有回复的消息
    chat_history = [(h[0], h[1]) for h in history if h[1] is not None]
    
    # 调用聊天服务发送请求并获取回复
    response = ChatService.send_chat_request(message_with_language_request, chat_history, lora_path, prompt_choice, promptFormat_choice)
    
    # 从响应中获取回复文本，如果没有则使用默认错误消息
    reply = response.get("reply", "Sorry, the server did not return a valid reply")
    
    # 根据选择的TTS风格进行语音合成
    print(f"Audio playback debug: Starting speech synthesis, TTS choice = {tts_choice}, reply language = {reply_language_choice}")
    print(f"Audio playback debug: Reply text length = {len(reply)}")
    
    # 中文语言使用百度云TTS，其他语言使用火山引擎TTS
    if reply_language_choice == "Chinese":
        if tts_choice == "Soft Female Voice - Cancan (Normal)":
            # 中文-使用百度云TTS - 女声，标准语速和音调
            print("Audio playback debug: Using Baidu Cloud TTS - Female voice, standard speed and pitch")
            audio_file_path = AudioService.tts_synthesize(reply, language=reply_language_choice, 
                                                  spd=5, pit=5, per=0)
        elif tts_choice == "Energetic Female Voice - Cancan (Fast)":
            # 中文-使用百度云TTS - 女声，快速语速和高音调
            print("Audio playback debug: Using Baidu Cloud TTS - Female voice, fast speed and high pitch")
            audio_file_path = AudioService.tts_synthesize(reply, language=reply_language_choice, 
                                                  spd=7, pit=6, per=0)
        elif tts_choice == "Professional Foreign Voice - Stefan (Normal)":
            # 中文-使用百度云TTS - 男声，标准语速和音调
            print("Audio playback debug: Using Baidu Cloud TTS - Male voice, standard speed and pitch")
            audio_file_path = AudioService.tts_synthesize(reply, language=reply_language_choice, 
                                                  spd=5, pit=5, per=1)
        elif tts_choice == "Expressive Foreign Voice - Stefan (Fast)":
            # 中文-使用百度云TTS - 男声，快速语速和高音调
            print("Audio playback debug: Using Baidu Cloud TTS - Male voice, fast speed and high pitch")
            audio_file_path = AudioService.tts_synthesize(reply, language=reply_language_choice, 
                                                  spd=7, pit=6, per=1)
        else:
            # 中文-默认使用百度云TTS - 女声，标准语速和音调
            print("Audio playback debug: Using default Baidu Cloud TTS - Female voice, standard speed and pitch")
            audio_file_path = AudioService.tts_synthesize(reply, language=reply_language_choice, 
                                                  spd=5, pit=5, per=0)
    else:
        # 非中文语言使用火山引擎TTS
        if tts_choice == "Soft Female Voice - Cancan (Normal)":
            # 使用火山引擎TTS - 灿灿音色，标准语速和音调
            print("Audio playback debug: Using Volcano Engine TTS - Cancan voice, standard speed and pitch")
            audio_file_path = AudioService.tts_synthesize(reply, tts_engine="volcano", language=reply_language_choice, 
                                                  voice_type="BV700_streaming", speed=1.0, pitch=1.0)
        elif tts_choice == "Energetic Female Voice - Cancan (Fast)":
            # 使用火山引擎TTS - 灿灿音色，快速语速和高音调
            print("Audio playback debug: Using Volcano Engine TTS - Cancan voice, fast speed and high pitch")
            audio_file_path = AudioService.tts_synthesize(reply, tts_engine="volcano", language=reply_language_choice, 
                                                  voice_type="BV700_streaming", speed=1.3, pitch=1.2)
        elif tts_choice == "Professional Foreign Voice - Stefan (Normal)":
            # 使用火山引擎TTS - Stefan音色，标准语速和音调
            print("Audio playback debug: Using Volcano Engine TTS - Stefan voice, standard speed and pitch")
            audio_file_path = AudioService.tts_synthesize(reply, tts_engine="volcano", language=reply_language_choice, 
                                                  voice_type="BV702_streaming", speed=1.0, pitch=1.0)
        elif tts_choice == "Expressive Foreign Voice - Stefan (Fast)":
            # 使用火山引擎TTS - Stefan音色，快速语速和高音调
            print("Audio playback debug: Using Volcano Engine TTS - Stefan voice, fast speed and high pitch")
            audio_file_path = AudioService.tts_synthesize(reply, tts_engine="volcano", language=reply_language_choice, 
                                                  voice_type="BV702_streaming", speed=1.3, pitch=1.2)
        else:
            # 默认使用火山引擎TTS - 灿灿音色，标准语速和音调
            print("Audio playback debug: Using default Volcano Engine TTS - Cancan voice, standard speed and pitch")
            audio_file_path = AudioService.tts_synthesize(reply, tts_engine="volcano", language=reply_language_choice, 
                                                  voice_type="BV700_streaming", speed=1.0, pitch=1.0)
    
    # 将文件路径赋值给audio_path变量，保持与后续代码兼容
    audio_path = audio_file_path
    
    # 更新显示历史，将最后一条用户消息与AI回复配对
    history_for_display[-1] = (message, reply)
    
    # 检查文件是否在静态目录中，如果不是，则复制到静态目录
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
    
    # 直接使用TTS返回的音频文件
    if audio_path and os.path.exists(audio_path):
        try:
            # 获取文件名
            filename = os.path.basename(audio_path)
            # 构建静态目录中的目标路径
            static_audio_path = os.path.join(static_dir, filename)
            
            # 如果文件不在静态目录中，复制到静态目录
            if audio_path != static_audio_path:
                import shutil
                shutil.copy2(audio_path, static_audio_path)
            
            # 构建相对URL路径
            relative_path = os.path.join("static", filename)
            
            # 不在对话框中添加音频标签，只返回音频路径供专门的卡片播放
            history_for_display[-1] = (history_for_display[-1][0], reply)
            
            # 返回结果，使用绝对文件路径而非URL路径
            audio_output_value = static_audio_path
            return history_for_display, chat_history + [(message, reply)], audio_output_value, ""
        except Exception as e:
            print(f"音频处理失败: {e}")
    
    # 如果所有方法都失败，返回没有音频的结果
    return history_for_display, chat_history + [(message, reply)], None, ""

# 处理文件上传函数
def handle_upload(file, history, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, asr_language_choice):
    """
    Handle user uploaded audio files, convert them to text and generate replies
    
    Parameters:
        file: User uploaded audio file object
        history: Current chat history
        lora_path: Selected model path
        prompt_choice: Selected prompt template
        promptFormat_choice: Selected prompt format
        tts_choice: Selected TTS style
        reply_language_choice: Selected reply language
        asr_language_choice: Selected ASR recognition language
        
    Returns:
        history_for_display: Updated chat interface history
        chat_history + [(message, reply)]: Updated chat history data
        audio_path: Generated reply audio path
        "": Clear input box
    """
    # Check if file is empty
    if file is None:
        return history, history
    
    # Call speech recognition service to convert uploaded audio file to text, using ASR language selection
    message = AudioService.asr_recognize(file.name, language=asr_language_choice)
    
    # Check speech recognition result, return error message if failed
    if not message or message == "Speech recognition failed" or message == "Speech recognition request failed" or message == "Unable to get Baidu access token":
        return history + [("<audio src='" + file.name + "' controls style='display:none;'></audio>", "Speech recognition failed, please try again")], history
    
    # Create a copy of history for display
    history_for_display = history.copy()
    # Add user message to history
    history_for_display.append((message, None))
    
    # Add corresponding request based on selected reply language (not displayed in frontend, but passed to model)
    if reply_language_choice == "Chinese":
        message_with_language_request = message + " 请用中文回答"
    elif reply_language_choice == "English":
        message_with_language_request = message + " Please answer in English"
    elif reply_language_choice == "Japanese":
        message_with_language_request = message + " 日本語で答えてください"
    elif reply_language_choice == "French":
        message_with_language_request = message + " Veuillez répondre en français"
    else:
        # Default to English
        message_with_language_request = message + " Please answer in English"
    
    # Prepare chat history data, filter out messages without replies
    chat_history = [(h[0], h[1]) for h in history if h[1] is not None]
    # Call chat service to send request and get reply
    response = ChatService.send_chat_request(message_with_language_request, chat_history, lora_path, prompt_choice, promptFormat_choice)
    
    # Get reply text from response, use default error message if none
    reply = response.get("reply", "Sorry, the server did not return a valid reply")
    
    # 根据选择的TTS风格进行语音合成
    if tts_choice == "Standard Voice":
        # Use standard female voice configuration
        audio_path = AudioService.tts_synthesize(reply, language=reply_language_choice)
    elif tts_choice == "Gentle Female Voice":
        # Use gentle female voice configuration (DuYaYa voice, lower speed, higher pitch)
        audio_path = AudioService.tts_synthesize(reply, spd=4, pit=6, per=4, language=reply_language_choice)
    elif tts_choice == "Energetic Male Voice":
        # Use energetic male voice configuration (DuXiaoyao voice, higher speed and pitch)
        audio_path = AudioService.tts_synthesize(reply, spd=6, pit=6, per=3, language=reply_language_choice)
    elif tts_choice == "Volcano Engine TTS":
        # Use Volcano Engine TTS
        audio_path = AudioService.tts_synthesize(reply, tts_engine="volcano", language=reply_language_choice)
    else:
        # Default to standard configuration
        audio_path = AudioService.tts_synthesize(reply, language=reply_language_choice)
    
    # Update display history, pair the last user message with AI reply
    history_for_display[-1] = (message, reply)
    
    # Process audio file, convert to numpy format for Gradio use
    audio_output_value = None
    if audio_path and os.path.exists(audio_path):
        try:
            import soundfile as sf
            import numpy as np
            data, samplerate = sf.read(audio_path)
            # Ensure data type is float32 and within [-1, 1] range
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            # If data exceeds range, normalize
            if np.max(np.abs(data)) > 1.0:
                data = data / np.max(np.abs(data))
            print(f"Audio playback debug: handle_upload successfully read audio file, sample rate = {samplerate}, shape = {data.shape}, data type = {data.dtype}")
            audio_output_value = (int(samplerate), data)
        except Exception as e:
            print(f"音频播放调试: handle_upload failed to read audio file - {e}")
            try:
                from pydub import AudioSegment
                import numpy as np
                audio = AudioSegment.from_file(audio_path, format="mp3")
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                # If stereo, take one channel
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))[:, 0]
                # Normalize to [-1, 1] range
                if audio.sample_width == 2:  # 16-bit
                    samples = samples / 32768.0
                elif audio.sample_width == 4:  # 32-bit
                    samples = samples / 2147483648.0
                else:
                    samples = samples / np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else samples
                audio_output_value = (int(audio.frame_rate), samples)
                print(f"音频播放调试: handle_upload使用pydub成功读取音频，数据类型 = {samples.dtype}")
            except Exception as e2:
                print(f"音频播放调试: handle_upload所有方法都失败 - {e2}")
                import numpy as np
                audio_output_value = (16000, np.zeros(1000, dtype=np.int16))
    else:
        print(f"音频播放调试: handle_upload音频文件不存在: {audio_path}")
        import numpy as np
        audio_output_value = (16000, np.zeros(1000, dtype=np.int16))
    
    # 返回更新后的历史、聊天记录、音频数据和空字符串（清空输入框）
    return history_for_display, chat_history + [(message, reply)], audio_output_value, ""

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
                    avatar_images=None,  # 移除头像图片，使用CSS自定义头像
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
                        upload_btn = gr.UploadButton("➕", file_types=["audio/*"], size="lg")  # 仅接受音频文件
                    
                    # 发送按钮（小列，固定宽度）
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button("Send", variant="primary", size="lg")  # 主要按钮样式
                        
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
                        # 不播放和显示录制的音频，录制和停止按钮始终可见
                        audio_input = gr.Audio(
                            label="Voice Input",  # 组件标签
                            elem_id="mic_input",  # HTML元素ID
                            visible=True,  # 可见性
                            autoplay=False,  # 不自动播放录制的音频
                            show_download_button=False,  # 隐藏下载按钮
                            show_share_button=False,  # 隐藏分享按钮
                            show_label=True,  # 显示标签
                            waveform_options={"show_controls": False},  # 隐藏音频控制条
                            interactive=True,  # 允许用户交互
                            type="filepath",  # 返回文件路径而非音频数据
                            sources=["microphone"]  # 仅允许麦克风输入
                        )
                        # 录音状态显示区域，由JavaScript更新
                        recording_status = gr.HTML("", elem_id="recording_status", visible=True)
                
                # 简化的语音回复播放组件
                audio_output = gr.Audio(
                    label="Voice Output",  # 移除标签
                    visible=True, 
                    autoplay=True, 
                    type="filepath",  # 使用文件路径类型
                    show_download_button=False,  # 隐藏下载按钮
                    show_share_button=False,  # 隐藏分享按钮
                    elem_id="voice_reply_audio"  # 添加元素ID便于调试
                )
                # 添加JavaScript以在页面上显示调试信息
                gr.HTML("""
                <script>
                function updateDebugInfo(message) {
                    const debugElement = document.getElementById('audio_debug_info');
                    if (debugElement) {
                        const timestamp = new Date().toLocaleTimeString();
                        debugElement.innerHTML += `<div>[${timestamp}] ${message}</div>`;
                        // 保持最新的消息可见
                        debugElement.scrollTop = debugElement.scrollHeight;
                    }
                }
                
                // 监听音频元素事件
                document.addEventListener('DOMNodeInserted', function(e) {
                    if (e.target.tagName === 'AUDIO') {
                        updateDebugInfo(`发现新的音频元素: ${e.target.src.substring(0, 50)}...`);
                        
                        e.target.addEventListener('play', function() {
                            updateDebugInfo(`音频开始播放: ${this.src.substring(0, 50)}...`);
                        });
                        
                        e.target.addEventListener('error', function() {
                            updateDebugInfo(`音频播放错误: ${this.src.substring(0, 50)}... 错误代码: ${this.error ? this.error.code : '未知'}`);
                        });
                        
                        e.target.addEventListener('canplay', function() {
                            updateDebugInfo(`音频可以播放: ${this.src.substring(0, 50)}...`);
                        });
                    }
                });
                </script>
                """)
                
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
                    [
                        "Soft Female Voice - Cancan (Normal)",  # 灿灿音色-标准语速
                        "Energetic Female Voice - Cancan (Fast)",  # 灿灿音色-快速语调高
                        "Professional Foreign Voice - Stefan (Normal)",  # Stefan音色-标准语速
                        "Expressive Foreign Voice - Stefan (Fast)"  # Stefan音色-快速语调高
                    ],
                    label="Voice Style Selection",  # 标签改为英文
                    value="Soft Female Voice - Cancan (Normal)"  # 默认使用灿灿标准语速
                )
                
                # 回复语言选择 - 控制AI回复的语言
                reply_language_choice = gr.Radio(
                    ["Chinese", "English", "Japanese"],  # 不同回复语言(英文显示)，去掉法语选项
                    label="Reply Language Selection",  # 标签
                    value="English"  # 默认使用英文回复
                )
                
                # ASR语言选择 - 控制语音识别的语言
                asr_language_choice = gr.Radio(
                    ["Chinese", "English"],  # ASR只支持中文和英文
                    label="ASR Language Selection",  # 标签
                    value="English"  # 默认使用英文识别
                )
                
                # 清除聊天按钮 - 重置对话历史
                clear_btn = gr.Button("Clear Chat")
            
            # 存储聊天历史的状态
            chat_history = gr.State([])
            
            # 确保音频输入与聊天函数关联（在定义所有变量后）
            # audio_input.change(fn=handle_chat, inputs=[audio_input, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice], outputs=[chatbot, chat_history, audio_output, audio_input])
            
            # # 添加JavaScript代码，用于浏览器本地TTS
            # js_code = """
            #     function browserTTS(text) {
            #         if ('speechSynthesis' in window) {
            #             // 创建语音合成实例
            #             const utterance = new SpeechSynthesisUtterance(text);
            #             // 设置语音参数
            #             utterance.rate = 1.0;  // 语速
            #             utterance.pitch = 1.0; // Pitch
            #             utterance.volume = 1.0; // Volume
            #             // Get Chinese voice
            #             const voices = window.speechSynthesis.getVoices();
            #             const chineseVoice = voices.find(voice => voice.lang.includes('zh'));
            #             if (chineseVoice) {
            #                 utterance.voice = chineseVoice;
            #             }
            #             // Play voice
            #             window.speechSynthesis.speak(utterance);
            #             return true;
            #         }
            #         return false;
            #     }
                
            #     // Listen for chat reply updates and add avatars
            #     document.addEventListener('DOMContentLoaded', function() {
            #         console.log('音频播放调试(JS): DOM加载完成，开始监听音频元素');
                    
            #         // 监听音频元素加载
            #         document.addEventListener('play', function(e) {
            #             if (e.target.tagName === 'AUDIO') {
            #                 console.log('音频播放调试(JS): 音频开始播放', e.target.src);
            #             }
            #         }, true);
                    
            #         document.addEventListener('error', function(e) {
            #             if (e.target.tagName === 'AUDIO') {
            #                 console.log('音频播放调试(JS): 音频播放错误', e.target.src, e.target.error);
            #             }
            #         }, true);
                    
            #         // 定期检查音频元素
            #         setInterval(function() {
            #             const audioElements = document.querySelectorAll('audio');
            #             console.log(`音频播放调试(JS): 当前页面有 ${audioElements.length} 个音频元素`);
            #             audioElements.forEach((audio, index) => {
            #                 console.log(`音频播放调试(JS): 音频元素 ${index+1}:`, {
            #                     src: audio.src,
            #                     readyState: audio.readyState,
            #                     paused: audio.paused,
            #                     ended: audio.ended,
            #                     error: audio.error
            #                 });
            #             });
            #         }, 5000);
                    
            #         // Regularly check for new chat messages
            #         setInterval(function() {
            #             // TTS processing
            #             const ttsChoice = document.querySelector('input[name="tts_choice"]:checked');
            #             if (ttsChoice && ttsChoice.value === "Browser Local TTS") {
            #                 console.log('音频播放调试(JS): 使用浏览器本地TTS');
            #                 const chatMessages = document.querySelectorAll('.message:not(.user)');
            #                 const lastMessage = chatMessages[chatMessages.length - 1];
            #                 if (lastMessage && !lastMessage.hasAttribute('data-tts-processed')) {
            #                     const messageText = lastMessage.textContent.trim();
            #                     if (messageText) {
            #                         console.log('音频播放调试(JS): 尝试使用浏览器TTS播放消息', messageText.substring(0, 50) + '...');
            #                         browserTTS(messageText);
            #                         lastMessage.setAttribute('data-tts-processed', 'true');
            #                     }
            #                 }
            #             }
                        
            #             // Add avatars to all messages
            #             addAvatarsToMessages();
            #         }, 1000);
                    
            #         // Function to add avatars
            #         function addAvatarsToMessages() {
            #             // Get all message elements
            #             const userMessages = document.querySelectorAll('.message.user');
            #             const botMessages = document.querySelectorAll('.message.bot');
                        
            #             // 为用户消息添加头像
            #             userMessages.forEach(msg => {
            #                 if (!msg.hasAttribute('data-avatar-added')) {
            #                     const avatar = document.createElement('div');
            #                     avatar.className = 'user-avatar';
            #                     avatar.style.cssText = 'position:absolute;left:-40px;top:0;width:35px;height:35px;background-image:url("https://cdn.jsdelivir.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png");background-size:cover;border-radius:50%;';
                                
            #                     // 确保消息容器是相对定位
            #                     msg.style.position = 'relative';
            #                     msg.style.marginLeft = '40px';
            #                     msg.style.marginBottom = '10px';
                                
            #                     // 插入头像
            #                     msg.insertBefore(avatar, msg.firstChild);
            #                     msg.setAttribute('data-avatar-added', 'true');
            #                 }
            #             });
                        
            #             // 为机器人消息添加头像
            #             botMessages.forEach(msg => {
            #                 if (!msg.hasAttribute('data-avatar-added')) {
            #                     const avatar = document.createElement('div');
            #                     avatar.className = 'bot-avatar';
            #                     avatar.style.cssText = 'position:absolute;left:-40px;top:0;width:35px;height:35px;background-image:url("https://cdn.jsdelivir.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png");background-size:cover;border-radius:50%;';
                                
            #                     // 确保消息容器是相对定位
            #                     msg.style.position = 'relative';
            #                     msg.style.marginLeft = '40px';
            #                     msg.style.marginBottom = '10px';
                                
            #                     // 插入头像
            #                     msg.insertBefore(avatar, msg.firstChild);
            #                     msg.setAttribute('data-avatar-added', 'true');
            #                 }
            #             });
            #         }
            #     });
            #     """
            # gr.HTML("<script>" + js_code + "</script>", visible=False)
        
        # 事件处理部分 - 定义UI组件的交互行为
        # 提交按钮点击事件 - 处理文本输入并生成回复
        submit_btn.click(
            fn=handle_chat,  # 处理函数
            # 输入参数列表
            inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="submit"  # API端点名称
        )
        
        # 文本框回车提交事件 - 与提交按钮功能相同
        msg.submit(
            fn=handle_chat,  # 处理函数
            # 输入参数列表
            inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg]
        )
        
        # 音频输入处理事件（录音完成后自动处理）
        audio_input.change(
            fn=handle_audio,  # 处理函数
            # 输入参数列表
            inputs=[audio_input, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, asr_language_choice],
            # 输出参数列表
            outputs=[chatbot, chat_history, audio_output, msg],
            api_name="audio"  # API端点名称
        )
        
        # 麦克风录音停止事件 - 录音结束后自动处理语音输入
        # audio_input.stop_recording(
        #     fn=handle_chat,  # 处理函数
        #     # 确保参数顺序与handle_chat函数定义一致
        #     inputs=[msg, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, audio_input],
        #     # 输出参数列表
        #     outputs=[chatbot, chat_history, audio_output, msg],
        #     api_name="mic_recording"  # API端点名称
        # )
        
        # 文件上传事件 - 处理上传的音频文件
        upload_btn.upload(
            fn=handle_upload,  # 处理函数
            # 输入参数列表
            inputs=[upload_btn, chatbot, lora_path, prompt_choice, promptFormat_choice, tts_choice, reply_language_choice, asr_language_choice],
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
        
        /* 强制覆盖所有可能的Gradio头像选择器 - 用户头像 */
        #chatbot .user::before,
        .chatbot .user::before,
        [data-testid="chatbot"] .user::before,
        .message-wrap .user::before,
        .message.user::before,
        .message-wrap .message.user::before,
        .chatbot .message.user::before {
            content: '👤' !important;            /* 用户表情符号 */
            position: absolute !important;      /* 绝对定位 */
            left: -40px !important;             /* 左侧位置 */
            top: 50% !important;                /* 垂直居中位置 */
            transform: translateY(-50%) !important; /* 垂直居中对齐 */
            width: 35px !important;             /* 头像宽度 */
            height: 35px !important;            /* 头像高度 */
            background-color: #4CAF50 !important; /* 绿色背景 */
            background-image: none !important;  /* 强制移除背景图片 */
            color: white !important;            /* 白色表情符号 */
            display: flex !important;           /* 弹性布局 */
            align-items: center !important;     /* 垂直居中 */
            justify-content: center !important; /* 水平居中 */
            font-size: 18px !important;         /* 表情符号大小 */
            border-radius: 50% !important;      /* 圆形头像 */
            z-index: 9999 !important;           /* 最高层级 */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; /* 轻微阴影 */
        }
        
        /* 强制覆盖所有可能的Gradio头像选择器 - 机器人头像 */
        #chatbot .bot::before,
        .chatbot .bot::before,
        [data-testid="chatbot"] .bot::before,
        .message-wrap .bot::before,
        .message.bot::before,
        .message-wrap .message.bot::before,
        .chatbot .message.bot::before {
            content: '🤖' !important;            /* 机器人表情符号 */
            position: absolute !important;      /* 绝对定位 */
            left: -40px !important;             /* 左侧位置 */
            top: 50% !important;                /* 垂直居中位置 */
            transform: translateY(-50%) !important; /* 垂直居中对齐 */
            width: 35px !important;             /* 头像宽度 */
            height: 35px !important;            /* 头像高度 */
            background-color: #2196F3 !important; /* 蓝色背景 */
            background-image: none !important;  /* 强制移除背景图片 */
            color: white !important;            /* 白色表情符号 */
            display: flex !important;           /* 弹性布局 */
            align-items: center !important;     /* 垂直居中 */
            justify-content: center !important; /* 水平居中 */
            font-size: 18px !important;         /* 表情符号大小 */
            border-radius: 50% !important;      /* 圆形头像 */
            z-index: 9999 !important;           /* 最高层级 */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; /* 轻微阴影 */
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
        
        /* 隐藏麦克风选择下拉菜单 */
        .mic-wrap select, 
        .mic-wrap .wrap-inner > div:first-child,
        #mic_input .wrap-inner > div:first-child,
        #mic_input select {
            display: none !important;
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