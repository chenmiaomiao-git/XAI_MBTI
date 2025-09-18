#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio processing service module
Provides speech recognition (ASR) and text-to-speech (TTS) functionality
Based on Baidu Cloud API and Volcano Engine API implementation
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

# Import configuration
from config import (BAIDU_API_KEY, BAIDU_SECRET_KEY, DEFAULT_VOICE_SPEED, DEFAULT_VOICE_PITCH, 
                   DEFAULT_VOICE_VOLUME, DEFAULT_VOICE_PERSON, VOLCANO_APP_ID, VOLCANO_ACCESS_TOKEN, VOLCANO_SECRET_KEY)

class AudioService:
    """Audio processing service class"""
    
    @staticmethod
    def get_baidu_access_token():
        """Get Baidu Cloud access_token"""
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
            print(f"Failed to get Baidu access_token: {response.text}")
            return None
    
    @staticmethod
    def convert_to_wav(audio_file_path):
        """Convert audio to WAV format"""
        try:
            # Get file extension
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            
            # If already in wav format with correct sample rate, return directly
            if file_ext == ".wav":
                try:
                    data, samplerate = sf.read(audio_file_path)
                    if samplerate == 16000:
                        return audio_file_path
                except Exception as e:
                    print(f"Error checking WAV file: {e}")
                    # Continue trying to convert
            
            # Create temporary file
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Convert using pydub
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav_file, format="wav")
            
            return temp_wav_file
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            # Return None instead of original file path, let caller know conversion failed
            return None
    
    @classmethod
    def asr_recognize(cls, audio_file_path, language="English"):
        """Speech recognition (ASR)
        
        Args:
            audio_file_path: Audio file path
            language: Language selection, supports "Chinese", "English"
        """
        token = cls.get_baidu_access_token()
        if not token:
            return "Unable to get Baidu access token"
        
        # Convert to wav format and set correct sample rate
        temp_wav_file = cls.convert_to_wav(audio_file_path)
        if temp_wav_file is None:
            return "Audio format conversion failed"
            
        try:
            with open(temp_wav_file, "rb") as f:
                audio_data = f.read()
        except Exception as e:
            print(f"Failed to read audio file: {e}")
            return "Failed to read audio file"
        
        # If it's a temporarily created file, delete after use
        if temp_wav_file != audio_file_path:
            try:
                os.unlink(temp_wav_file)
            except Exception as e:
                print(f"Failed to delete temporary file: {e}")
        
        # Set different speech recognition models based on language selection
        dev_pid = 1737  # Default English model
        if language == "Chinese":
            dev_pid = 1537  # Mandarin model
        elif language == "English":
            dev_pid = 1737  # English model
        
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
                    print(f"Speech recognition failed: {result}")
                    return "Speech recognition failed"
            else:
                print(f"Speech recognition request failed: {response.text}")
                return "Speech recognition request failed"
        except requests.exceptions.Timeout:
            print("Speech recognition request timeout")
            return "Speech recognition request timeout"
        except requests.exceptions.RequestException as e:
            print(f"Speech recognition network request exception: {e}")
            return "Speech recognition network request failed"
    
    @classmethod
    def asr_recognize_from_numpy(cls, audio_data, sample_rate, language="English"):
        """Speech recognition from NumPy array
        
        Args:
            audio_data: Audio data NumPy array
            sample_rate: Sample rate
            language: Language selection, supports "Chinese", "English", "Japanese", "French"
        """
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_file.close()
            
            # Perform speech recognition
            result = cls.asr_recognize(temp_file.name, language=language)
            
            return result
        except Exception as e:
            print(f"Error processing NumPy audio data: {e}")
            return "Audio data processing failed"
        finally:
            # Ensure temporary file is deleted
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                 print(f"Failed to delete temporary file: {e}")
    
    @classmethod
    def tts_synthesize(cls, text, spd=DEFAULT_VOICE_SPEED, pit=DEFAULT_VOICE_PITCH, vol=DEFAULT_VOICE_VOLUME, per=DEFAULT_VOICE_PERSON, tts_engine="baidu", language="English", voice_type=None, speed=1.0, pitch=1.0):
        """Text-to-speech synthesis (TTS)
        
        Args:
            text: Text to synthesize
            spd: Speech speed (Baidu Cloud parameter)
            pit: Pitch (Baidu Cloud parameter)
            vol: Volume (Baidu Cloud parameter)
            per: Speaker (Baidu Cloud parameter)
            tts_engine: TTS engine, supports "baidu" and "volcano"
            language: Language selection, supports "Chinese", "English", "Japanese"
            voice_type: Volcano Engine voice type, such as "BV700_streaming"(Cancan), "BV701_streaming"(Qingcang), "BV702_streaming"(Stefan)
            speed: Speech speed (Volcano Engine parameter, range: 0.5~2.0)
            pitch: Pitch (Volcano Engine parameter, range: 0.5~2.0)
        """
        if tts_engine == "volcano":
            return cls.volcano_tts_synthesize(text, language=language, voice_type=voice_type, speed=speed, pitch=pitch)
        
        # Default to Baidu Cloud TTS
        token = cls.get_baidu_access_token()
        if not token:
            return None
        
        # Set different language codes based on language selection
        lan = "en"  # Default English
        if language == "Chinese":
            lan = "zh"
        elif language == "English":
            lan = "en"
        # Baidu API doesn't support Japanese and French, use English as fallback
        # Note: Baidu API doesn't support jp as language code
        elif language == "Japanese" or language == "French":
            lan = "en"
        
        url = "https://tsn.baidu.com/text2audio"
        params = {
            "tex": text,
            "tok": token,
            "cuid": "123456PYTHON",
            "ctp": 1,
            "lan": lan,
            "spd": spd,  # Speech speed
            "pit": pit,  # Pitch
            "vol": vol,  # Volume
            "per": per   # Speaker, 0=female 1=male 3=DuXiaoyao 4=DuYaya
        }
        
        response = requests.post(url, params=params)
        if response.headers.get("Content-Type", "").startswith("audio/"):
            # Create temporary file to save audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            print(f"Speech synthesis failed: {response.text}")
            return None
            
    @classmethod
    def volcano_tts_synthesize(cls, text, language="English", voice_type=None, speed=1.0, pitch=1.0):
        """Use Volcano Engine for text-to-speech synthesis
        
        Args:
            text: Text to synthesize
            language: Language selection, supports "Chinese", "English", "Japanese" (does not support "French")
            voice_type: Voice type, if None, automatically select based on language
            speed: Speech speed, range: 0.5~2.0
            pitch: Pitch, range: 0.5~2.0
        """
        import json
        import time
        import uuid
        import hmac
        import hashlib
        
        # Volcano Engine TTS API address - include token in URL
        url = f"https://openspeech.bytedance.com/api/v1/tts?token={VOLCANO_ACCESS_TOKEN}"
        
        # Generate unique request ID
        req_id = str(uuid.uuid4())
        
        # Set different language codes based on language selection
        lang_code = "en"  # Default English
        if language == "Chinese":
            lang_code = "zh"  # Volcano Engine uses zh instead of cn
        elif language == "English":
            lang_code = "en"
        elif language == "Japanese":
            lang_code = "ja"  # Volcano Engine uses ja instead of jp
        else:
            # Other languages not supported, use English as fallback
            lang_code = "en"
            
        # Select appropriate voice based on language
        if voice_type is None:
            if language == "Chinese":
                voice_type = "BV700_streaming"  # Cancan
            elif language == "English":
                voice_type = "BV702_streaming"  # Stefan
            elif language == "Japanese":
                voice_type = "BV700_streaming"  # Cancan (supports Japanese)
            else:
                voice_type = "BV700_streaming"  # Default to Cancan
        
        # Ensure text is not empty and format processing
        if not text or text.strip() == "":
            print("Volcano Engine TTS error: Text content is empty")
            return None
        
        # Process text to ensure it meets Volcano Engine requirements
        processed_text = text.strip()
        
        # Remove special characters that might cause issues
        import re
        # Simplify text processing, remove control characters and special characters
        processed_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', processed_text)  # Remove control characters and invisible characters
            
        # Prepare request body - according to Volcano Engine latest API documentation format
        request_data = {
            "app": {
                "appid": VOLCANO_APP_ID,  # Use appid from config file
                "token": "",  # Use signature authentication, token is empty
                "cluster": "volcano_tts"  # Use correct cluster value according to documentation
            },
            "user": {
                "uid": "user_" + str(int(time.time()))  # Dynamically generate user ID
            },
            "audio": {
                "voice_type": voice_type,  # Use passed voice parameter
                "encoding": "mp3",
                "rate": 24000,
                "speed_ratio": speed,  # Use passed speed parameter, range: 0.2~3.0
                "volume_ratio": 1.0,  # Volume, range: 0.1~3.0
                "pitch_ratio": pitch,   # Use passed pitch parameter, range: 0.1~3.0
                "language": lang_code  # Use language code set based on language parameter
            },
            "request": {
                "reqid": req_id,
                "text": processed_text,  # Use processed text
                "text_type": "plain",
                "operation": "query",
                "silence_duration": 125
            }
        }
        
        # Special handling for Chinese, use Baidu Cloud TTS instead of Volcano Engine
        if lang_code == "zh":
            # Use Baidu Cloud TTS for Chinese synthesis
            return AudioService.tts_synthesize(text, spd=int(speed*5), pit=int(pitch*5), vol=15, per=0, tts_engine="baidu", language="Chinese")
        
        # Set request headers - use correct Bearer Token format (note semicolon instead of space)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer;{VOLCANO_ACCESS_TOKEN}"
        }
        
        # Generate timestamp at function start to ensure same timestamp is used throughout processing
        timestamp = int(time.time())
        # Use project's static directory to save audio files
        static_dir = os.path.join("/data/LLM-SFT/CCYTest/XAI_MBTI/static")
        os.makedirs(static_dir, exist_ok=True)
        audio_file_path = os.path.join(static_dir, f"tts_output_{timestamp}.mp3")
        print(f"Saving audio data to: {audio_file_path}")
        
        # Print request information for debugging
        print(f"Volcano Engine TTS request: language={lang_code}, text length={len(processed_text)}")
        # print(f"Request data: {json.dumps(request_data, ensure_ascii=False)}")
        # print(f"Request headers: {headers}")
        
        try:
            response = requests.post(url, headers=headers, json=request_data)
            print(f"Volcano Engine response status code: {response.status_code}")
            # Don't print response headers to avoid long data segments
            
            if response.status_code == 200:
                # Check response content type
                content_type = response.headers.get("Content-Type", "")
                print(f"Response content type: {content_type}")
                
                # First try to parse as JSON
                try:
                    result = response.json()
                    # Don't print complete response JSON to avoid long data segments
                    print(f"Volcano Engine response JSON status code: {result.get('code')}")
                    
                    # Use pre-generated timestamp and file path
                    
                    # Check if successful
                    if result.get("code") == 0 or result.get("code") == 3000:  # Volcano Engine success status codes
                        # Get base64 encoded audio data from data field
                        audio_base64 = result.get("data", "")
                        if audio_base64:
                            # Decode base64 data
                            audio_data = base64.b64decode(audio_base64)
                            
                            # Save directly as audio file without additional processing
                            try:
                                print(f"Saving audio data to: {audio_file_path}")
                                with open(audio_file_path, "wb") as f:
                                    f.write(audio_data)
                                if os.path.exists(audio_file_path):
                                    return audio_file_path
                                else:
                                    print(f"Error: File save failed: {audio_file_path}")
                            except Exception as save_error:
                                print(f"Error occurred while saving audio file: {save_error}")
                        else:
                            # Try to get audio data from payload field
                            payload = result.get("payload", {})
                            if isinstance(payload, dict) and "audio_data" in payload:
                                audio_base64 = payload.get("audio_data", "")
                                if audio_base64:
                                    # Decode base64 data and save directly
                                    audio_data = base64.b64decode(audio_base64)
                                    try:
                                        with open(audio_file_path, "wb") as f:
                                            f.write(audio_data)
                                        if os.path.exists(audio_file_path):
                                            return audio_file_path
                                        else:
                                            print(f"Error: payload file save failed: {audio_file_path}")
                                    except Exception as save_error:
                                        print(f"Error occurred while saving payload audio file: {save_error}")
                            print("No audio data found in Volcano Engine response")
                    else:
                        print(f"Volcano Engine TTS failed: {result.get('message', 'Unknown error')}")
                except ValueError as e:
                    print(f"Failed to parse JSON response: {e}")
                    # If not JSON format, might be direct audio data return
                    if content_type.startswith("audio/"):
                        # Save as audio file - use pre-generated file path
                        with open(audio_file_path, "wb") as f:
                            f.write(response.content)
                        print(f"Saved direct audio content to static file: {audio_file_path}")
                        return audio_file_path
            else:
                print(f"Volcano Engine TTS request failed: {response.text}")
        except Exception as e:
            print(f"Volcano Engine TTS exception: {e}")
            
        return None

# Test code
if __name__ == "__main__":
    # Test TTS
    audio_file = AudioService.tts_synthesize("This is a test for Baidu speech synthesis interface.")
    if audio_file:
        print(f"Speech synthesis successful, file saved at: {audio_file}")
    else:
        print("Speech synthesis failed")
    
    # If audio file exists, test ASR
    if audio_file:
        text = AudioService.asr_recognize(audio_file)
        print(f"Speech recognition result: {text}")