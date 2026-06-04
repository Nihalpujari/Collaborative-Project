"""
Audio generation module - handles 4 audio models (Text-to-Speech)
- Camb AI
- Deepgram
- Cartesia AI
- Fish Audio
"""

import requests
import base64
import time
from pathlib import Path
from config import AUDIO_MODELS, OUTPUT_DIRS

class AudioGenerator:
    def __init__(self):
        self.models = AUDIO_MODELS
        self.output_dir = Path(OUTPUT_DIRS["audio"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_with_camb_ai(self, text):
        """Generate speech using Camb AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['camb_ai']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "model": self.models['camb_ai']['model_name'],
                "voice": "professional",
                "language": "en",
                "speed": 1.0
            }
            response = requests.post(
                self.models['camb_ai']['api_url'],
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('audio'):
                return response.content
            elif 'audio' in response.json():
                return base64.b64decode(response.json()['audio'])
            return None
        except Exception as e:
            print(f"Camb AI error: {e}")
            return None
    
    def generate_with_deepgram(self, text):
        """Generate speech using Deepgram API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['deepgram']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "model": self.models['deepgram']['model_name'],
                "encoding": "linear16",
                "container": "wav"
            }
            response = requests.post(
                self.models['deepgram']['api_url'],
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('audio'):
                return response.content
            return None
        except Exception as e:
            print(f"Deepgram error: {e}")
            return None
    
    def generate_with_cartesia(self, text):
        """Generate speech using Cartesia API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['cartesia']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "transcript": text,
                "model_id": self.models['cartesia']['model_name'],
                "voice": {
                    "mode": "id",
                    "id": "default"
                },
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000
                }
            }
            response = requests.post(
                self.models['cartesia']['api_url'],
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('audio'):
                return response.content
            return None
        except Exception as e:
            print(f"Cartesia error: {e}")
            return None
    
    def generate_with_fish_audio(self, text):
        """Generate speech using Fish Audio API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['fish_audio']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "model": self.models['fish_audio']['model_name'],
                "voice_id": "default",
                "language": "en"
            }
            response = requests.post(
                self.models['fish_audio']['api_url'],
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('audio'):
                return response.content
            elif 'audio' in response.json():
                # May return base64 encoded audio
                audio_data = response.json()['audio']
                if isinstance(audio_data, str):
                    return base64.b64decode(audio_data)
                return audio_data
            return None
        except Exception as e:
            print(f"Fish Audio error: {e}")
            return None
    
    def save_audio(self, audio_data, model_name, prompt_id):
        """Save audio to file"""
        try:
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            file_path = model_dir / f"prompt_{prompt_id}.wav"
            
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            return str(file_path)
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None
    
    def generate_all(self, text, prompt_id):
        """Generate audio from all 4 models for a single text"""
        results = {}
        
        print(f"  Generating audio for prompt {prompt_id}...")
        
        # Camb AI
        print(f"    - Camb AI...", end="", flush=True)
        audio = self.generate_with_camb_ai(text)
        if audio:
            self.save_audio(audio, "camb_ai", prompt_id)
            results["camb_ai"] = True
            print(" ✓")
        else:
            results["camb_ai"] = False
            print(" ✗")
        time.sleep(1)
        
        # Deepgram
        print(f"    - Deepgram...", end="", flush=True)
        audio = self.generate_with_deepgram(text)
        if audio:
            self.save_audio(audio, "deepgram", prompt_id)
            results["deepgram"] = True
            print(" ✓")
        else:
            results["deepgram"] = False
            print(" ✗")
        time.sleep(1)
        
        # Cartesia
        print(f"    - Cartesia...", end="", flush=True)
        audio = self.generate_with_cartesia(text)
        if audio:
            self.save_audio(audio, "cartesia", prompt_id)
            results["cartesia"] = True
            print(" ✓")
        else:
            results["cartesia"] = False
            print(" ✗")
        time.sleep(1)
        
        # Fish Audio
        print(f"    - Fish Audio...", end="", flush=True)
        audio = self.generate_with_fish_audio(text)
        if audio:
            self.save_audio(audio, "fish_audio", prompt_id)
            results["fish_audio"] = True
            print(" ✓")
        else:
            results["fish_audio"] = False
            print(" ✗")
        time.sleep(1)
        
        return results

def main():
    """Test the audio generator"""
    generator = AudioGenerator()
    
    test_text = "An elegant owl librarian wearing reading glasses, sitting in a grand library."
    print(f"\nTesting audio generation with: '{test_text}'")
    results = generator.generate_all(test_text, 1)
    
    for model, success in results.items():
        status = "✓ Saved" if success else "✗ Failed"
        print(f"{model}: {status}")

if __name__ == "__main__":
    main()
