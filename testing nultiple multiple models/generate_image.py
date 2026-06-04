"""
Image generation with CORRECT API endpoints
- Black Forest Labs FLUX (api.bfl.ai with x-key header)
- Stability AI (Bearer token)
- Hugging Face Inference API (Bearer token)
- Pixazo v2 (api.pixai.art v2 with Bearer token)
"""

import requests
import base64
import time
from pathlib import Path
from config import IMAGE_MODELS, OUTPUT_DIRS

class ImageGenerator:
    def __init__(self):
        self.models = IMAGE_MODELS
        self.output_dir = Path(OUTPUT_DIRS["image"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_with_black_forest_labs(self, prompt):
        """Black Forest Labs FLUX - Uses x-key header, api.bfl.ai endpoint"""
        try:
            headers = {
                "x-key": self.models['black_forest_labs']['api_key'],
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": prompt,
                "width": 1024,
                "height": 1024
            }
            
            # Submit request
            response = requests.post(
                "https://api.bfl.ai/v1/flux-2-pro-preview",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                return None
            
            polling_url = result.get("polling_url")
            if not polling_url:
                return None
            
            # Poll for results
            for _ in range(120):
                time.sleep(0.5)
                poll_response = requests.get(
                    polling_url,
                    headers={"x-key": self.models['black_forest_labs']['api_key']},
                    timeout=10
                )
                poll_response.raise_for_status()
                poll_result = poll_response.json()
                
                if poll_result.get("status") == "Ready":
                    image_url = poll_result.get("result", {}).get("sample")
                    if image_url:
                        img_response = requests.get(image_url, timeout=30)
                        return img_response.content
                    return None
                elif poll_result.get("status") in ["Error", "Failed"]:
                    return None
            
            return None
        except Exception as e:
            print(f"BFL error: {e}")
            return None
    
    def generate_with_stability_ai(self, prompt):
        """Stability AI with 1024x1024 dimensions"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['stability_ai']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7.5,
                "height": 1024,
                "width": 1024,
                "steps": 30,
                "samples": 1
            }
            response = requests.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            if response.json().get('artifacts'):
                return base64.b64decode(response.json()['artifacts'][0]['base64'])
            return None
        except Exception as e:
            print(f"Stability AI error: {e}")
            return None
    
    def generate_with_huggingface(self, prompt):
        """Hugging Face Inference API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['huggingface']['api_key']}"
            }
            
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            
            payload = {"inputs": prompt}
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('image'):
                return response.content
            
            return None
        except Exception as e:
            print(f"Hugging Face error: {e}")
            return None
    
    def generate_with_pixazo(self, prompt):
        """Pixazo v2 API - https://api.pixai.art/v2/image/create"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['pixazo']['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "modelVersionId": "1983308862240288769",
                "prompt": prompt,
                "aspectRatio": "1:1",
                "size": "1k",
                "mode": "standard"
            }
            
            # Create task
            response = requests.post(
                "https://api.pixai.art/v2/image/create",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            task_id = result.get("id")
            if not task_id:
                return None
            
            # Poll for results
            for _ in range(120):
                time.sleep(1)
                
                poll_response = requests.get(
                    f"https://api.pixai.art/v2/image/{task_id}",
                    headers=headers,
                    timeout=10
                )
                poll_response.raise_for_status()
                poll_result = poll_response.json()
                
                status = poll_result.get("status")
                
                if status == "completed":
                    outputs = poll_result.get("outputs", {})
                    media_urls = outputs.get("mediaUrls", [])
                    
                    if media_urls:
                        img_response = requests.get(media_urls[0], timeout=30)
                        return img_response.content
                    return None
                    
                elif status in ["failed", "error"]:
                    return None
            
            return None
        except Exception as e:
            print(f"Pixazo error: {e}")
            return None
    
    def save_image(self, image_data, model_name, prompt_id):
        """Save image to file"""
        try:
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            file_path = model_dir / f"prompt_{prompt_id}.png"
            
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            return str(file_path)
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
    
    def generate_all(self, prompt, prompt_id):
        """Generate images from all 4 models"""
        results = {}
        
        print(f"  Generating images for prompt {prompt_id}...")
        
        # Black Forest Labs
        print(f"    - Black Forest Labs...", end="", flush=True)
        image = self.generate_with_black_forest_labs(prompt)
        if image:
            self.save_image(image, "black_forest_labs", prompt_id)
            results["black_forest_labs"] = True
            print(" ✓")
        else:
            results["black_forest_labs"] = False
            print(" ✗")
        time.sleep(1)
        
        # Stability AI
        print(f"    - Stability AI...", end="", flush=True)
        image = self.generate_with_stability_ai(prompt)
        if image:
            self.save_image(image, "stability_ai", prompt_id)
            results["stability_ai"] = True
            print(" ✓")
        else:
            results["stability_ai"] = False
            print(" ✗")
        time.sleep(1)
        
        # Hugging Face
        print(f"    - Hugging Face...", end="", flush=True)
        image = self.generate_with_huggingface(prompt)
        if image:
            self.save_image(image, "huggingface", prompt_id)
            results["huggingface"] = True
            print(" ✓")
        else:
            results["huggingface"] = False
            print(" ✗")
        time.sleep(1)
        
        # Pixazo
        print(f"    - Pixazo...", end="", flush=True)
        image = self.generate_with_pixazo(prompt)
        if image:
            self.save_image(image, "pixazo", prompt_id)
            results["pixazo"] = True
            print(" ✓")
        else:
            results["pixazo"] = False
            print(" ✗")
        time.sleep(1)
        
        return results