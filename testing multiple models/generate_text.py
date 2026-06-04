"""
Text generation module - handles 3 text models
- Cerebras
- Groq
- Mistral AI
"""

import requests
import json
import time
from pathlib import Path
from config import TEXT_MODELS, OUTPUT_DIRS

class TextGenerator:
    def __init__(self):
        self.models = TEXT_MODELS
        self.output_dir = Path(OUTPUT_DIRS["text"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_with_cerebras(self, prompt):
        """Generate text using Cerebras API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['cerebras']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-oss-120b",
                "messages": [
                    {"role": "user", "content": f"Describe this scene in detail: {prompt}"}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            }
            response = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Cerebras error: {e}")
            return None
    
    def generate_with_groq(self, prompt):
        """Generate text using Groq API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['groq']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "user", "content": f"Describe this scene in detail: {prompt}"}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Groq error: {e}")
            return None
    
    def generate_with_mistral(self, prompt):
        """Generate text using Mistral AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.models['mistral']['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.models['mistral']['model_name'],
                "messages": [
                    {"role": "user", "content": f"Describe this scene in detail: {prompt}"}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            }
            response = requests.post(
                self.models['mistral']['api_url'],
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Mistral error: {e}")
            return None
    
    def generate_all(self, prompt, prompt_id):
        """Generate text from all 3 models for a single prompt"""
        results = {}
        
        print(f"  Generating text for prompt {prompt_id}...")
        
        # Cerebras
        print(f"    - Cerebras...", end="", flush=True)
        text = self.generate_with_cerebras(prompt)
        results["cerebras"] = text
        print(" ✓" if text else " ✗")
        time.sleep(3)  # Rate limiting
        
        # Groq
        print(f"    - Groq...", end="", flush=True)
        text = self.generate_with_groq(prompt)
        results["groq"] = text
        print(" ✓" if text else " ✗")
        time.sleep(3)
        
        # Mistral
        print(f"    - Mistral...", end="", flush=True)
        text = self.generate_with_mistral(prompt)
        results["mistral"] = text
        print(" ✓" if text else " ✗")
        time.sleep(3)
        
        # Save to files
        for model_name, text in results.items():
            if text:
                model_dir = self.output_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                file_path = model_dir / f"prompt_{prompt_id}.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Write prompt and generated text
                    f.write(f"ORIGINAL PROMPT:\n")
                    f.write(f"{prompt}\n")
                    f.write(f"\n{'='*70}\n\n")
                    f.write(f"GENERATED TEXT:\n")
                    f.write(f"{text}")
        
        return results

def main():
    """Test the text generator"""
    from prompts import get_prompts
    
    generator = TextGenerator()
    prompts = get_prompts()
    
    # Test with first prompt
    test_prompt = prompts[0]
    print(f"\nTesting text generation with: '{test_prompt}'")
    results = generator.generate_all(test_prompt, 1)
    
    for model, text in results.items():
        if text:
            print(f"\n{model.upper()}:")
            print(text[:200] + "..." if len(text) > 200 else text)

if __name__ == "__main__":
    main()