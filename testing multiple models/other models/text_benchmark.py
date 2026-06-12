"""
TEXT-ONLY BENCHMARK
3 Text Models: Mistral, Groq, Cerebras
50 Prompts
Compare outputs with scoring
"""

import sys
sys.path.insert(0, '/home/claude')

import requests
import json
import time
from pathlib import Path
from config import TEXT_MODELS
from prompts import get_prompts

class TextBenchmark:
    def __init__(self):
        self.output_dir = Path("outputs/text_benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def generate_with_mistral(self, prompt):
        """Generate with Mistral API"""
        try:
            config = TEXT_MODELS.get("mistral", {})
            api_key = config.get("api_key")
            api_url = config.get("api_url")
            model = config.get("model_name", "mistral-large-latest")
            
            if not api_key or api_key.startswith("YOUR_"):
                return None
            
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            return None
    
    def generate_with_groq(self, prompt):
        """Generate with Groq API"""
        try:
            config = TEXT_MODELS.get("groq", {})
            api_key = config.get("api_key")
            api_url = config.get("api_url")
            model = config.get("model_name", "llama-3.3-70b-versatile")
            
            if not api_key or api_key.startswith("YOUR_"):
                return None
            
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            return None
    
    def generate_with_cerebras(self, prompt):
        """Generate with Cerebras API"""
        try:
            config = TEXT_MODELS.get("cerebras", {})
            api_key = config.get("api_key")
            api_url = config.get("api_url")
            model = config.get("model_name", "gpt-oss-120b")
            
            if not api_key or api_key.startswith("YOUR_"):
                return None
            
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            return None
    
    def save_output(self, model_name, prompt_id, text):
        """Save generated text to file"""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        file_path = model_dir / f"prompt_{prompt_id}.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def run_benchmark(self, num_prompts=50):
        """Run benchmark on N prompts"""
        all_prompts = get_prompts()  # Returns a LIST
        
        print("\n" + "="*70)
        print(f"TEXT BENCHMARK - {num_prompts} PROMPTS")
        print("Models: Mistral, Groq, Cerebras")
        print("="*70 + "\n")
        
        models = {
            "mistral": self.generate_with_mistral,
            "groq": self.generate_with_groq,
            "cerebras": self.generate_with_cerebras
        }
        
        # all_prompts is a LIST, so iterate from 0 to num_prompts-1
        for idx in range(min(num_prompts, len(all_prompts))):
            prompt_text = all_prompts[idx]
            prompt_id = idx + 1  # For file naming (1-based)
            
            print(f"Prompt {prompt_id}: {prompt_text[:50]}...")
            
            for model_name, generate_func in models.items():
                print(f"  {model_name}...", end="", flush=True)
                
                output = generate_func(prompt_text)
                
                if output:
                    self.save_output(model_name, prompt_id, output)
                    self.results.setdefault(model_name, []).append({
                        "prompt_id": prompt_id,
                        "success": True,
                        "length": len(output)
                    })
                    print(" ✓")
                else:
                    self.results.setdefault(model_name, []).append({
                        "prompt_id": prompt_id,
                        "success": False,
                        "length": 0
                    })
                    print(" ✗")
                
                time.sleep(0.5)
            
            print()
        
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70 + "\n")
        
        for model_name, results in self.results.items():
            success_count = sum(1 for r in results if r["success"])
            total = len(results)
            avg_length = sum(r["length"] for r in results) / total if total > 0 else 0
            
            print(f"{model_name}:")
            print(f"  Success rate: {success_count}/{total} ({100*success_count/total:.1f}%)")
            print(f"  Avg output length: {avg_length:.0f} chars")
            print()
        
        print("="*70)
        print(f"Outputs saved to: {self.output_dir}")
        print("="*70 + "\n")

if __name__ == "__main__":
    benchmark = TextBenchmark()
    benchmark.run_benchmark(num_prompts=50)