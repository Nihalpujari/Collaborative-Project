"""
Image scoring module - computes metrics for generated images
- CLIP (semantic alignment between image and prompt)
- Aesthetic (image quality and aesthetic score)
"""

import os
from pathlib import Path
import csv
import numpy as np
from config import OUTPUT_DIRS

class ImageScorer:
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIRS["scores"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = Path(OUTPUT_DIRS["image"])
        self.models = ["black_forest_labs", "stability_ai", "huggingface", "pixazo"]
    
    def compute_clip_score(self, image_path, text_prompt):
        """
        Compute CLIP score (image-text alignment)
        Returns a score between -1 and 1 (normalized to 0-1)
        """
        try:
            import torch
            from PIL import Image
            from transformers import CLIPProcessor, CLIPModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True).to(device)
            
            # Get outputs
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Calculate similarity
            logits_per_image = outputs.logits_per_image
            similarity = torch.sigmoid(logits_per_image).item()
            
            return round(similarity, 4)
        except Exception as e:
            print(f"CLIP score error: {e}")
            return 0.0
    
    def compute_aesthetic_score(self, image_path):
        """
        Compute aesthetic score (image quality)
        Uses a simplified aesthetic model or heuristic
        Returns a score between 0 and 10
        """
        try:
            from PIL import Image
            import torch
            from transformers import AutoProcessor, AutoModelForImageClassification
            
            # Using a simple aesthetic predictor
            # This is a placeholder - in production, use a proper aesthetic model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try to load aesthetic model
            try:
                model_name = "nateraw/vit-base-aesthetic"
                model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
                processor = AutoProcessor.from_pretrained(model_name)
                
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Normalize logits to 0-10 scale
                logits = outputs.logits[0]
                score = torch.softmax(logits, dim=0)[1].item() * 10  # Assuming higher class is better
                return round(score, 4)
            except:
                # Fallback: use image analysis heuristics
                image = Image.open(image_path).convert("RGB")
                pixels = np.array(image)
                
                # Simple heuristic: score based on color variance and contrast
                contrast = np.std(pixels)
                variance = np.mean([np.std(pixels[:,:,i]) for i in range(3)])
                
                # Normalize to 0-10
                score = (contrast + variance) / 50
                score = min(10.0, max(0.0, score))
                return round(score, 4)
                
        except Exception as e:
            print(f"Aesthetic score error: {e}")
            return 0.0
    
    def score_all_images(self, reference_prompts):
        """
        Score all generated images for all models
        reference_prompts: dict with prompt_id -> original prompt text
        """
        results = []
        
        print("\nScoring generated images...")
        
        for model in self.models:
            model_dir = self.image_dir / model
            
            if not model_dir.exists():
                print(f"  {model}: No files found")
                continue
            
            print(f"  Scoring {model}...", end="", flush=True)
            scores_data = []
            
            for prompt_id in range(1, len(reference_prompts) + 1):
                file_path = model_dir / f"prompt_{prompt_id}.png"
                
                if not file_path.exists():
                    continue
                
                try:
                    prompt_text = reference_prompts.get(prompt_id, "")
                    
                    # Compute metrics
                    clip_score = self.compute_clip_score(str(file_path), prompt_text)
                    aesthetic_score = self.compute_aesthetic_score(str(file_path))
                    
                    scores_data.append({
                        'prompt_id': prompt_id,
                        'model': model,
                        'clip': clip_score,
                        'aesthetic': aesthetic_score
                    })
                except Exception as e:
                    print(f"Error processing {model} prompt {prompt_id}: {e}")
            
            results.extend(scores_data)
            print(" ✓")
        
        # Save to CSV
        csv_path = self.output_dir / "image_scores.csv"
        if results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['prompt_id', 'model', 'clip', 'aesthetic'])
                writer.writeheader()
                writer.writerows(results)
            print(f"  Image scores saved to {csv_path}")
        
        return results
    
    def get_average_scores(self, scores_data):
        """Calculate average scores per model"""
        from collections import defaultdict
        
        averages = defaultdict(lambda: {'clip': [], 'aesthetic': []})
        
        for item in scores_data:
            model = item['model']
            averages[model]['clip'].append(item['clip'])
            averages[model]['aesthetic'].append(item['aesthetic'])
        
        result = {}
        for model, scores in averages.items():
            result[model] = {
                'clip': round(sum(scores['clip']) / len(scores['clip']), 4) if scores['clip'] else 0,
                'aesthetic': round(sum(scores['aesthetic']) / len(scores['aesthetic']), 4) if scores['aesthetic'] else 0
            }
        
        return result

def main():
    """Test the image scorer"""
    from prompts import get_prompts
    
    scorer = ImageScorer()
    prompts = get_prompts()
    
    # Create reference dict
    reference_prompts = {i+1: prompt for i, prompt in enumerate(prompts)}
    
    # Score all
    scores = scorer.score_all_images(reference_prompts)
    
    if scores:
        print("\nAverage scores per model:")
        averages = scorer.get_average_scores(scores)
        for model, avg in averages.items():
            print(f"\n{model}:")
            for metric, value in avg.items():
                print(f"  {metric}: {value}")
    else:
        print("No scores to display (no generated images found)")

if __name__ == "__main__":
    main()
