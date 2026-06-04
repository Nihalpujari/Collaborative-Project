"""
Text scoring module - computes metrics for generated text
- BERTScore (semantic similarity)
- ROUGE (overlap-based similarity)
- Readability (Flesch-Kincaid index)
"""

import os
from pathlib import Path
import csv
from config import OUTPUT_DIRS, SCORING_CONFIG

class TextScorer:
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIRS["scores"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir = Path(OUTPUT_DIRS["text"])
        self.models = ["cerebras", "groq", "mistral", "deepseek"]
    
    def compute_bertscore(self, reference, candidate):
        """
        Compute BERTScore (semantic similarity)
        Returns a score between 0 and 1
        """
        try:
            from bert_score import score
            
            # BERTScore returns (precision, recall, f1)
            P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
            return F1.item()  # Return F1 score
        except Exception as e:
            print(f"BERTScore error: {e}")
            return 0.0
    
    def compute_rouge(self, reference, candidate):
        """
        Compute ROUGE score (primarily ROUGE-L)
        Returns a score between 0 and 1
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            print(f"ROUGE error: {e}")
            return 0.0
    
    def compute_readability(self, text):
        """
        Compute Flesch-Kincaid readability score (0-100, higher = easier to read)
        Returns a score between 0 and 100
        """
        try:
            from textstat import flesch_kincaid_grade
            
            score = flesch_kincaid_grade(text)
            # Normalize to 0-10 scale for consistency
            return min(10.0, max(0.0, score / 10.0))
        except Exception as e:
            print(f"Readability error: {e}")
            return 0.0
    
    def score_all_prompts(self, reference_prompts):
        """
        Score all generated text for all models
        reference_prompts: dict with prompt_id -> original prompt text
        """
        results = []
        
        print("\nScoring generated text...")
        
        for model in self.models:
            model_dir = self.text_dir / model
            
            if not model_dir.exists():
                print(f"  {model}: No files found")
                continue
            
            print(f"  Scoring {model}...", end="", flush=True)
            scores_data = []
            
            for prompt_id in range(1, len(reference_prompts) + 1):
                file_path = model_dir / f"prompt_{prompt_id}.txt"
                
                if not file_path.exists():
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        generated_text = f.read()
                    
                    reference = reference_prompts.get(prompt_id, "")
                    
                    # Compute metrics
                    bertscore = self.compute_bertscore(reference, generated_text)
                    rouge = self.compute_rouge(reference, generated_text)
                    readability = self.compute_readability(generated_text)
                    
                    scores_data.append({
                        'prompt_id': prompt_id,
                        'model': model,
                        'bertscore': round(bertscore, 4),
                        'rouge': round(rouge, 4),
                        'readability': round(readability, 4)
                    })
                except Exception as e:
                    print(f"Error processing {model} prompt {prompt_id}: {e}")
            
            results.extend(scores_data)
            print(" ✓")
        
        # Save to CSV
        csv_path = self.output_dir / "text_scores.csv"
        if results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['prompt_id', 'model', 'bertscore', 'rouge', 'readability'])
                writer.writeheader()
                writer.writerows(results)
            print(f"  Text scores saved to {csv_path}")
        
        return results
    
    def get_average_scores(self, scores_data):
        """Calculate average scores per model"""
        from collections import defaultdict
        
        averages = defaultdict(lambda: {'bertscore': [], 'rouge': [], 'readability': []})
        
        for item in scores_data:
            model = item['model']
            averages[model]['bertscore'].append(item['bertscore'])
            averages[model]['rouge'].append(item['rouge'])
            averages[model]['readability'].append(item['readability'])
        
        result = {}
        for model, scores in averages.items():
            result[model] = {
                'bertscore': round(sum(scores['bertscore']) / len(scores['bertscore']), 4) if scores['bertscore'] else 0,
                'rouge': round(sum(scores['rouge']) / len(scores['rouge']), 4) if scores['rouge'] else 0,
                'readability': round(sum(scores['readability']) / len(scores['readability']), 4) if scores['readability'] else 0
            }
        
        return result

def main():
    """Test the text scorer"""
    from prompts import get_prompts
    
    scorer = TextScorer()
    prompts = get_prompts()
    
    # Create reference dict
    reference_prompts = {i+1: prompt for i, prompt in enumerate(prompts)}
    
    # Score all
    scores = scorer.score_all_prompts(reference_prompts)
    
    if scores:
        print("\nAverage scores per model:")
        averages = scorer.get_average_scores(scores)
        for model, avg in averages.items():
            print(f"\n{model}:")
            for metric, value in avg.items():
                print(f"  {metric}: {value}")
    else:
        print("No scores to display (no generated text files found)")

if __name__ == "__main__":
    main()
