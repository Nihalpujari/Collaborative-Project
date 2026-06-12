"""
Audio scoring module - computes metrics for generated audio
- CLAP (audio-text alignment)
- MOS (Mean Opinion Score - quality rating)
- WER (Word Error Rate - speech-to-text accuracy)
"""

import os
from pathlib import Path
import csv
from config import OUTPUT_DIRS

class AudioScorer:
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIRS["scores"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = Path(OUTPUT_DIRS["audio"])
        self.models = ["camb_ai", "deepgram", "cartesia", "fish_audio"]
    
    def compute_clap_score(self, audio_path, text_prompt):
        """
        Compute CLAP score (audio-text alignment)
        Returns a score between 0 and 1
        """
        try:
            import torch
            import librosa
            from transformers import ClapProcessor, ClapModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
            processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=48000)
            
            # Process
            inputs = processor(text=[text_prompt], audios=[audio], return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Calculate similarity
            logits_per_audio = outputs.logits_per_audio
            similarity = torch.sigmoid(logits_per_audio).item()
            
            return round(similarity, 4)
        except Exception as e:
            print(f"CLAP score error: {e}")
            return 0.0
    
    def compute_mos_score(self, audio_path):
        """
        Compute Mean Opinion Score (MOS) - audio quality
        Uses a simplified model or heuristic
        Returns a score between 1 and 5 (normalized)
        """
        try:
            import librosa
            import numpy as np
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features for quality assessment
            # Using simple heuristics based on audio analysis
            
            # 1. Signal-to-noise ratio estimation
            S = librosa.feature.melspectrogram(y=audio, sr=sr)
            power = np.mean(S)
            
            # 2. RMS energy
            rms = np.sqrt(np.mean(audio ** 2))
            
            # 3. Spectral centroid (quality indicator)
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
            
            # Combine into MOS-like score (1-5 scale)
            # Normalize features
            power_norm = min(5.0, (power / 100) * 5)
            rms_norm = min(5.0, (rms * 10) * 5)
            centroid_norm = min(5.0, (centroid / sr) * 5)
            
            # Average the normalized scores
            mos = (power_norm + rms_norm + centroid_norm) / 3
            mos = min(5.0, max(1.0, mos))
            
            return round(mos, 4)
        except Exception as e:
            print(f"MOS score error: {e}")
            return 0.0
    
    def compute_wer_score(self, audio_path, reference_text):
        """
        Compute Word Error Rate (WER) - accuracy of speech-to-text
        Returns a score between 0 and 1 (0 = perfect, 1 = completely wrong)
        """
        try:
            import librosa
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Transcribe
            inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(device)
            with torch.no_grad():
                generated_ids = model.generate(inputs, max_length=225)
            
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Compute WER
            from jiwer import wer
            error_rate = wer(reference_text.lower(), transcription.lower())
            
            # Normalize to 0-1 (can be > 1 for very bad transcriptions)
            error_rate = min(1.0, error_rate)
            
            return round(error_rate, 4)
        except Exception as e:
            print(f"WER score error: {e}")
            return 0.0
    
    def score_all_audio(self, reference_prompts, reference_texts):
        """
        Score all generated audio for all models
        reference_prompts: dict with prompt_id -> original prompt text
        reference_texts: dict with prompt_id -> generated text description
        """
        results = []
        
        print("\nScoring generated audio...")
        
        for model in self.models:
            model_dir = self.audio_dir / model
            
            if not model_dir.exists():
                print(f"  {model}: No files found")
                continue
            
            print(f"  Scoring {model}...", end="", flush=True)
            scores_data = []
            
            for prompt_id in range(1, len(reference_prompts) + 1):
                file_path = model_dir / f"prompt_{prompt_id}.wav"
                
                if not file_path.exists():
                    continue
                
                try:
                    prompt_text = reference_prompts.get(prompt_id, "")
                    ref_text = reference_texts.get(prompt_id, prompt_text)
                    
                    # Compute metrics
                    clap_score = self.compute_clap_score(str(file_path), prompt_text)
                    mos_score = self.compute_mos_score(str(file_path))
                    wer_score = self.compute_wer_score(str(file_path), ref_text)
                    
                    scores_data.append({
                        'prompt_id': prompt_id,
                        'model': model,
                        'clap': clap_score,
                        'mos': mos_score,
                        'wer': wer_score
                    })
                except Exception as e:
                    print(f"Error processing {model} prompt {prompt_id}: {e}")
            
            results.extend(scores_data)
            print(" ✓")
        
        # Save to CSV
        csv_path = self.output_dir / "audio_scores.csv"
        if results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['prompt_id', 'model', 'clap', 'mos', 'wer'])
                writer.writeheader()
                writer.writerows(results)
            print(f"  Audio scores saved to {csv_path}")
        
        return results
    
    def get_average_scores(self, scores_data):
        """Calculate average scores per model"""
        from collections import defaultdict
        
        averages = defaultdict(lambda: {'clap': [], 'mos': [], 'wer': []})
        
        for item in scores_data:
            model = item['model']
            averages[model]['clap'].append(item['clap'])
            averages[model]['mos'].append(item['mos'])
            averages[model]['wer'].append(item['wer'])
        
        result = {}
        for model, scores in averages.items():
            result[model] = {
                'clap': round(sum(scores['clap']) / len(scores['clap']), 4) if scores['clap'] else 0,
                'mos': round(sum(scores['mos']) / len(scores['mos']), 4) if scores['mos'] else 0,
                'wer': round(sum(scores['wer']) / len(scores['wer']), 4) if scores['wer'] else 0
            }
        
        return result

def main():
    """Test the audio scorer"""
    from prompts import get_prompts
    
    scorer = AudioScorer()
    prompts = get_prompts()
    
    # Create reference dicts
    reference_prompts = {i+1: prompt for i, prompt in enumerate(prompts)}
    reference_texts = {i+1: prompt for i, prompt in enumerate(prompts)}  # Same as prompts for this test
    
    # Score all
    scores = scorer.score_all_audio(reference_prompts, reference_texts)
    
    if scores:
        print("\nAverage scores per model:")
        averages = scorer.get_average_scores(scores)
        for model, avg in averages.items():
            print(f"\n{model}:")
            for metric, value in avg.items():
                print(f"  {metric}: {value}")
    else:
        print("No scores to display (no generated audio found)")

if __name__ == "__main__":
    main()
