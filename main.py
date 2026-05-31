"""
Main orchestration script
Runs the complete pipeline:
1. Generate text from all models
2. Generate images from prompts
3. Generate audio from generated text
4. Score all outputs
5. Generate comparison graphs
"""

import time
from pathlib import Path
from prompts import get_prompts, get_total_prompts
from generate_text import TextGenerator
from generate_image import ImageGenerator
from generate_audio import AudioGenerator
from score_text import TextScorer
from score_image import ImageScorer
from score_audio import AudioScorer
from graph import GraphGenerator
from config import OUTPUT_DIRS

class Pipeline:
    def __init__(self):
        self.total_prompts = get_total_prompts()
        self.prompts = get_prompts()
        self.generated_texts = {}  # Store generated texts for audio generation
        
        self.text_gen = TextGenerator()
        self.image_gen = ImageGenerator()
        self.audio_gen = AudioGenerator()
        
        self.text_scorer = TextScorer()
        self.image_scorer = ImageScorer()
        self.audio_scorer = AudioScorer()
        
        self.graph_gen = GraphGenerator()
    
    def setup_output_dirs(self):
        """Create all necessary output directories"""
        for dir_name in OUTPUT_DIRS.values():
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        print("✓ Output directories created")
    
    def run_text_generation(self):
        """Generate text from all 4 text models"""
        print("\n" + "="*70)
        print("STEP 1: TEXT GENERATION")
        print("="*70)
        
        start_time = time.time()
        
        for prompt_id in range(1, self.total_prompts + 1):
            prompt = self.prompts[prompt_id - 1]
            print(f"\nPrompt {prompt_id}/{self.total_prompts}: {prompt[:60]}...")
            
            results = self.text_gen.generate_all(prompt, prompt_id)
            
            # Store first successful text for each prompt (for audio generation)
            for model, text in results.items():
                if text and prompt_id not in self.generated_texts:
                    self.generated_texts[prompt_id] = text
        
        elapsed = time.time() - start_time
        print(f"\n✓ Text generation completed in {elapsed:.1f}s")
    
    def run_image_generation(self):
        """Generate images from all 4 image models"""
        print("\n" + "="*70)
        print("STEP 2: IMAGE GENERATION")
        print("="*70)
        
        start_time = time.time()
        
        for prompt_id in range(1, self.total_prompts + 1):
            prompt = self.prompts[prompt_id - 1]
            print(f"\nPrompt {prompt_id}/{self.total_prompts}: {prompt[:60]}...")
            
            results = self.image_gen.generate_all(prompt, prompt_id)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Image generation completed in {elapsed:.1f}s")
    
    def run_audio_generation(self):
        """Generate audio (TTS) from generated texts"""
        print("\n" + "="*70)
        print("STEP 3: AUDIO GENERATION (Text-to-Speech)")
        print("="*70)
        
        start_time = time.time()
        
        for prompt_id in range(1, self.total_prompts + 1):
            # Use generated text if available, otherwise use original prompt
            text = self.generated_texts.get(prompt_id, self.prompts[prompt_id - 1])
            print(f"\nPrompt {prompt_id}/{self.total_prompts}: {text[:60]}...")
            
            results = self.audio_gen.generate_all(text, prompt_id)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Audio generation completed in {elapsed:.1f}s")
    
    def run_text_scoring(self):
        """Score all generated text"""
        print("\n" + "="*70)
        print("STEP 4: TEXT SCORING")
        print("="*70)
        
        # Create reference prompts dict
        reference_prompts = {i+1: prompt for i, prompt in enumerate(self.prompts)}
        
        scores = self.text_scorer.score_all_prompts(reference_prompts)
        
        if scores:
            averages = self.text_scorer.get_average_scores(scores)
            print("\nAverage scores per model:")
            for model, avg in sorted(averages.items()):
                print(f"\n  {model.upper()}:")
                for metric, value in avg.items():
                    print(f"    {metric}: {value}")
    
    def run_image_scoring(self):
        """Score all generated images"""
        print("\n" + "="*70)
        print("STEP 5: IMAGE SCORING")
        print("="*70)
        
        # Create reference prompts dict
        reference_prompts = {i+1: prompt for i, prompt in enumerate(self.prompts)}
        
        scores = self.image_scorer.score_all_images(reference_prompts)
        
        if scores:
            averages = self.image_scorer.get_average_scores(scores)
            print("\nAverage scores per model:")
            for model, avg in sorted(averages.items()):
                print(f"\n  {model.upper()}:")
                for metric, value in avg.items():
                    print(f"    {metric}: {value}")
    
    def run_audio_scoring(self):
        """Score all generated audio"""
        print("\n" + "="*70)
        print("STEP 6: AUDIO SCORING")
        print("="*70)
        
        # Create reference dicts
        reference_prompts = {i+1: prompt for i, prompt in enumerate(self.prompts)}
        reference_texts = self.generated_texts.copy()
        if not reference_texts:
            reference_texts = reference_prompts
        
        scores = self.audio_scorer.score_all_audio(reference_prompts, reference_texts)
        
        if scores:
            averages = self.audio_scorer.get_average_scores(scores)
            print("\nAverage scores per model:")
            for model, avg in sorted(averages.items()):
                print(f"\n  {model.upper()}:")
                for metric, value in avg.items():
                    print(f"    {metric}: {value}")
    
    def run_graph_generation(self):
        """Generate comparison graphs"""
        print("\n" + "="*70)
        print("STEP 7: GRAPH GENERATION")
        print("="*70)
        
        self.graph_gen.generate_all_graphs()
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "  AI BENCHMARK PIPELINE - COMPLETE EXECUTION".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)
        
        self.setup_output_dirs()
        
        try:
            # Generation phase
            self.run_text_generation()
            self.run_image_generation()
            self.run_audio_generation()
            
            # Scoring phase
            self.run_text_scoring()
            self.run_image_scoring()
            self.run_audio_scoring()
            
            # Visualization phase
            self.run_graph_generation()
            
            # Final summary
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Pipeline interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "  ✓ PIPELINE COMPLETED SUCCESSFULLY".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)
        
        print("\n📁 Output Structure:")
        print(f"  outputs/")
        print(f"  ├── text/        → Text outputs by model")
        print(f"  ├── images/      → Image outputs by model")
        print(f"  ├── audio/       → Audio outputs by model")
        print(f"  ├── scores/      → CSV files with all scores")
        print(f"  │   ├── text_scores.csv")
        print(f"  │   ├── image_scores.csv")
        print(f"  │   └── audio_scores.csv")
        print(f"  └── graphs/      → 8 comparison graphs")
        print(f"      ├── bertscore_comparison.png")
        print(f"      ├── rouge_comparison.png")
        print(f"      ├── readability_comparison.png")
        print(f"      ├── clip_comparison.png")
        print(f"      ├── aesthetic_comparison.png")
        print(f"      ├── clap_comparison.png")
        print(f"      ├── mos_comparison.png")
        print(f"      └── wer_comparison.png")
        
        print("\n📊 Metrics Computed:")
        print("  TEXT:")
        print("    • BERTScore (semantic similarity)")
        print("    • ROUGE (overlap-based)")
        print("    • Readability (Flesch-Kincaid)")
        print("  IMAGE:")
        print("    • CLIP (image-text alignment)")
        print("    • Aesthetic (quality score)")
        print("  AUDIO:")
        print("    • CLAP (audio-text alignment)")
        print("    • MOS (mean opinion score)")
        print("    • WER (word error rate)")
        
        print("\n🔄 Models Compared:")
        print("  SMALL MODELS (your 12):")
        print("    Text: Cerebras, Groq, Mistral, DeepSeek")
        print("    Image: Black Forest Labs, Stability AI, Hugging Face, Pixazo")
        print("    Audio: Camb AI, Deepgram, Cartesia, Fish Audio")
        print("  BIG PLAYERS (benchmarks):")
        print("    • Claude")
        print("    • ChatGPT")
        print("    • Gemini")

def main():
    """Main entry point"""
    pipeline = Pipeline()
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
