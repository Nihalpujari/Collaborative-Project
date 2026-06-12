"""
Graphing module - creates 8 static graphs comparing all models
Including big players (Claude, ChatGPT, Gemini) benchmarks
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config import OUTPUT_DIRS, BIG_PLAYERS_BENCHMARKS

class GraphGenerator:
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIRS["graphs"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir = Path(OUTPUT_DIRS["scores"])
    
    def load_scores_from_csv(self, csv_file):
        """Load scores from CSV file"""
        scores = {}
        
        if not csv_file.exists():
            return scores
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['model']
                if model not in scores:
                    scores[model] = []
                scores[model].append(row)
        
        return scores
    
    def calculate_averages(self, scores_data, metric_keys):
        """Calculate average scores per model for given metrics"""
        averages = {}
        
        for model, rows in scores_data.items():
            model_scores = {}
            for metric in metric_keys:
                values = [float(row.get(metric, 0)) for row in rows if metric in row]
                avg = sum(values) / len(values) if values else 0
                model_scores[metric] = round(avg, 4)
            averages[model] = model_scores
        
        return averages
    
    def add_big_players(self, averages, metrics):
        """Add big player benchmarks to the averages dict"""
        for player, benchmarks in BIG_PLAYERS_BENCHMARKS.items():
            player_scores = {}
            for metric in metrics:
                player_scores[metric] = benchmarks.get(metric, 0)
            averages[player] = player_scores
        
        return averages
    
    def plot_metric(self, title, metric_key, averages, ylabel, filename, color_map=None):
        """Create a single metric plot"""
        models = list(averages.keys())
        values = [averages[model].get(metric_key, 0) for model in models]
        
        # Separate small models and big players
        small_models = [m for m in models if m not in ['claude', 'chatgpt', 'gemini']]
        big_players = [m for m in models if m in ['claude', 'chatgpt', 'gemini']]
        
        small_values = [averages[m].get(metric_key, 0) for m in small_models]
        big_values = [averages[m].get(metric_key, 0) for m in big_players]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot small models
        x_pos_small = np.arange(len(small_models))
        bars_small = ax.bar(x_pos_small, small_values, label='Small Models', color='steelblue', alpha=0.8, width=0.6)
        
        # Plot big players
        x_pos_big = np.arange(len(small_models), len(small_models) + len(big_players))
        bars_big = ax.bar(x_pos_big, big_values, label='Big Players', color='coral', alpha=0.8, width=0.6)
        
        # Customization
        all_models = small_models + big_players
        all_positions = list(x_pos_small) + list(x_pos_big)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_models, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bars in [bars_small, bars_big]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {filename}")
    
    def generate_all_graphs(self):
        """Generate all 8 graphs"""
        print("\nGenerating graphs...")

        # Load scores — check both possible locations
        def find_csv(filename):
            candidates = [
                self.scores_dir / filename,
                Path("outputs/text_benchmark/scores") / filename,
            ]
            for p in candidates:
                if p.exists():
                    return p
            return self.scores_dir / filename  # fallback (may not exist)

        text_scores = self.load_scores_from_csv(find_csv("text_scores.csv"))
        image_scores = self.load_scores_from_csv(find_csv("image_scores.csv"))
        audio_scores = self.load_scores_from_csv(find_csv("audio_scores.csv"))
        
        # Calculate averages
        text_avgs = self.calculate_averages(text_scores, ['bertscore', 'rouge', 'readability'])
        image_avgs = self.calculate_averages(image_scores, ['clip', 'aesthetic'])
        audio_avgs = self.calculate_averages(audio_scores, ['clap', 'mos', 'wer'])
        
        # Add big players
        text_avgs = self.add_big_players(text_avgs, ['bertscore', 'rouge', 'readability'])
        image_avgs = self.add_big_players(image_avgs, ['clip', 'aesthetic'])
        audio_avgs = self.add_big_players(audio_avgs, ['clap', 'mos', 'wer'])
        
        # Generate TEXT graphs
        print("  TEXT Metrics:")
        self.plot_metric(
            'BERTScore: Semantic Similarity',
            'bertscore',
            text_avgs,
            'Score (0-1)',
            'bertscore_comparison.png'
        )
        
        self.plot_metric(
            'ROUGE: Overlap-based Similarity',
            'rouge',
            text_avgs,
            'Score (0-1)',
            'rouge_comparison.png'
        )
        
        self.plot_metric(
            'Readability: Flesch-Kincaid Grade',
            'readability',
            text_avgs,
            'Score (0-10)',
            'readability_comparison.png'
        )
        
        # Generate IMAGE graphs
        print("  IMAGE Metrics:")
        self.plot_metric(
            'CLIP: Image-Text Alignment',
            'clip',
            image_avgs,
            'Score (0-1)',
            'clip_comparison.png'
        )
        
        self.plot_metric(
            'Aesthetic: Image Quality',
            'aesthetic',
            image_avgs,
            'Score (0-10)',
            'aesthetic_comparison.png'
        )
        
        # Generate AUDIO graphs
        print("  AUDIO Metrics:")
        self.plot_metric(
            'CLAP: Audio-Text Alignment',
            'clap',
            audio_avgs,
            'Score (0-1)',
            'clap_comparison.png'
        )
        
        self.plot_metric(
            'MOS: Mean Opinion Score (Quality)',
            'mos',
            audio_avgs,
            'Score (1-5)',
            'mos_comparison.png'
        )
        
        self.plot_metric(
            'WER: Word Error Rate',
            'wer',
            audio_avgs,
            'Error Rate (0-1, lower is better)',
            'wer_comparison.png'
        )
        
        print("\n✓ All graphs generated successfully!")
        print(f"  Saved to: {self.output_dir}")

def main():
    """Test the graph generator"""
    generator = GraphGenerator()
    generator.generate_all_graphs()

if __name__ == "__main__":
    main()
