"""
Compare text outputs from 3 models
Generate comparison report
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def compare_outputs():
    """Compare text outputs and generate report"""
    
    output_dir = Path("outputs/text_benchmark")
    
    print("\n" + "="*70)
    print("TEXT COMPARISON ANALYSIS")
    print("="*70 + "\n")
    
    models = ["mistral", "groq", "cerebras"]
    model_stats = {}
    
    # Analyze outputs
    for model in models:
        model_dir = output_dir / model
        if not model_dir.exists():
            print(f"❌ {model}: No outputs found")
            continue
        
        files = list(model_dir.glob("prompt_*.txt"))
        
        lengths = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                lengths.append(len(content))
        
        if lengths:
            model_stats[model] = {
                "count": len(files),
                "total_chars": sum(lengths),
                "avg_length": np.mean(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "std_dev": np.std(lengths)
            }
            
            print(f"{model.upper()}:")
            print(f"  Generated: {len(files)} outputs")
            print(f"  Total chars: {sum(lengths):,}")
            print(f"  Avg output: {np.mean(lengths):.0f} chars")
            print(f"  Range: {min(lengths)} - {max(lengths)} chars")
            print()
    
    # Create comparison chart
    if len(model_stats) > 0:
        create_comparison_chart(model_stats)
    
    print("="*70 + "\n")

def create_comparison_chart(stats):
    """Create comparison chart"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(stats.keys())
    avg_lengths = [stats[m]["avg_length"] for m in models]
    counts = [stats[m]["count"] for m in models]
    
    # Chart 1: Average output length
    axes[0].bar(models, avg_lengths, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_ylabel('Average Length (chars)', fontsize=12)
    axes[0].set_title('Average Output Length Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Chart 2: Success rate
    axes[1].bar(models, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_ylabel('Outputs Generated', fontsize=12)
    axes[1].set_title('Output Count Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = Path("outputs/text_benchmark/charts")
    chart_dir.mkdir(exist_ok=True)
    chart_path = chart_dir / "comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison chart saved: {chart_path}\n")
    plt.close()

if __name__ == "__main__":
    compare_outputs()
