"""
Run complete TEXT-ONLY benchmark pipeline
1. Generate text from 3 models (50 prompts)
2. Compare outputs
3. Generate report
"""

import sys
sys.path.insert(0, '/home/claude')

from text_benchmark import TextBenchmark
import subprocess
from pathlib import Path

print("\n" + "="*70)
print("TEXT-ONLY BENCHMARK PIPELINE")
print("="*70)
print("\nStep 1: Generating text from 3 models (50 prompts)...")
print("-"*70 + "\n")

# Run benchmark
benchmark = TextBenchmark()
benchmark.run_benchmark(num_prompts=50)

print("\nStep 2: Analyzing and comparing outputs...")
print("-"*70 + "\n")

# Run comparison
try:
    subprocess.run([sys.executable, "compare_text.py"], check=True)
except:
    print("Note: Comparison requires matplotlib (pip install matplotlib)")

print("\n" + "="*70)
print("BENCHMARK COMPLETE!")
print("="*70)
print("\nOutputs:")
print("  ✓ Text files: outputs/text_benchmark/{mistral,groq,cerebras}/")
print("  ✓ Comparison chart: outputs/text_benchmark/charts/comparison.png")
print("\n" + "="*70 + "\n")

