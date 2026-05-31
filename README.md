## AI Benchmark Pipeline

Complete pipeline for benchmarking small AI models against big players (Claude, ChatGPT, Gemini).

### 📋 Overview

This project generates outputs from 12 AI models (4 text, 4 image, 4 audio) across 50 prompts, computes 8 different evaluation metrics, and creates comparison graphs.

**Models:**
- **Text (4):** Cerebras, Groq, Mistral AI, DeepSeek
- **Image (4):** Black Forest Labs, Stability AI, Hugging Face, Pixazo
- **Audio (4):** Camb AI, Deepgram, Cartesia AI, Fish Audio

**Metrics:**
- **Text:** BERTScore, ROUGE, Readability
- **Image:** CLIP, Aesthetic
- **Audio:** CLAP, MOS, WER

---

## 🚀 Setup Instructions

### 1. Install Python 3.10+
```bash
python --version  # Should be 3.10 or higher
```

### 2. Clone/Download the Project
Place all `.py` files in the same folder.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If you're using GPU (CUDA), also install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configure API Keys

**IMPORTANT:** Fill in your API keys in `config.py` before running.

Open `config.py` and replace all `YOUR_*_API_KEY` values:

```python
TEXT_MODELS = {
    "cerebras": {
        "api_key": "your_actual_cerebras_key_here",
        ...
    },
    ...
}
```

All sections to update:
- `TEXT_MODELS` (4 keys)
- `IMAGE_MODELS` (4 keys)
- `AUDIO_MODELS` (4 keys)

---

## ▶️ Running the Pipeline

### Full Pipeline (All Steps)
```bash
python main.py
```

This runs:
1. Text generation (all 4 models × 50 prompts)
2. Image generation (all 4 models × 50 prompts)
3. Audio generation (all 4 models × 50 prompts)
4. Scoring (all metrics)
5. Graph generation (8 comparison graphs)

### Individual Components (For Testing)

**Test text generation:**
```bash
python generate_text.py
```

**Test image generation:**
```bash
python generate_image.py
```

**Test audio generation:**
```bash
python generate_audio.py
```

**Score outputs:**
```bash
python score_text.py
python score_image.py
python score_audio.py
```

**Generate graphs:**
```bash
python graph.py
```

---

## 📁 Output Structure

```
outputs/
├── text/
│   ├── cerebras/
│   │   ├── prompt_1.txt
│   │   ├── prompt_2.txt
│   │   └── ...
│   ├── groq/
│   ├── mistral/
│   └── deepseek/
│
├── images/
│   ├── black_forest_labs/
│   ├── stability_ai/
│   ├── huggingface/
│   └── pixazo/
│
├── audio/
│   ├── camb_ai/
│   ├── deepgram/
│   ├── cartesia/
│   └── fish_audio/
│
├── scores/
│   ├── text_scores.csv
│   ├── image_scores.csv
│   └── audio_scores.csv
│
└── graphs/
    ├── bertscore_comparison.png
    ├── rouge_comparison.png
    ├── readability_comparison.png
    ├── clip_comparison.png
    ├── aesthetic_comparison.png
    ├── clap_comparison.png
    ├── mos_comparison.png
    └── wer_comparison.png
```

---

## 📊 Metric Explanations

### Text Metrics
- **BERTScore:** Semantic similarity (0-1, higher=better)
- **ROUGE:** Overlap-based similarity (0-1, higher=better)
- **Readability:** Flesch-Kincaid grade (0-10, higher=easier)

### Image Metrics
- **CLIP:** Image-text alignment (0-1, higher=better)
- **Aesthetic:** Image quality score (0-10, higher=better)

### Audio Metrics
- **CLAP:** Audio-text alignment (0-1, higher=better)
- **MOS:** Mean Opinion Score (1-5, higher=better)
- **WER:** Word Error Rate (0-1, lower=better)

---

## ⚠️ Important Notes

1. **Rate Limiting:** The pipeline respects rate limits with built-in delays. Adjust in the `generate_*.py` files if needed.

2. **GPU vs CPU:** Models run on GPU if available, otherwise CPU (slower). Some models may require significant memory.

3. **First Run:** Initial runs download pre-trained models (~5-10GB). Be patient on the first execution.

4. **API Costs:** Check your API providers' pricing! Running 50 prompts × 12 models = 600+ API calls.

5. **Error Handling:** If an API call fails, the pipeline continues and marks it as ✗. Check logs for details.

6. **Scoring Speed:** Scoring can be slow (especially CLIP, WER). Processing 50 prompts × metrics can take hours.

---

## 🔧 Customization

### Change Output Directory
Edit `config.py`:
```python
OUTPUT_DIRS = {
    "text": "/custom/path/text",
    ...
}
```

### Adjust Number of Prompts
Edit `prompts.py` - add/remove from the `PROMPTS` list.

### Change Model Parameters
Edit the `generate_*.py` files:
```python
payload = {
    "temperature": 0.7,  # Adjust this
    "max_tokens": 300,   # Or this
    ...
}
```

### Modify Benchmark Values
Edit `config.py` under `BIG_PLAYERS_BENCHMARKS`:
```python
BIG_PLAYERS_BENCHMARKS = {
    "claude": {
        "bertscore": 0.89,  # Update these
        ...
    }
}
```

---

## 🐛 Troubleshooting

### "API Key Invalid" Error
- Check spelling in `config.py`
- Verify key hasn't expired
- Ensure key has necessary permissions

### "Module not found" Error
```bash
pip install -r requirements.txt --upgrade
```

### Out of Memory
- Reduce batch sizes in `generate_*.py`
- Process fewer prompts at a time
- Use CPU mode (slower but uses less memory)

### Slow Performance
- Check GPU usage: `nvidia-smi` (if using CUDA)
- Reduce model precision (use fp16 instead of fp32)
- Process in parallel (advanced)

### Graphs Not Generated
- Check if CSV files exist in `outputs/scores/`
- Ensure scores are computed before graphing
- Check matplotlib installation: `pip install matplotlib --upgrade`

---

## 📝 File Reference

| File | Purpose |
|------|---------|
| `config.py` | API keys and configuration |
| `prompts.py` | 50 test prompts |
| `generate_text.py` | Text generation (4 models) |
| `generate_image.py` | Image generation (4 models) |
| `generate_audio.py` | Audio generation (4 models) |
| `score_text.py` | Text scoring (BERTScore, ROUGE, Readability) |
| `score_image.py` | Image scoring (CLIP, Aesthetic) |
| `score_audio.py` | Audio scoring (CLAP, MOS, WER) |
| `graph.py` | Graph generation (8 metrics) |
| `main.py` | Orchestration (full pipeline) |
| `requirements.txt` | Python dependencies |

---

## ✅ Quick Checklist

- [ ] Python 3.10+ installed
- [ ] All files in same directory
- [ ] `requirements.txt` installed
- [ ] API keys filled in `config.py`
- [ ] Run `python main.py`
- [ ] Check `outputs/` folder for results
- [ ] Review graphs in `outputs/graphs/`

---

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Review troubleshooting section
3. Verify API keys and permissions
4. Check API provider's documentation
5. Review model-specific parameters

Good luck! 🚀
