# Multimodal Trio: Text, Image and Audio from One Prompt

**M.Sc. Applied Data Science & Artificial Intelligence, Case Studies 1**
**SRH University of Applied Sciences, Heidelberg · September 2026**

**Team:** Namrata Bhoyar · Nihal Pujari · Pramodkumar Shivanna · Anuj Kamble · Gourav Somanna

---

## What this project does

Give it one short prompt, such as *"an owl librarian wearing glasses"*, and it returns three things at once:

- a written **paragraph** about it,
- an AI-generated **image** of it,
- a spoken **audio** narration of the paragraph.

Three separate models do the work, each from the same prompt, and a Gradio web app shows the results side by side.

The harder problem, and the main contribution, is **measuring whether the three outputs actually match each other**. Nothing guarantees they will, because no model sees what the others produced. We built a five-stage evaluation pipeline and compared three scoring methods (averaging, Naive Bayes and Likelihood Ratio) against ratings from an LLM judge on 500 prompts. The app shows the winning method as the **Tricoherence** score.

---

## Contents

1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [API keys](#api-keys)
4. [Running the web app](#running-the-web-app)
5. [Reproducing the evaluation](#reproducing-the-evaluation)
6. [Other tools](#other-tools)
7. [Project structure](#project-structure)
8. [Troubleshooting](#troubleshooting)
9. [Results](#results) · [How the score works](#how-the-tricoherence-score-works) · [Models](#models-used) · [References](#references)

---

## Quick start

```bash
git clone https://github.com/Nihalpujari/Collaborative-Project.git
cd Collaborative-Project
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy api_keys.example.py api_keys.py   # macOS/Linux: cp api_keys.example.py api_keys.py
# edit api_keys.py and add your Cloudflare account ID and API token
python run_trio_local.py
```

Then open **http://127.0.0.1:7860**, enter a prompt and generate. Click **Tricoherence Score** under the results to score them.

> **First run:** the Tricoherence score downloads the ImageBind checkpoint, about **4.8 GB**, into `.checkpoints/`. Scoring waits until that finishes. Later runs start in seconds.

---

## Installation

### Requirements

| | |
|---|---|
| Python | **3.11** (tested on 3.11.9). 3.10 should also work; 3.12+ is untested with ImageBind. |
| OS | Windows 11 tested; macOS and Linux should work the same way. |
| RAM | 16 GB recommended. ImageBind-huge uses about 6 GB while scoring. |
| Disk | About 10 GB for packages and model checkpoints. |
| GPU | Optional. Everything runs on CPU. Generation happens in the cloud, only scoring runs locally. |
| FFmpeg | **Not needed.** Audio is decoded with `miniaudio`. |
| Git | Needed, because ImageBind installs straight from GitHub. |

### Steps

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a **GPU**, install the CUDA build of `torch`, `torchvision` and `torchaudio` first from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.

`requirements.txt` is grouped by purpose: core, deep learning, web app, providers, scoring, ImageBind, benchmarking and scoring API. If you only want the web app, you can skip the last two groups.

---

## API keys

All keys stay on your machine. `api_keys.py` and every `.env` file are in `.gitignore`, so **never commit them**.

### For the web app: `api_keys.py`

```bash
copy api_keys.example.py api_keys.py   # macOS/Linux: cp ...
```

| Key | Required? | Where to get it |
|---|---|---|
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | **Yes.** All default models run on Cloudflare. | Cloudflare dashboard → Workers AI → *Use REST API* |
| `CLOUDFLARE_POOL` | Optional | Several free accounts, each with 10,000 neurons a day. The app moves to the next account when one fails or runs out. |
| `GROQ_KEY` | Only for Groq text models | [console.groq.com/keys](https://console.groq.com/keys) |
| `GEMINI_API_KEY` | Only for Gemini text or TTS | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

> **Don't rename `api_keys.py` to `secrets.py`.** A file called `secrets.py` in the project root shadows Python's built-in `secrets` module and breaks `transformers` on import.

### For the evaluation pipeline: `.env` files

```ini
# pipeline/.env
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
GEMINI_API_KEY=...

# evaluation/.env
GEMINI_API_KEY=...
```

---

## Running the web app

```bash
python run_trio_local.py
```

`run_trio_local.py` loads `api_keys.py`, finds a Cloudflare account that still has quota, and starts `hf_app.py` on **http://127.0.0.1:7860**.

**What you get for each prompt:**

1. **Text**, **Image** and **Audio** cards. You can copy the text, download the PNG, play the audio and read the Whisper transcript.
2. **Tricoherence Score** button: scores the three outputs with ImageBind and shows:
   - **s1, s2, s3**: raw cosine similarity for text↔image, text↔audio and image↔audio,
   - **f1, f2, f3**: those similarities weighted by per-output quality,
   - **Tricoherence v2**: the likelihood-ratio score. Above 0 means **Good** (the three outputs agree), below 0 means **Not-Good**.

You can pick models from the dropdowns. Text, image and audio all offer several Cloudflare models; Groq and Gemini models appear only when their keys are set.

> **Hosted demo vs local.** The Hugging Face Space runs on ZeroGPU, which cannot keep ImageBind loaded. It approximates s1–s3 with CLIP and BERTScore on the Whisper transcript and uses hand-set thresholds. **Only the local app computes the real ImageBind score** with the parameters fitted on 500 prompts.

---

## Reproducing the evaluation

### Just look at the results (no keys, no downloads)

The final CSVs are already in `results/`. Open the notebook from the project root:

```bash
pip install notebook
jupyter notebook visualize_results.ipynb
```

### Re-run the full pipeline

Run from the project root, in order:

| Step | Command | What it does | Needs |
|---|---|---|---|
| 1 | `python pipeline/step1_generate.py` | Generates 500 × (text, image, audio) with Cloudflare. Resumes if interrupted. | `pipeline/.env` (Cloudflare) |
| 2 | `python pipeline/step2_quality.py` | Q_text, Q_image, Q_audio, the Gemini judge, learned weights and ImageBind | `pipeline/.env` (Gemini), ImageBind |
| 3 | `python pipeline/step3_nb_lr.py` | Naive Bayes and Likelihood Ratio scoring | output of step 2 |
| 4 | `python pipeline/run_v2.py` | Final V2 run: NB and LR on the ImageBind-weighted features, with leave-one-out cross-validation | see note below |
| 5 | `python pipeline/compare.py` | Prints the comparison table | see note below |

> **Before running steps 4–5, and `pipeline/imagebind.py`, `pipeline/process_raw.py` or `pipeline/run_v1.py`:**
> - **Hard-coded file paths.** These scripts point at files on the original author's machine (`raw_scores_500.csv`, `judge_gemini_gemini-3.1-flash-lite.csv`, the `outputs500/` folder). Edit the path constants near the top of each script so they point at your own step 1–2 outputs.
> - **`pipeline/compare.py` reads `pipeline/scores/all_500_v2_results.csv`.** Either copy `results/all_500_v2_results.csv` into `pipeline/scores/`, or change the path in the script.
> - **Run `pipeline/imagebind.py` as a module:** `python -m pipeline.imagebind`. Running `python pipeline/imagebind.py` fails, because the script has the same name as the `imagebind` package it imports.

Step 1 costs Cloudflare quota (500 images and 500 audio clips), and step 2 makes 500 Gemini calls. Expect several hours in total on CPU.

### Earlier 53-prompt evaluation

`evaluation/full_pipeline.py` runs every stage on the 53-prompt set in `evaluation/prompts_with_outputs.csv` and writes `evaluation/complete_scores.csv`. It needs `evaluation/.env`.

---

## Other tools

| Tool | Command | Notes |
|---|---|---|
| Streamlit generator (older UI) | `streamlit run benchmarking/frontend.py` | Reads `api_keys.py` |
| Cloudflare benchmark (56 prompts) | `python benchmarking/cloudflare_benchmark.py` | Text: BERTScore, ROUGE, readability. Audio: DNSMOS. See `benchmarking/README.md` |
| Cloudflare vs other providers | `python benchmarking/cf_vs_bigplayers.py` | Comparison graphs |
| LR scoring REST API | `cd scoring && uvicorn lr_scorer:app --port 8000` | `POST /score`, `GET /params`. Lightweight quality-only scorer, no ImageBind |

---

## Project structure

```
Collaborative-Project/
│
├── run_trio_local.py          ← START HERE: local launcher (loads keys, picks a live account, starts the app)
├── hf_app.py                  Gradio web app: generation + Tricoherence scoring (ImageBind)
├── api_keys.example.py        template for api_keys.py (copy it, fill it in, never commit it)
├── requirements.txt           all Python dependencies, grouped by purpose
├── config.example.py          key template for the multi-provider benchmark scripts
├── visualize_results.ipynb    charts of the final 500-prompt results (reads results/)
│
├── pipeline/                  500-prompt evaluation pipeline (run in order, see above)
│   ├── step1_generate.py      generate text/image/audio via Cloudflare Workers AI
│   ├── step2_quality.py       quality scores + Gemini judge + ImageBind + WTSAS
│   ├── step3_nb_lr.py         Naive Bayes + Likelihood Ratio
│   ├── run_v1.py              V1: NB/LR on quality features [Q_text, Q_image, Q_audio]
│   ├── run_v2.py              V2: NB/LR on ImageBind-weighted features (best results)
│   ├── imagebind.py           ImageBind coherence s1/s2/s3 + WTSAS for 500 prompts
│   ├── process_raw.py         merge teammates' raw feature files
│   ├── compare.py             final comparison table
│   └── prompts_500.csv        the 500 input prompts
│
├── evaluation/                earlier 53-prompt evaluation (reference implementation)
│   ├── full_pipeline.py       all five stages in one file
│   ├── quality_scores.py      CLIP, aesthetic, BERTScore, WER scoring
│   ├── llm_judge.py           Gemini judge
│   ├── learn_weights.py       linear regression → w1, w2, w3
│   ├── imagebind_score.py     cross-modal coherence
│   ├── wtsas.py               quality-weighted alignment score
│   ├── naive_bayes_score*.py  NB scorer (v1, v2)
│   ├── likelihood_ratio_score*.py  LR scorer (v1, v2)
│   ├── clap_comparison.py     CLAP audio-text baseline
│   └── *.csv                  inputs and scores for the 53 prompts
│
├── results/                   final output CSVs (500 prompts)
│   ├── final_3approaches.csv  key result: WTSAS vs NB vs LR
│   ├── all_500_v2_results.csv NB and LR scores per prompt
│   ├── imagebind_500.csv      raw s1/s2/s3 per prompt
│   └── ...                    V1 results, summaries, WTSAS
│
├── scoring/                   standalone scorer
│   ├── lr_scorer.py           FastAPI server: POST /score, GET /params
│   └── parameter.txt          derivation notes: TSAS → WTSAS → NB → LR
│
├── benchmarking/              provider benchmarks + Streamlit UI
│   ├── frontend.py            Streamlit generator
│   ├── cloudflare_benchmark.py
│   ├── cf_vs_bigplayers.py
│   ├── cloudflare_visualizations.ipynb
│   └── other_models/          Groq / Mistral / Cerebras / Stability / Deepgram comparison
│
├── testing multiple models/   early experiments and model comparisons (archive)
├── research_papers/           reference PDFs
└── contributions.docx/.html   team contribution statement
```

**Not in the repo (created locally, gitignored):** `api_keys.py`, `pipeline/.env`, `evaluation/.env`, `.checkpoints/` (ImageBind weights), `pipeline/outputs/` (generated media).

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'token_hex' from 'secrets'` | A `secrets.py` file in the project root shadows Python's module | Rename it. Keys belong in `api_keys.py`. |
| `Error 404: Could not route to /client/v4/accounts/ai/run/...` | Cloudflare account ID is empty | Fill in `CLOUDFLARE_ACCOUNT_ID` in `api_keys.py` |
| `429` / quota errors | Daily free Cloudflare quota used up | Add more accounts to `CLOUDFLARE_POOL` or wait for the reset (00:00 UTC) |
| `numpy.core.multiarray failed to import` | numpy 2.x or a broken numpy install | `pip install "numpy<2.0" --force-reinstall` |
| `RuntimeError: Failed to create AudioDecoder ... libtorchcodec` | torchaudio tried to use TorchCodec/FFmpeg | `pip install miniaudio`. The app decodes audio with it and never calls `torchaudio.load`. |
| `imagebind is not a package` when running `pipeline/imagebind.py` | The script name shadows the package | `python -m pipeline.imagebind` |
| Scoring is slow the first time | ImageBind (4.8 GB) and Whisper are downloading | Wait once; they're cached afterwards |

---

## Results

500 prompts, leave-one-out cross-validation, compared with the Gemini judge's ratings:

| Method | Spearman ρ | Pearson r | Notes |
|--------|-----------|----------|-------|
| Likelihood Ratio | **0.195** | **0.205** | Best composite, recommended |
| Naive Bayes | 0.190 | 0.193 | Statistically tied with LR |
| TSAS (plain averaging) | 0.137 | 0.121 | No labels needed |
| WTSAS (quality-weighted) | 0.082 | 0.076 | Withdrawn: the weighting hurts |
| s1 alone (text↔image cosine) | 0.203 | 0.201 | Beats all composites |

**Key finding:** the likelihood ratio is the best composite scorer, but the raw text–image cosine similarity on its own beats every composite method. Coherence and quality are only weakly related (ρ ≈ 0.20): a high-quality output doesn't guarantee that the text, image and audio describe the same thing.

---

## How the Tricoherence score works

```
Prompt
  │
  ├─[Text model]──[Image model]──[Audio model]      (Cloudflare Workers AI)
  │
  ▼
Per-output quality
  Q_text  = BERTScore(generated text, prompt)
  Q_image = avg(CLIP, aesthetic) − λ·var
  Q_audio = avg(semantic, 1−WER) − λ·var            (Whisper transcript)
  │
  ▼
Cross-modal coherence (ImageBind embeddings)
  s1 = text ↔ image cosine similarity
  s2 = text ↔ audio cosine similarity
  s3 = image ↔ audio cosine similarity
  │
  ▼
Quality-weighted features         w1=7.87  w2=2.28  w3=0.78 (learned by linear regression)
  f1 = s1 × (w1·Qt + w2·Qi) / (w1+w2)
  f2 = s2 × (w1·Qt + w3·Qa) / (w1+w3)
  f3 = s3 × (w2·Qi + w3·Qa) / (w2+w3)
  │
  ▼
Likelihood Ratio
  S(p) = Σᵢ [ log P(fᵢ|Good) − log P(fᵢ|Not-Good) ]
  S > 0 → Good     S < 0 → Not-Good
```

### Fitted parameters

Fitted on 500 prompts with leave-one-out cross-validation. `Good` means a judge mean above 3.5 (273 Good, 227 Not-Good).

| Feature | μ⁺ (Good) | σ⁺ | μ⁻ (Not-Good) | σ⁻ |
|---------|-----------|-----|--------------|-----|
| f1 | 0.3062 | 0.0344 | 0.2902 | 0.0358 |
| f2 | 0.0545 | 0.0416 | 0.0532 | 0.0421 |
| f3 | 0.0487 | 0.0274 | 0.0468 | 0.0280 |

The Good and Not-Good means are close together, especially for f2 and f3, which is why the score separates the two classes only weakly (r ≈ 0.20).

---

## Models used

| Role | Model | Provider |
|------|-------|----------|
| Text (default) | Llama 4 Scout 17B · also Llama 3.1 8B, Kimi K2.6, GPT-OSS 20B | Cloudflare Workers AI |
| Image (default) | FLUX.1 [schnell] · also FLUX.2 Dev, Leonardo Lucid Origin, Phoenix 1.0 | Cloudflare Workers AI |
| Audio (default) | Deepgram Aura-1 · also Aura-2, MeloTTS | Cloudflare Workers AI |
| Judge | Gemini 3.1 Flash Lite | Google AI Studio |
| Coherence | ImageBind-Huge | Meta AI (local) |
| Quality | CLIP, BERTScore, Whisper base | OpenAI / local |

---

## References

| Paper | What we used from it |
|-------|---------------------|
| Girdhar et al., *ImageBind* (Meta, CVPR 2023) | Cross-modal embeddings s1, s2, s3 |
| Nandakumar et al., *LR Biometric Fusion* (IEEE TPAMI 2008) | The LR scoring method |
| Wang et al., *Multimodal Diffusion for Text–Image–Audio* | TSAS coherence framing |
| Lu et al., *Multimodal Consistency* (ACL Findings 2025) | Judge-and-regression design |
| Dhimoïla et al., *Cross-Modal Redundancy* (ICLR 2026) | Why s1 ≫ s2 and s3 |
| Dosovitskiy et al., *ViT* (ICLR 2021) | Image → vector encoding |
| Gong et al., *AST* (2021) | Audio spectrogram preprocessing |

PDFs are in `research_papers/`.
