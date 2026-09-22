# Multimodal AI Content Generator

**M.Sc. Applied Data Science & Artificial Intelligence — Case Studies 1**
**SRH University of Applied Sciences, Heidelberg · September 2026**

**Team:** Namrata Bhoyar · Nihal Pujari · Pramodkumar Shivanna · Anuj Kamble · Gourav Somanna

---

## What This Project Does

Give it one short prompt. Get back three things at once.

The system takes a text description — *"an owl librarian wearing glasses"* — and returns a written paragraph about it, an AI-generated image of it, and a spoken audio narration of that paragraph. Three separate models do the work independently, each from the same prompt, and their outputs are shown together in a live Gradio web app.

The harder problem — and the main contribution — is **evaluating whether the three outputs actually match each other**. Nothing guarantees they will, because each model never sees what the others produced. We built a five-layer evaluation pipeline that compares three scoring methods (averaging, Naive Bayes, Likelihood Ratio) against ratings from an LLM judge across 500 prompts.

---

## Live Demo (Gradio App)

```bash
# Start the app — automatically probes the 9-account Cloudflare pool and picks
# a live account, then launches the Gradio UI
python run_trio_local.py
```

Open **http://localhost:7860** · Enter a prompt · Get text + image + audio + a live LR score badge.

The badge shows:
- **Good / Not-Good** — whether the three outputs are coherent with each other
- **LR score** — positive means coherent, negative means not
- **Quality bars** — per-modality quality (text similarity, image–prompt match, audio)
- **text↔image** — raw cosine similarity between the text and image embeddings

---

## Scoring Results (500 prompts, LOO cross-validation)

| Method | Spearman ρ | Pearson r | Notes |
|--------|-----------|----------|-------|
| Likelihood Ratio | **0.195** | **0.205** | Best composite — recommended |
| Naive Bayes | 0.190 | 0.193 | Statistically tied with LR |
| TSAS (plain averaging) | 0.137 | 0.121 | No labels needed |
| WTSAS (quality-weighted) | 0.082 | 0.076 | Withdrawn — weighting hurts |
| s1 alone (text↔image cosine) | 0.203 | 0.201 | Beats all composites |

**Key finding:** The likelihood ratio is the best composite scorer, but the raw text–image cosine similarity alone beats every composite method. The main result is that coherence and quality are weakly related (ρ ≈ 0.20): a high-quality output does not guarantee that text, image and audio describe the same thing.

---

## Project Structure

```
Collaborative-Project/
│
├── benchmarking/              ← Streamlit app + Cloudflare provider benchmarks
│   ├── frontend.py            the live web app (text + image + audio generator)
│   ├── cloudflare_benchmark.py
│   ├── cf_vs_bigplayers.py    Cloudflare vs GPT-4 / Mistral / Groq
│   └── outputs/               56-prompt benchmark graphs + scores
│
├── pipeline/                  ← 500-prompt evaluation pipeline (run in order)
│   ├── step1_generate.py      generate text/image/audio via Cloudflare AI
│   ├── step2_quality.py       compute Q_text, Q_image, Q_audio + LLM judge
│   ├── step3_nb_lr.py         Naive Bayes + Likelihood Ratio scoring
│   ├── imagebind.py           ImageBind cross-modal coherence (s1, s2, s3)
│   ├── run_v2.py              full V2 pipeline (best results)
│   ├── compare.py             final comparison of all approaches
│   └── prompts_500.csv        the 500 input prompts
│
├── evaluation/                ← earlier 53-prompt evaluation (reference)
│   ├── imagebind_score.py
│   ├── quality_scores.py      CLIP, aesthetic, BERTScore, WER scoring
│   ├── learn_weights.py       linear regression → learned weights w1, w2, w3
│   ├── naive_bayes_score_v2.py
│   ├── likelihood_ratio_score_v2.py
│   └── llm_judge.py           Gemini judge
│
├── results/                   ← output CSVs
│   ├── final_3approaches.csv  KEY RESULT: all three methods, 500 prompts
│   ├── all_500_v2_results.csv NB and LR scores
│   └── imagebind_500.csv      raw s1/s2/s3 coherence scores
│
├── research_papers/           ← reference PDFs
│
├── scoring/                   ← final LR approach (the live scorer)
│   ├── lr_scorer.py           FastAPI server — POST /score, GET /params
│   └── parameter.txt          derivation notes: TSAS → WTSAS → NB → LR evolution
│
├── hf_app.py                  main Gradio app (text + image + audio + live scoring)
├── run_trio_local.py          local launcher — probes 9-account Cloudflare pool
├── config.example.py          API key template (safe to commit)
└── requirements.txt
```

---

## How to Run the Evaluation Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

You also need:
- Cloudflare Workers AI account → add keys to `pipeline/.env`
- Gemini API key → add to `evaluation/.env`
- ImageBind: `pip install git+https://github.com/facebookresearch/ImageBind`

### Steps

```bash
python pipeline/step1_generate.py   # generate 500 × 3 outputs
python pipeline/step2_quality.py    # quality scores + LLM judge ratings
python pipeline/run_v2.py           # NB + LR (best: NB ρ=0.190, LR ρ=0.195)
python pipeline/imagebind.py        # cross-modal coherence
python pipeline/compare.py          # final results table
```

---

## Evaluation Architecture

```
Prompt
  │
  ├─[Text: Llama 3.1 8B]──[Image: FLUX.1]──[Audio: MeloTTS]
  │
  ▼
Per-modality quality
  Q_text (sentence-transformer sim)
  Q_image (CLIP sim)
  Q_audio (0.75 default / Whisper optional)
  │
  ▼
Cross-modal coherence (ImageBind embeddings)
  s1 = text ↔ image cosine similarity
  s2 = text ↔ audio cosine similarity
  s3 = image ↔ audio cosine similarity
  │
  ▼
Quality-weighted features
  f1 = s1 × (w1·Qt + w2·Qi) / (w1+w2)    w1=7.87 w2=2.28 w3=0.78
  f2 = s2 × (w1·Qt + w3·Qa) / (w1+w3)    (learned by linear regression on 500 prompts)
  f3 = s3 × (w2·Qi + w3·Qa) / (w2+w3)
  │
  ▼
Likelihood Ratio score (best method)
  S(p) = Σᵢ [ log P(fᵢ|Good) − log P(fᵢ|Not-Good) ]
  S > 0 → Good   S < 0 → Not-Good
```

---

## Likelihood Ratio — Fitted Parameters

Trained on 500 prompts with leave-one-out cross-validation. `Good` = judge mean > 3.5.

**Pipeline parameters** (f1/f2/f3 = ImageBind-weighted cross-modal features):

| Feature | μ⁺ (Good) | σ⁺ | μ⁻ (Not-Good) | σ⁻ |
|---------|-----------|-----|--------------|-----|
| f1 | 0.3062 | 0.0344 | 0.2902 | 0.0358 |
| f2 | 0.0545 | 0.0416 | 0.0532 | 0.0421 |
| f3 | 0.0487 | 0.0274 | 0.0468 | 0.0280 |

**Live-app parameters** (s1/s2/s3 = BERTScore / CLIP similarity / audio quality — lighter features that run in real time without ImageBind):

| Feature | μ⁺ (Good) | σ⁺ | μ⁻ (Not-Good) | σ⁻ |
|---------|-----------|-----|--------------|-----|
| s1 (BERTScore) | 0.8400 | 0.0088 | 0.8372 | 0.0097 |
| s2 (CLIP sim) | 0.4656 | 0.0255 | 0.4585 | 0.0245 |
| s3 (audio quality) | 0.8723 | 0.0153 | 0.8706 | 0.0153 |

---

## Models Used

| Modality | Model | Provider |
|----------|-------|----------|
| Text | Llama 4 Scout 17B (default) | Cloudflare Workers AI |
| Image | FLUX.1 [schnell] | Cloudflare Workers AI |
| Audio | Deepgram Aura-1 (default) | Cloudflare Workers AI |
| Judge | Gemini 1.5 Flash Lite | Google AI Studio |
| Coherence | ImageBind-Huge | Meta AI (local) |
| Quality | CLIP ViT-B/32 + CLAP | OpenAI / LAION (local) |

---

## References

| Paper | What we used from it |
|-------|---------------------|
| Girdhar et al. — *ImageBind* (Meta, CVPR 2023) | Cross-modal embeddings s1, s2, s3 |
| Nandakumar et al. — *LR Biometric Fusion* (IEEE TPAMI 2008) | The LR scoring method |
| Wang et al. — *Multimodal Diffusion for Text–Image–Audio* | TSAS coherence framing |
| Lu et al. — *Multimodal Consistency* (ACL Findings 2025) | Judge-and-regression design |
| Dhimoïla et al. — *Cross-Modal Redundancy* (ICLR 2026) | Why s1 ≫ s2 and s3 |
| Dosovitskiy et al. — *ViT* (ICLR 2021) | Image → vector encoding |
| Gong et al. — *AST* (2021) | Audio spectrogram preprocessing |
