# Tri-Modal Alignment Score (TSAS / WTSAS)

**SRH University of Applied Sciences, Heidelberg**
M.Sc. Applied Data Science and Artificial Intelligence — Case Studies 1

**Team:** Namrata · Nihal · Pramod · Anuj · Gourav

---

## What This Project Does

We evaluate how well Cloudflare Workers AI generates coherent tri-modal content (text + image + audio) from a single text prompt. Given a prompt like *"A fox exploring a cave in a snowy village"*, the system generates a text description, an image, and audio narration — then scores how well the three outputs match each other and how good the overall quality is.

We compare three automated scoring approaches against ratings from an LLM judge (Gemini):

| Approach | Method | Pearson r with LLM Judge |
|----------|--------|--------------------------|
| 1 — WTSAS | Quality-weighted tri-modal coherence (ImageBind) | 0.0763 |
| 2 — Naive Bayes | P(Good) on weighted coherence features, LOO CV | 0.1931 |
| 3 — Likelihood Ratio | log P(Good) - log P(Rest), LOO CV | **0.2047** |

**Key finding:** Tri-modal coherence is only weakly predictive of perceived quality (r ≈ 0.20, pairwise accuracy ≈ 57%). Coherence and quality are distinct dimensions of multimodal generation.

---

## Project Structure

```
Collaborative-Project/
│
├── pipeline/                  ← main 500-prompt pipeline (run these in order)
│   ├── step1_generate.py      step 1: generate text/image/audio via Cloudflare AI
│   ├── step2_quality.py       step 2: compute Q_text, Q_image, Q_audio + LLM judge
│   ├── step3_nb_lr.py         step 3: Naive Bayes + Likelihood Ratio on Q features (V1)
│   ├── imagebind.py           compute ImageBind coherence scores (s1, s2, s3) + WTSAS
│   ├── run_v1.py              V1 pipeline: NB + LR on raw quality features
│   ├── run_v2.py              V2 pipeline: NB + LR on quality-weighted coherence features
│   ├── compare.py             compare all 3 approaches with Pearson r + pairwise accuracy
│   ├── process_raw.py         merge and clean raw score CSVs
│   ├── prompts_500.csv        the 500 input prompts
│   └── .env                   Cloudflare API keys (gitignored — not committed)
│
├── results/                   ← all output score CSVs
│   ├── final_3approaches.csv      KEY RESULT: WTSAS + NB + LR for all 500 prompts
│   ├── all_500_v2_results.csv     V2 NB (r=0.1931) and LR (r=0.2047) scores
│   ├── all_500_v2_summary.csv     V2 Pearson r comparison table
│   ├── imagebind_500.csv          raw s1/s2/s3/TSAS scores, 500 prompts
│   ├── wtsas_500.csv              WTSAS final scores, 500 prompts
│   ├── all_500_results.csv        V1 NB + LR scores
│   └── all_500_summary.csv        V1 Pearson r comparison table
│
├── evaluation/                ← 53-prompt evaluation pipeline (earlier work, reference)
│   ├── imagebind_score.py     original ImageBind scoring (working reference)
│   ├── full_pipeline.py       end-to-end 53-prompt pipeline
│   ├── quality_scores.py      CLIP, aesthetic, BERTScore, WER scoring
│   ├── learn_weights.py       linear regression to learn w1, w2, w3
│   ├── wtsas.py               WTSAS formula implementation
│   ├── naive_bayes_score.py   Naive Bayes V1 (raw Q features)
│   ├── naive_bayes_score_v2.py Naive Bayes V2 (weighted coherence features)
│   ├── likelihood_ratio_score.py  LR V1
│   ├── likelihood_ratio_score_v2.py LR V2
│   ├── llm_judge.py           Gemini judge scoring
│   ├── aesthetic_weights.pth  pretrained aesthetic scorer weights (gitignored)
│   └── .env                   Gemini API key (gitignored)
│
├── benchmarking/              ← early exploration: Cloudflare vs GPT-4 / Mistral / Groq
│   ├── cloudflare_benchmark.py
│   ├── cf_vs_bigplayers.py
│   ├── cloudflare_visualizations.ipynb
│   ├── outputs/               56-prompt test outputs (graphs + scores)
│   └── other_models/          Groq / Mistral / Cerebras benchmarks
│
├── research_papers/           ← reference PDFs (ImageBind, TSAS, LR fusion, etc.)
│
├── config.py                  Cloudflare model IDs and endpoint config
├── config.example.py          template (safe to commit, no keys)
├── requirements.txt           Python dependencies
└── parameter.txt              hyperparameter notes
```

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

You also need:
- A Cloudflare Workers AI account — add keys to `pipeline/.env`
- A Gemini API key — add to `evaluation/.env`
- The [ImageBind model](https://github.com/facebookresearch/ImageBind) installed in the repo root

### Step-by-step

```bash
# 1. Generate text, image, audio for all 500 prompts
python pipeline/step1_generate.py

# 2. Compute quality scores + LLM judge ratings
python pipeline/step2_quality.py

# 3. Run V2 pipeline (best results: NB r=0.193, LR r=0.205)
python pipeline/run_v2.py

# 4. Compute ImageBind coherence + WTSAS
python pipeline/imagebind.py

# 5. See the final comparison
python pipeline/compare.py
```

---

## Scoring Formulas

### Quality scores (per modality)
```
Q_text  = BERTScore(generated_text, prompt)
Q_image = quality_pair(CLIP_score, aesthetic_score)
Q_audio = quality_pair(semantic_score, WER_inv)

quality_pair(a, b) = avg(a, b) - 0.5 * variance(a, b)
```

### ImageBind coherence scores
```
s1 = cosine_sim(text_embedding,  image_embedding)   # text <-> image
s2 = cosine_sim(text_embedding,  audio_embedding)   # text <-> audio
s3 = cosine_sim(image_embedding, audio_embedding)   # image <-> audio
TSAS = (s1 + s2 + s3) / 3
```

### Weighted coherence scores (V2)
Weights w1, w2, w3 are learned by regressing Q features on judge_mean:
```
s1_w = s1 * (w1*Q_text + w2*Q_image) / (w1 + w2)
s2_w = s2 * (w1*Q_text + w3*Q_audio) / (w1 + w3)
s3_w = s3 * (w2*Q_image + w3*Q_audio) / (w2 + w3)

Learned: w1=7.87 (text), w2=2.28 (image), w3=0.78 (audio)
```

### WTSAS (Approach 1)
```
WTSAS = avg(s1_w, s2_w, s3_w) - lambda * variance(s1_w, s2_w, s3_w)
lambda = 1.8  (optimal from 53-prompt sweep)
```

---

## Results Summary

```
Approach                            Pearson r   Pairwise Accuracy
───────────────────────────────────────────────────────────────────
WTSAS (coherence only)                0.0763         52.4%
Naive Bayes on [s1_w, s2_w, s3_w]    0.1931         56.2%
Likelihood Ratio on [s1_w,s2_w,s3_w] 0.2047         56.6%  <- BEST
───────────────────────────────────────────────────────────────────
Random baseline                       0.0000         50.0%
```

LR wins on both Pearson r and Spearman r. LOO cross-validation was used throughout for honest generalization estimates.

---

## References

See `research_papers/` for the full PDFs.

- Girdhar et al. — *ImageBind: One Embedding Space to Bind Them All* (Meta AI, 2023)
- Nandakumar et al. — *Likelihood Ratio Fusion*
- `cosine_sim_tsas.pdf` — TSAS methodology
- `multimodal_consistency_coherence.pdf` — coherence evaluation background