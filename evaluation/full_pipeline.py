"""
=============================================================
  Tri-modal Evaluation Pipeline  —  All Stages in One File
=============================================================

INPUT  : prompts_with_outputs.csv
         Columns: prompt_id, prompt, generated_text, image_path, audio_path

OUTPUT : complete_scores.csv
         All quality scores + LLM ratings + learned weights + coherence + WTSAS

Stages:
  1. Quality Scores   Q_text, Q_image, Q_audio
  2. LLM Judge        Gemini rates each prompt overall + per-modality (1-10)
  3. Learn Weights    Linear Regression on [Q_text, Q_image, Q_audio] -> y=llm_overall
  4. ImageBind        s1 (text<->image), s2 (text<->audio), s3 (image<->audio)
  5. WTSAS            Quality-Weighted Tri-modal Alignment Score (final metric)

Usage:
  1. Edit CONFIG section below (paths + Gemini key)
  2. pip install bert-score transformers torch clip-by-openai
              whisper-openai soundfile jiwer scipy pandas
              google-genai python-dotenv imagebind
  3. python full_pipeline.py
=============================================================
"""

import os
import sys
import re
import time
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

# ============================================================
# CONFIG  — edit these for your setup
# ============================================================
_HERE            = Path(__file__).parent
INPUT_CSV        = str(_HERE / "prompts_with_outputs.csv")   # your input file
OUTPUT_CSV       = str(_HERE / "complete_scores.csv")         # final output (all metrics)
CHECKPOINT_DIR   = str(_HERE / "pipeline_checkpoints")        # temp folder for crash recovery

GEMINI_API_KEY   = ""    # paste key here, or set GEMINI_API_KEY in a .env file
GEMINI_MODEL     = "gemini-2.0-flash"

LAMBDA           = 0.5   # variance penalty for quality_pair and WTSAS

SKIP_IMAGEBIND   = False  # set True if ImageBind not installed / no GPU
# ============================================================

load_dotenv()
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# ============================================================
#  LOAD MODELS  (done once at startup)
# ============================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")

print("Loading CLIP...")
from transformers import CLIPModel, CLIPProcessor
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("Loading Whisper...")
import whisper
whisper_model = whisper.load_model("base")

print("Loading Aesthetic model (cafeai/cafe_aesthetic)...")
from transformers import pipeline as hf_pipeline
aesthetic_pipe = hf_pipeline(
    "image-classification",
    model="cafeai/cafe_aesthetic",
    device=0 if DEVICE == "cuda" else -1,
)

print("Loading BERTScore...")
from bert_score import score as bert_score_fn

if not SKIP_IMAGEBIND:
    print("Loading ImageBind...")
    try:
        import torchaudio
        import soundfile as sf
        from scipy import signal as scipy_signal
        from imagebind import data as ib_data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        ib_model = imagebind_model.imagebind_huge(pretrained=True)
        ib_model.eval()
        ib_model.to(DEVICE)
        IMAGEBIND_OK = True
        print("ImageBind loaded.")
    except Exception as e:
        print(f"[WARN] ImageBind failed to load: {e}. Skipping Stage 4.")
        IMAGEBIND_OK = False
else:
    import soundfile as sf
    from scipy import signal as scipy_signal
    IMAGEBIND_OK = False

print("Loading Gemini client...")
from google import genai
from google.genai import types
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

from jiwer import wer as compute_wer
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("\nAll models ready.\n")


# ============================================================
#  SHARED HELPERS
# ============================================================
def quality_pair(a, b, lam=LAMBDA):
    avg = (a + b) / 2
    var = float(np.var([a, b]))
    return float(np.clip(avg - lam * var, 0, 1))

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    return scipy_signal.resample(audio, n).astype("float32")


# ============================================================
#  STAGE 1 — QUALITY SCORES
# ============================================================
print("=" * 60)
print("  STAGE 1 — Quality Scores")
print("=" * 60)

STAGE1_CSV = Path(CHECKPOINT_DIR) / "stage1_quality.csv"

def get_q_text(generated_text, prompt):
    _, _, F1 = bert_score_fn([generated_text], [prompt], lang="en", verbose=False)
    return float(F1.mean())

def get_clip_score(prompt, image_path):
    image = Image.open(image_path).convert("RGB")
    t_in  = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    i_in  = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        t_emb = clip_model.get_text_features(**t_in).squeeze().numpy()
        i_emb = clip_model.get_image_features(**i_in).squeeze().numpy()
    return float(np.clip(cosine_sim(t_emb, i_emb), 0, 1))

def get_aesthetic_score(image_path):
    result = aesthetic_pipe(str(image_path))
    for r in result:
        if r["label"] == "aesthetic":
            return float(r["score"])
    return 0.5

def transcribe_audio(audio_path):
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample_audio(audio, sr, 16000)
        return whisper_model.transcribe(audio)["text"].strip()
    except Exception as e:
        print(f"  [WARN] Whisper error: {e}")
        return ""

def get_q_audio(prompt, audio_path, generated_text):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return 0.0, 0.0, 0.0, ""
    _, _, F1   = bert_score_fn([transcript], [prompt], lang="en", verbose=False)
    semantic_s = float(F1.mean())
    try:
        err     = float(np.clip(compute_wer(generated_text.lower(), transcript.lower()), 0, 1))
        wer_inv = 1 - err
    except Exception:
        wer_inv = 0.5
    q_audio = quality_pair(semantic_s, wer_inv)
    return semantic_s, wer_inv, q_audio, transcript

# -- run stage 1 -------------------------------------------------------------
if STAGE1_CSV.exists():
    print(f"  [RESUME] Found {STAGE1_CSV}, loading saved quality scores...")
    df_q = pd.read_csv(STAGE1_CSV)
else:
    df_in = pd.read_csv(INPUT_CSV)
    required = {"prompt_id", "prompt", "generated_text", "image_path", "audio_path"}
    missing  = required - set(df_in.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    rows = []
    n    = len(df_in)
    for i, row in df_in.iterrows():
        pid   = int(row["prompt_id"])
        pmpt  = str(row["prompt"])
        gtext = str(row["generated_text"])
        imgp  = str(row["image_path"])
        audp  = str(row["audio_path"])
        print(f"  [{i+1:03d}/{n}] {pmpt[:55]}...")

        # Q_text
        try:
            q_text = get_q_text(gtext, pmpt)
        except Exception as e:
            print(f"    [WARN] Q_text error: {e}")
            q_text = 0.5

        # Q_image
        try:
            clip_s = get_clip_score(pmpt, imgp)
            aes_s  = get_aesthetic_score(imgp)
            q_img  = quality_pair(clip_s, aes_s)
        except Exception as e:
            print(f"    [WARN] Q_image error: {e}")
            clip_s, aes_s, q_img = 0.5, 0.5, 0.5

        # Q_audio
        try:
            sem_s, wer_inv, q_aud, transcript = get_q_audio(pmpt, audp, gtext)
        except Exception as e:
            print(f"    [WARN] Q_audio error: {e}")
            sem_s, wer_inv, q_aud, transcript = 0.5, 0.5, 0.5, ""

        rows.append({
            "prompt_id":      pid,
            "prompt":         pmpt,
            "generated_text": gtext,
            "image_path":     imgp,
            "audio_path":     audp,
            "q_text":         round(q_text, 4),
            "clip_score":     round(clip_s,  4),
            "aesthetic":      round(aes_s,   4),
            "q_image":        round(q_img,   4),
            "semantic_score": round(sem_s,   4),
            "wer_inv":        round(wer_inv, 4),
            "q_audio":        round(q_aud,   4),
            "transcript":     transcript,
        })
        print(f"    Q_text={q_text:.4f}  Q_image={q_img:.4f}  Q_audio={q_aud:.4f}")

    df_q = pd.DataFrame(rows)
    df_q.to_csv(STAGE1_CSV, index=False)
    print(f"\n  Stage 1 saved to {STAGE1_CSV}\n")

print(f"\n  Stage 1 complete. Samples: {len(df_q)}")
print(f"  Q_text avg={df_q['q_text'].mean():.4f}  "
      f"Q_image avg={df_q['q_image'].mean():.4f}  "
      f"Q_audio avg={df_q['q_audio'].mean():.4f}")


# ============================================================
#  STAGE 2 — LLM JUDGE (Gemini)
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 2 — LLM Judge (Gemini)")
print("=" * 60)

STAGE2_CSV = Path(CHECKPOINT_DIR) / "stage2_llm_ratings.csv"

JUDGE_PROMPT = """You are an expert evaluator of AI-generated multimodal content.

You will be shown:
1. An original text prompt
2. AI-generated text based on that prompt
3. An AI-generated image based on that prompt
4. A transcript of AI-generated speech audio based on that prompt

Rate the OVERALL quality of all three outputs together on a scale of 1 to 10.
Also rate each modality individually.

Scoring guide:
  9-10 = Excellent across all modalities, highly consistent with the prompt
  7-8  = Good quality, minor issues in one modality
  5-6  = Average, noticeable issues but still relevant
  3-4  = Poor, significant issues in one or more modalities
  1-2  = Very poor, outputs largely irrelevant to the prompt

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{"overall_score": <1-10>, "text_score": <1-10>, "image_score": <1-10>, "audio_score": <1-10>, "reasoning": "<one sentence>"}
"""

def build_judge_message(prompt, generated_text, transcript):
    return (
        f'{JUDGE_PROMPT}\n\n'
        f'Original Prompt: "{prompt}"\n\n'
        f'Generated Text:\n{generated_text}\n\n'
        f'Audio Transcript:\n{transcript}\n\n'
        f'Please evaluate the image shown alongside this text.'
    )

def parse_rating(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    m = re.search(r'overall_score["\s:]+(\d+(?:\.\d+)?)', text)
    if m:
        return {"overall_score": float(m.group(1)), "text_score": 5,
                "image_score": 5, "audio_score": 5, "reasoning": "parsed fallback"}
    return None

def call_gemini(prompt, generated_text, transcript, image_path):
    user_msg = build_judge_message(prompt, generated_text, transcript)
    image    = Image.open(image_path).convert("RGB")
    for attempt in range(5):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[user_msg, image],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            result = parse_rating(resp.text)
            if result:
                return result
            print(f"    [WARN] Could not parse: {resp.text[:100]}")
            return None
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["503", "429", "UNAVAILABLE", "quota"]):
                wait = 10 * (attempt + 1)
                print(f"    [RETRY {attempt+1}/5] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] {msg[:100]}")
                return None
    print("    [FAILED] Max retries reached")
    return None

# -- run stage 2 (with checkpointing) ----------------------------------------
if STAGE2_CSV.exists():
    print(f"  [RESUME] Found {STAGE2_CSV}, loading saved ratings...")
    df_ratings = pd.read_csv(STAGE2_CSV)
    done_ids   = set(df_ratings["prompt_id"].tolist())
else:
    df_ratings = pd.DataFrame()
    done_ids   = set()

rows2 = df_ratings.to_dict("records") if not df_ratings.empty else []
n     = len(df_q)

for i, row in df_q.iterrows():
    pid = int(row["prompt_id"])
    if pid in done_ids:
        continue

    print(f"  [{i+1:03d}/{n}] Judging prompt {pid}...")
    result = call_gemini(
        row["prompt"], row["generated_text"],
        row["transcript"], row["image_path"]
    )
    if result:
        rows2.append({
            "prompt_id":    pid,
            "llm_overall":  float(result.get("overall_score", 5)),
            "llm_text":     float(result.get("text_score",    5)),
            "llm_image":    float(result.get("image_score",   5)),
            "llm_audio":    float(result.get("audio_score",   5)),
            "llm_reasoning":str(result.get("reasoning",       "")),
        })
        print(f"    overall={result.get('overall_score')}  "
              f"text={result.get('text_score')}  "
              f"image={result.get('image_score')}  "
              f"audio={result.get('audio_score')}")
    else:
        rows2.append({
            "prompt_id": pid, "llm_overall": 5, "llm_text": 5,
            "llm_image": 5,   "llm_audio":   5, "llm_reasoning": "FAILED",
        })

    # save checkpoint every 10 prompts
    if (i + 1) % 10 == 0:
        pd.DataFrame(rows2).to_csv(STAGE2_CSV, index=False)
        print(f"    [CHECKPOINT] Saved {len(rows2)} ratings so far...")

    time.sleep(1.0)

df_ratings = pd.DataFrame(rows2)
df_ratings.to_csv(STAGE2_CSV, index=False)

print(f"\n  Stage 2 complete. Rated: {len(df_ratings)} prompts")
print(f"  LLM overall avg = {df_ratings['llm_overall'].mean():.2f}")


# ============================================================
#  STAGE 3 — LEARN WEIGHTS (Linear Regression)
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 3 — Learn Weights (Linear Regression)")
print("=" * 60)

df_merge = pd.merge(df_q, df_ratings, on="prompt_id")

X_raw = df_merge[["q_text", "q_image", "q_audio"]].values
y_llm = df_merge["llm_overall"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw)

lr = LinearRegression()
lr.fit(X_sc, y_llm)

r2 = lr.score(X_sc, y_llm)

raw_w = lr.coef_
raw_w = np.maximum(raw_w, 0)         # clip negatives
w_sum = raw_w.sum()
if w_sum < 1e-8:
    raw_w = np.array([1/3, 1/3, 1/3])
    w_sum = 1.0
w1, w2, w3 = raw_w / w_sum

# Pearson correlations per feature
r_text,  _ = pearsonr(df_merge["q_text"].values,  y_llm)
r_image, _ = pearsonr(df_merge["q_image"].values, y_llm)
r_audio, _ = pearsonr(df_merge["q_audio"].values, y_llm)

print(f"\n  Pearson r  —  Q_text={r_text:.4f}  Q_image={r_image:.4f}  Q_audio={r_audio:.4f}")
print(f"  Raw coefs  —  {lr.coef_}")
print(f"  Norm weights — w1(text)={w1:.4f}  w2(image)={w2:.4f}  w3(audio)={w3:.4f}")
print(f"  R² (train) = {r2:.4f}")

# Lambda sweep (0.0 to 2.0 step 0.1) to find best variance penalty
LAMBDAS_SWEEP = np.arange(0.0, 2.05, 0.1)
best_lam, best_r = LAMBDA, -99

for lam in LAMBDAS_SWEEP:
    scores = []
    for _, row in df_merge.iterrows():
        qt, qi, qa = row["q_text"], row["q_image"], row["q_audio"]
        s = (w1*qt + w2*qi + w3*qa) - lam * float(np.var([qt, qi, qa]))
        scores.append(s)
    r, _ = pearsonr(scores, y_llm)
    if r > best_r:
        best_r   = r
        best_lam = lam

print(f"  Optimal lambda = {best_lam:.2f}  (r = {best_r:.4f})")
OPT_LAMBDA = best_lam


# ============================================================
#  STAGE 4 — IMAGEBIND COHERENCE
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 4 — ImageBind Coherence (s1, s2, s3)")
print("=" * 60)

STAGE4_CSV = Path(CHECKPOINT_DIR) / "stage4_imagebind.csv"

def load_audio_for_imagebind(audio_path):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n     = int(len(audio) * 16000 / sr)
        audio = scipy_signal.resample(audio, n).astype("float32")
    waveform      = torch.tensor(audio).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400,
        hop_length=160, n_mels=128, f_min=0.0, f_max=8000.0,
    )
    mel       = mel_transform(waveform)
    mel       = (mel + 1e-6).log()
    mel       = (mel + 4.268) / 4.569
    target    = 204
    t         = mel.shape[-1]
    if t < target:
        mel = torch.nn.functional.pad(mel, (0, target - t))
    else:
        mel = mel[:, :, :target]
    return mel.unsqueeze(0)

def get_imagebind_embeddings(text, image_path, audio_path):
    audio_tensor = load_audio_for_imagebind(audio_path).to(DEVICE)
    inputs = {
        ModalityType.TEXT:   ib_data.load_and_transform_text([text], DEVICE),
        ModalityType.VISION: ib_data.load_and_transform_vision_data([str(image_path)], DEVICE),
        ModalityType.AUDIO:  audio_tensor,
    }
    with torch.no_grad():
        embs = ib_model(inputs)
    t_emb = embs[ModalityType.TEXT][0].cpu().numpy()
    i_emb = embs[ModalityType.VISION][0].cpu().numpy()
    a_emb = embs[ModalityType.AUDIO][0].cpu().numpy()
    return t_emb, i_emb, a_emb

if IMAGEBIND_OK:
    if STAGE4_CSV.exists():
        print(f"  [RESUME] Found {STAGE4_CSV}, loading saved coherence scores...")
        df_ib = pd.read_csv(STAGE4_CSV)
    else:
        rows4 = []
        for i, row in df_merge.iterrows():
            pid  = int(row["prompt_id"])
            pmpt = row["prompt"]
            print(f"  [{i+1:03d}/{len(df_merge)}] {pmpt[:55]}...")
            try:
                t_emb, i_emb, a_emb = get_imagebind_embeddings(
                    row["generated_text"], row["image_path"], row["audio_path"]
                )
                s1   = cosine_sim(t_emb, i_emb)
                s2   = cosine_sim(t_emb, a_emb)
                s3   = cosine_sim(i_emb, a_emb)
                tsas = (s1 + s2 + s3) / 3
                rows4.append({"prompt_id": pid, "s1": round(s1, 4),
                              "s2": round(s2, 4), "s3": round(s3, 4),
                              "tsas": round(tsas, 4)})
                print(f"    s1={s1:.4f}  s2={s2:.4f}  s3={s3:.4f}  TSAS={tsas:.4f}")
            except Exception as e:
                print(f"    [ERROR] {e}")
                rows4.append({"prompt_id": pid, "s1": 0.0, "s2": 0.0,
                              "s3": 0.0, "tsas": 0.0})
        df_ib = pd.DataFrame(rows4)
        df_ib.to_csv(STAGE4_CSV, index=False)
        print(f"\n  Stage 4 saved to {STAGE4_CSV}")
    print(f"  TSAS avg = {df_ib['tsas'].mean():.4f}")
else:
    print("  [SKIP] ImageBind not available. Filling s1, s2, s3 with 0.")
    df_ib = pd.DataFrame({
        "prompt_id": df_merge["prompt_id"],
        "s1": 0.0, "s2": 0.0, "s3": 0.0, "tsas": 0.0,
    })


# ============================================================
#  STAGE 5 — WTSAS (Quality-Weighted Coherence)
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 5 — WTSAS")
print("=" * 60)
print(f"  Weights: w1={w1:.4f}  w2={w2:.4f}  w3={w3:.4f}")
print(f"  Lambda:  {OPT_LAMBDA:.2f}")

df_all = pd.merge(df_merge, df_ib, on="prompt_id")

final_rows = []
for _, row in df_all.iterrows():
    s1 = row["s1"]
    s2 = row["s2"]
    s3 = row["s3"]
    qt = row["q_text"]
    qi = row["q_image"]
    qa = row["q_audio"]

    s1_w  = s1 * (w1*qt + w2*qi) / (w1 + w2) if (w1 + w2) > 0 else 0
    s2_w  = s2 * (w1*qt + w3*qa) / (w1 + w3) if (w1 + w3) > 0 else 0
    s3_w  = s3 * (w2*qi + w3*qa) / (w2 + w3) if (w2 + w3) > 0 else 0

    wtsas    = (s1_w + s2_w + s3_w) / 3
    variance = float(np.var([s1_w, s2_w, s3_w]))
    final    = wtsas - OPT_LAMBDA * variance

    final_rows.append({
        "prompt_id":      int(row["prompt_id"]),
        "prompt":         row["prompt"],
        "generated_text": row["generated_text"],
        "image_path":     row["image_path"],
        "audio_path":     row["audio_path"],
        # Stage 1
        "q_text":         row["q_text"],
        "clip_score":     row["clip_score"],
        "aesthetic":      row["aesthetic"],
        "q_image":        row["q_image"],
        "semantic_score": row["semantic_score"],
        "wer_inv":        row["wer_inv"],
        "q_audio":        row["q_audio"],
        "transcript":     row["transcript"],
        # Stage 2
        "llm_overall":    row["llm_overall"],
        "llm_text":       row["llm_text"],
        "llm_image":      row["llm_image"],
        "llm_audio":      row["llm_audio"],
        "llm_reasoning":  row["llm_reasoning"],
        # Stage 3
        "w1":             round(w1, 4),
        "w2":             round(w2, 4),
        "w3":             round(w3, 4),
        "opt_lambda":     round(OPT_LAMBDA, 2),
        # Stage 4
        "s1_text_image":  round(s1, 4),
        "s2_text_audio":  round(s2, 4),
        "s3_image_audio": round(s3, 4),
        "tsas":           round(row["tsas"], 4),
        # Stage 5
        "s1_weighted":    round(s1_w, 4),
        "s2_weighted":    round(s2_w, 4),
        "s3_weighted":    round(s3_w, 4),
        "wtsas":          round(wtsas, 4),
        "wtsas_final":    round(final, 4),
    })

df_final = pd.DataFrame(final_rows)
df_final.to_csv(OUTPUT_CSV, index=False)


# ============================================================
#  SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Output saved to: {OUTPUT_CSV}")
print(f"  Total prompts:   {len(df_final)}")
print(f"\n  Average scores:")
print(f"    Q_text         = {df_final['q_text'].mean():.4f}")
print(f"    Q_image        = {df_final['q_image'].mean():.4f}")
print(f"    Q_audio        = {df_final['q_audio'].mean():.4f}")
print(f"    LLM overall    = {df_final['llm_overall'].mean():.2f}")
print(f"    TSAS           = {df_final['tsas'].mean():.4f}")
print(f"    WTSAS (final)  = {df_final['wtsas_final'].mean():.4f}")

if len(df_final) > 2:
    r_tsas,  _ = pearsonr(df_final["tsas"].values,        df_final["llm_overall"].values)
    r_wtsas, _ = pearsonr(df_final["wtsas_final"].values, df_final["llm_overall"].values)
    r_qimg,  _ = pearsonr(df_final["q_image"].values,     df_final["llm_overall"].values)
    print(f"\n  Correlations with LLM ratings:")
    print(f"    Q_image (CLIP) r = {r_qimg:.4f}")
    print(f"    TSAS           r = {r_tsas:.4f}")
    print(f"    WTSAS          r = {r_wtsas:.4f}")

print(f"\n  Learned weights:")
print(f"    w1 (Q_text)  = {w1:.4f}")
print(f"    w2 (Q_image) = {w2:.4f}")
print(f"    w3 (Q_audio) = {w3:.4f}")
print(f"    lambda       = {OPT_LAMBDA:.2f}")

print(f"\n  Next step for teammate:")
print(f"    python naive_bayes_score_v2.py  --input {OUTPUT_CSV}")
print(f"    python likelihood_ratio_score_v2.py --input {OUTPUT_CSV}")
print(f"\n  Done.")
