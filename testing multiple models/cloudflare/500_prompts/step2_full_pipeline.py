"""
STEP 2 — Full Evaluation Pipeline (All Stages)
================================================

Reads: outputs/text/, outputs/images/, outputs/audio/
Runs:
  Stage 1 — Q_text, Q_image, Q_audio
  Stage 2 — LLM Judge (Gemini)
  Stage 3 — Linear Regression → w1, w2, w3
  Stage 4 — ImageBind coherence (s1, s2, s3)
  Stage 5 — WTSAS final score

Saves: scores/complete_scores.csv
"""

import os, re, sys, time, json, math
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

TEXT_DIR  = HERE / "outputs/text"
IMAGE_DIR = HERE / "outputs/images"
AUDIO_DIR = HERE / "outputs/audio"
SCORES    = HERE / "scores"
SCORES.mkdir(exist_ok=True)
CKPT      = HERE / "scores/checkpoints"
CKPT.mkdir(exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
LAMBDA     = 0.5

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")

# ---- load prompts ----
df_prompts = pd.read_csv(HERE / "prompts_500.csv")

# ---- models ----
print("Loading CLIP...")
from transformers import CLIPModel, CLIPProcessor
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("Loading Whisper...")
import whisper
whisper_model = whisper.load_model("base")

print("Loading Aesthetic...")
from transformers import pipeline as hf_pipeline
aesthetic_pipe = hf_pipeline("image-classification",
                              model="cafeai/cafe_aesthetic",
                              device=0 if DEVICE=="cuda" else -1)

print("Loading BERTScore...")
from bert_score import score as bert_score_fn

print("Loading ImageBind...")
import torchaudio, soundfile as sf
try:
    from imagebind import data as ib_data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    ib_model = imagebind_model.imagebind_huge(pretrained=True)
    ib_model.eval().to(DEVICE)
    IB_OK = True
    print("ImageBind loaded.")
except Exception as e:
    print(f"[WARN] ImageBind: {e} — skipping Stage 4")
    IB_OK = False

print("Loading Gemini...")
from google import genai
from google.genai import types
gem = genai.Client(api_key=GEMINI_KEY)
GEM_MODEL = "gemini-2.0-flash"

from jiwer import wer as compute_wer
print("\nAll models ready.\n")


# ============================================================
#  HELPERS
# ============================================================
def quality_pair(a, b):
    avg = (a + b) / 2
    var = float(np.var([a, b]))
    return float(np.clip(avg - LAMBDA * var, 0, 1))

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def resample(audio, orig_sr, tgt=16000):
    if orig_sr == tgt: return audio
    n = int(len(audio) * tgt / orig_sr)
    return scipy_signal.resample(audio, n).astype("float32")


# ============================================================
#  STAGE 1 — QUALITY SCORES
# ============================================================
S1_CSV = CKPT / "s1_quality.csv"

def get_q_text(text, prompt):
    _, _, F1 = bert_score_fn([text], [prompt], lang="en", verbose=False)
    return float(F1.mean())

def get_clip(prompt, img_path):
    img = Image.open(img_path).convert("RGB")
    t_in = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    i_in = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        t = clip_model.get_text_features(**t_in).squeeze().numpy()
        i = clip_model.get_image_features(**i_in).squeeze().numpy()
    return float(np.clip(cosine_sim(t, i), 0, 1))

def get_aesthetic(img_path):
    for r in aesthetic_pipe(str(img_path)):
        if r["label"] == "aesthetic": return float(r["score"])
    return 0.5

def transcribe(aud_path):
    try:
        audio, sr = sf.read(str(aud_path), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = resample(audio, sr)
        return whisper_model.transcribe(audio)["text"].strip()
    except: return ""

def get_q_audio(prompt, aud_path, gen_text):
    tr = transcribe(aud_path)
    if not tr: return 0.0, 0.0, 0.0, ""
    _, _, F1 = bert_score_fn([tr], [prompt], lang="en", verbose=False)
    sem = float(F1.mean())
    try:
        wi = 1 - float(np.clip(compute_wer(gen_text.lower(), tr.lower()), 0, 1))
    except: wi = 0.5
    return sem, wi, quality_pair(sem, wi), tr

print("=" * 60)
print("  STAGE 1 — Quality Scores")
print("=" * 60)

if S1_CSV.exists():
    print(f"  [RESUME] Loading saved quality scores...")
    df_q = pd.read_csv(S1_CSV)
else:
    rows = []
    n    = len(df_prompts)
    for _, row in df_prompts.iterrows():
        pid  = int(row["prompt_id"])
        pmpt = str(row["prompt"])
        tp   = TEXT_DIR  / f"prompt_{pid}.txt"
        ip   = IMAGE_DIR / f"prompt_{pid}.png"
        ap   = AUDIO_DIR / f"prompt_{pid}.wav"

        if not tp.exists() or not ip.exists() or not ap.exists():
            print(f"  [{pid:03d}] SKIP — output files missing")
            continue

        gtext = tp.read_text(encoding="utf-8").strip()
        print(f"  [{pid:03d}/{n}] {pmpt[:50]}...")

        try: qt = get_q_text(gtext, pmpt)
        except: qt = 0.5

        try:
            cs = get_clip(pmpt, ip)
            ae = get_aesthetic(ip)
            qi = quality_pair(cs, ae)
        except: cs, ae, qi = 0.5, 0.5, 0.5

        try: sem, wi, qa, tr = get_q_audio(pmpt, ap, gtext)
        except: sem, wi, qa, tr = 0.5, 0.5, 0.5, ""

        rows.append({"prompt_id": pid, "prompt": pmpt, "generated_text": gtext,
                     "q_text": round(qt,4), "clip_score": round(cs,4),
                     "aesthetic": round(ae,4), "q_image": round(qi,4),
                     "semantic_score": round(sem,4), "wer_inv": round(wi,4),
                     "q_audio": round(qa,4), "transcript": tr})
        print(f"    Q_text={qt:.3f}  Q_image={qi:.3f}  Q_audio={qa:.3f}")

    df_q = pd.DataFrame(rows)
    df_q.to_csv(S1_CSV, index=False)

print(f"  Stage 1 done: {len(df_q)} prompts")
print(f"  Q_text={df_q['q_text'].mean():.4f}  Q_image={df_q['q_image'].mean():.4f}  Q_audio={df_q['q_audio'].mean():.4f}")


# ============================================================
#  STAGE 2 — LLM JUDGE
# ============================================================
S2_CSV = CKPT / "s2_llm_ratings.csv"

JUDGE = """You are an expert evaluator of AI-generated multimodal content.

You will see: (1) original prompt, (2) AI-generated text, (3) AI-generated image, (4) audio transcript.

Rate OVERALL quality 1-10. Also rate each modality.
9-10=Excellent  7-8=Good  5-6=Average  3-4=Poor  1-2=Very Poor

Respond ONLY with valid JSON:
{"overall_score":<1-10>,"text_score":<1-10>,"image_score":<1-10>,"audio_score":<1-10>,"reasoning":"<one sentence>"}"""

def parse_rating(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m: return json.loads(m.group())
    except: pass
    m = re.search(r'overall_score["\s:]+(\d+(?:\.\d+)?)', text)
    if m: return {"overall_score": float(m.group(1)), "text_score":5,
                  "image_score":5, "audio_score":5, "reasoning":"fallback"}
    return None

def call_gemini(row):
    msg  = (f'{JUDGE}\n\nPrompt: "{row["prompt"]}"\n\n'
            f'Text:\n{row["generated_text"]}\n\nTranscript:\n{row["transcript"]}')
    img  = Image.open(IMAGE_DIR / f"prompt_{int(row['prompt_id'])}.png").convert("RGB")
    for attempt in range(5):
        try:
            resp = gem.models.generate_content(
                model=GEM_MODEL, contents=[msg, img],
                config=types.GenerateContentConfig(
                    temperature=0.1, max_output_tokens=2048,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
            r = parse_rating(resp.text)
            if r: return r
        except Exception as e:
            if any(x in str(e) for x in ["503","429","UNAVAILABLE"]):
                wait = 10*(attempt+1)
                print(f"    [RETRY {attempt+1}] wait {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [ERR] {str(e)[:80]}")
                return None
    return None

print("\n" + "=" * 60)
print("  STAGE 2 — LLM Judge")
print("=" * 60)

if S2_CSV.exists():
    print(f"  [RESUME] Loading saved ratings...")
    df_r = pd.read_csv(S2_CSV)
    done = set(df_r["prompt_id"].tolist())
else:
    df_r, done = pd.DataFrame(), set()

rows2 = df_r.to_dict("records") if not df_r.empty else []

for i, row in df_q.iterrows():
    pid = int(row["prompt_id"])
    if pid in done: continue
    print(f"  [{pid:03d}] Judging...")
    r = call_gemini(row)
    rows2.append({"prompt_id": pid,
                  "llm_overall": float(r.get("overall_score",5)) if r else 5.0,
                  "llm_text":   float(r.get("text_score",5))    if r else 5.0,
                  "llm_image":  float(r.get("image_score",5))   if r else 5.0,
                  "llm_audio":  float(r.get("audio_score",5))   if r else 5.0,
                  "llm_reasoning": str(r.get("reasoning","")) if r else "FAILED"})
    if r: print(f"    overall={r['overall_score']}  text={r['text_score']}  image={r['image_score']}  audio={r['audio_score']}")
    if (i+1) % 10 == 0:
        pd.DataFrame(rows2).to_csv(S2_CSV, index=False)
        print(f"    [CHECKPOINT] saved {len(rows2)} ratings")
    time.sleep(1.0)

df_r = pd.DataFrame(rows2)
df_r.to_csv(S2_CSV, index=False)
print(f"  Stage 2 done: {len(df_r)} ratings, avg={df_r['llm_overall'].mean():.2f}")


# ============================================================
#  STAGE 3 — LEARN WEIGHTS
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 3 — Learn Weights (Linear Regression)")
print("=" * 60)

df_m = pd.merge(df_q, df_r, on="prompt_id")
X_raw = df_m[["q_text","q_image","q_audio"]].values
y     = df_m["llm_overall"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw)
lr     = LinearRegression().fit(X_sc, y)

raw_w = np.maximum(lr.coef_, 0)
if raw_w.sum() < 1e-8: raw_w = np.array([1/3,1/3,1/3])
w1, w2, w3 = raw_w / raw_w.sum()

# lambda sweep
best_lam, best_r = LAMBDA, -99
for lam in np.arange(0.0, 2.05, 0.1):
    sc = [(w1*r["q_text"]+w2*r["q_image"]+w3*r["q_audio"])
          - lam*float(np.var([r["q_text"],r["q_image"],r["q_audio"]]))
          for _,r in df_m.iterrows()]
    rv, _ = pearsonr(sc, y)
    if rv > best_r: best_r, best_lam = rv, lam
OPT_LAM = best_lam

print(f"  w1(text)={w1:.4f}  w2(image)={w2:.4f}  w3(audio)={w3:.4f}")
print(f"  R²={lr.score(X_sc,y):.4f}  optimal_lambda={OPT_LAM:.2f}")

for feat, rval in zip(["q_text","q_image","q_audio"],
                      [pearsonr(df_m[f].values,y)[0] for f in ["q_text","q_image","q_audio"]]):
    print(f"  Pearson r({feat}) = {rval:.4f}")


# ============================================================
#  STAGE 4 — IMAGEBIND COHERENCE
# ============================================================
S4_CSV = CKPT / "s4_imagebind.csv"

def load_audio_ib(aud_path):
    audio, sr = sf.read(str(aud_path), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = scipy_signal.resample(audio, int(len(audio)*16000/sr)).astype("float32")
    w  = torch.tensor(audio).unsqueeze(0)
    mt = torchaudio.transforms.MelSpectrogram(
         sample_rate=16000,n_fft=400,win_length=400,
         hop_length=160,n_mels=128,f_min=0.,f_max=8000.)
    mel = mt(w)
    mel = (mel+1e-6).log()
    mel = (mel+4.268)/4.569
    t   = mel.shape[-1]
    mel = torch.nn.functional.pad(mel,(0,max(0,204-t)))[:,:,:204]
    return mel.unsqueeze(0)

print("\n" + "=" * 60)
print("  STAGE 4 — ImageBind Coherence")
print("=" * 60)

if S4_CSV.exists():
    print(f"  [RESUME] Loading saved coherence scores...")
    df_ib = pd.read_csv(S4_CSV)
elif IB_OK:
    rows4 = []
    for _, row in df_m.iterrows():
        pid  = int(row["prompt_id"])
        ip   = IMAGE_DIR / f"prompt_{pid}.png"
        ap   = AUDIO_DIR / f"prompt_{pid}.wav"
        print(f"  [{pid:03d}] ImageBind...")
        try:
            at  = load_audio_ib(ap).to(DEVICE)
            inp = {
                ModalityType.TEXT:   ib_data.load_and_transform_text([row["generated_text"]], DEVICE),
                ModalityType.VISION: ib_data.load_and_transform_vision_data([str(ip)], DEVICE),
                ModalityType.AUDIO:  at,
            }
            with torch.no_grad(): embs = ib_model(inp)
            t_e = embs[ModalityType.TEXT][0].cpu().numpy()
            i_e = embs[ModalityType.VISION][0].cpu().numpy()
            a_e = embs[ModalityType.AUDIO][0].cpu().numpy()
            s1, s2, s3 = cosine_sim(t_e,i_e), cosine_sim(t_e,a_e), cosine_sim(i_e,a_e)
            tsas = (s1+s2+s3)/3
            rows4.append({"prompt_id":pid,"s1":round(s1,4),"s2":round(s2,4),
                          "s3":round(s3,4),"tsas":round(tsas,4)})
            print(f"    s1={s1:.4f}  s2={s2:.4f}  s3={s3:.4f}  TSAS={tsas:.4f}")
        except Exception as e:
            print(f"    [ERR] {e}")
            rows4.append({"prompt_id":pid,"s1":0.,"s2":0.,"s3":0.,"tsas":0.})
    df_ib = pd.DataFrame(rows4)
    df_ib.to_csv(S4_CSV, index=False)
else:
    print("  [SKIP] ImageBind not available — filling with 0")
    df_ib = pd.DataFrame({"prompt_id":df_m["prompt_id"],"s1":0.,"s2":0.,"s3":0.,"tsas":0.})

print(f"  TSAS avg={df_ib['tsas'].mean():.4f}")


# ============================================================
#  STAGE 5 — WTSAS
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 5 — WTSAS")
print("=" * 60)

df_all = pd.merge(df_m, df_ib, on="prompt_id")
final  = []

for _, row in df_all.iterrows():
    s1,s2,s3 = row["s1"],row["s2"],row["s3"]
    qt,qi,qa = row["q_text"],row["q_image"],row["q_audio"]
    s1w = s1*(w1*qt+w2*qi)/(w1+w2) if (w1+w2)>0 else 0
    s2w = s2*(w1*qt+w3*qa)/(w1+w3) if (w1+w3)>0 else 0
    s3w = s3*(w2*qi+w3*qa)/(w2+w3) if (w2+w3)>0 else 0
    wtsas = (s1w+s2w+s3w)/3
    var   = float(np.var([s1w,s2w,s3w]))
    wf    = wtsas - OPT_LAM*var
    final.append({
        "prompt_id":row["prompt_id"],"prompt":row["prompt"],
        "generated_text":row["generated_text"],
        "q_text":row["q_text"],"clip_score":row["clip_score"],
        "aesthetic":row["aesthetic"],"q_image":row["q_image"],
        "semantic_score":row["semantic_score"],"wer_inv":row["wer_inv"],
        "q_audio":row["q_audio"],"transcript":row["transcript"],
        "llm_overall":row["llm_overall"],"llm_text":row["llm_text"],
        "llm_image":row["llm_image"],"llm_audio":row["llm_audio"],
        "llm_reasoning":row["llm_reasoning"],
        "w1":round(w1,4),"w2":round(w2,4),"w3":round(w3,4),"opt_lambda":round(OPT_LAM,2),
        "s1_text_image":round(s1,4),"s2_text_audio":round(s2,4),"s3_image_audio":round(s3,4),
        "tsas":round(row["tsas"],4),
        "s1_weighted":round(s1w,4),"s2_weighted":round(s2w,4),"s3_weighted":round(s3w,4),
        "wtsas":round(wtsas,4),"wtsas_final":round(wf,4),
    })

df_final = pd.DataFrame(final)
OUT_CSV  = SCORES / "complete_scores.csv"
df_final.to_csv(OUT_CSV, index=False)

print(f"  Saved {len(df_final)} rows to {OUT_CSV}")
r_tsas,  _ = pearsonr(df_final["tsas"].values,        df_final["llm_overall"].values)
r_wtsas, _ = pearsonr(df_final["wtsas_final"].values, df_final["llm_overall"].values)
r_qimg,  _ = pearsonr(df_final["q_image"].values,     df_final["llm_overall"].values)
print(f"\n  Correlations with LLM ratings:")
print(f"    Q_image r = {r_qimg:.4f}")
print(f"    TSAS    r = {r_tsas:.4f}")
print(f"    WTSAS   r = {r_wtsas:.4f}")

print(f"\n  Next step:")
print(f"    python step3_score_naive_and_lr.py")
