"""
live_scores.py
==============
Live evaluation for a SINGLE generation, reusing the exact same scoring
methods as your team's cloudflare_benchmark.py — so the numbers match.

This lets frontend.py show scores live, right after generating, while the
batch benchmark (cloudflare_benchmark.py) still produces the full report.

Metrics (same as the benchmark):
    TEXT   : BERTScore, ROUGE-L, Readability
    IMAGE  : CLIP (ViT-L/14), Aesthetic (LAION MLP)
    (Audio's CLAP/MOS/WER are heavy and slow — left to the batch benchmark.)

Each function returns a dict with the score AND the intermediate detail,
so the frontend can show the "inside" of the calculation.
"""

import io
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# TEXT METRICS  (identical logic to cloudflare_benchmark.py)
# ---------------------------------------------------------------------------
def score_text(prompt, generated_text):
    out = {"bertscore": None, "rouge": None, "readability": None, "error": None}
    try:
        # BERTScore
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn([generated_text], [prompt], lang="en", verbose=False)
        out["bertscore"] = round(F1.item(), 4)
    except Exception as e:
        out["error"] = f"BERTScore: {e}"

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        s = scorer.score(prompt, generated_text)
        out["rouge"] = round(s["rougeL"].fmeasure, 4)
    except Exception as e:
        out["error"] = (out["error"] or "") + f" | ROUGE: {e}"

    try:
        from textstat import flesch_kincaid_grade
        raw = flesch_kincaid_grade(generated_text)
        out["readability"] = round(min(10.0, max(0.0, raw / 10.0)), 4)
    except Exception as e:
        out["error"] = (out["error"] or "") + f" | Readability: {e}"

    return out


# ---------------------------------------------------------------------------
# IMAGE METRICS  (identical logic to cloudflare_benchmark.py)
# ---------------------------------------------------------------------------
_clip_model = None
_clip_proc = None
_aes_model = None


def _load_image_models():
    global _clip_model, _clip_proc, _aes_model
    if _clip_model is not None:
        return
    import torch
    import torch.nn as nn
    from transformers import CLIPProcessor, CLIPModel
    import requests as req

    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    _clip_model.eval()

    class AestheticMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024), nn.Dropout(0.2),
                nn.Linear(1024, 128), nn.Dropout(0.2),
                nn.Linear(128, 64),   nn.Dropout(0.1),
                nn.Linear(64, 16),    nn.Linear(16, 1))
        def forward(self, x):
            return self.layers(x)

    # Always use the known-good LAION weights, downloaded to a fresh filename
    # (avoids clashing with an existing aesthetic_weights.pth of unknown architecture)
    weights_path = BASE_DIR / "laion_aesthetic_l14.pth"
    if not weights_path.exists():
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        weights_path.write_bytes(req.get(url, timeout=120).content)

    _aes_model = AestheticMLP()
    _aes_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    _aes_model.eval()


def score_image(prompt, image_bytes):
    out = {"clip": None, "aesthetic": None,
           "image_vector": None, "text_vector": None, "dim": None, "error": None}
    try:
        import torch
        from PIL import Image as PILImage
        _load_image_models()

        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = _clip_proc(text=[prompt], images=image, return_tensors="pt",
                            padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            img_out = _clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_out = _clip_model.get_text_features(input_ids=inputs["input_ids"])
            img_feat = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
            txt_feat = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
            img_feat = img_feat.float()
            txt_feat = txt_feat.float()
            img_n = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_n = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            clip_score = round(max(0.0, (img_n * txt_n).sum().item()), 4)
            aes_score = round(min(10.0, max(0.0, _aes_model(img_n).item())), 4)

        out["clip"] = clip_score
        out["aesthetic"] = aes_score
        out["image_vector"] = img_feat.squeeze(0).tolist()
        out["text_vector"] = txt_feat.squeeze(0).tolist()
        out["dim"] = img_feat.shape[-1]
    except Exception as e:
        out["error"] = str(e)
    return out
# ---------------------------------------------------------------------------
# AUDIO METRICS (CLAP, MOS, WER) — heavy, reads audio from a file path
# ---------------------------------------------------------------------------
def score_audio(prompt, audio_path):
    out = {"clap": None, "mos": None, "wer": None, "error": None}

    # CLAP — text <-> audio similarity
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
        import soundfile as sf
        from scipy import signal as _sig

        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        clap_proc  = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model.eval()

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 48000:
            audio = _sig.resample(audio, int(len(audio) * 48000 / sr))
        inp = clap_proc(text=[prompt], audio=[audio], return_tensors="pt",
                        padding=True, sampling_rate=48000)
        with torch.no_grad():
            af_out = clap_model.get_audio_features(input_features=inp["input_features"])
            tf_out = clap_model.get_text_features(
                input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        af = af_out.pooler_output if hasattr(af_out, "pooler_output") else af_out
        tf = tf_out.pooler_output if hasattr(tf_out, "pooler_output") else tf_out
        af = af.float() / af.float().norm(dim=-1, keepdim=True)
        tf = tf.float() / tf.float().norm(dim=-1, keepdim=True)
        out["clap"] = round(max(0.0, (af * tf).sum().item()), 4)
    except Exception as e:
        out["error"] = f"CLAP: {e}"

    # WER — transcribe and compare to prompt
    try:
        import whisper
        from jiwer import wer as jiwer_wer
        import soundfile as sf2
        from scipy import signal as _sig2
        wmodel = whisper.load_model("base")
        audio_w, sr_w = sf2.read(audio_path, dtype="float32")
        if audio_w.ndim > 1:
            audio_w = audio_w.mean(axis=1)
        if sr_w != 16000:
            audio_w = _sig2.resample(audio_w, int(len(audio_w) * 16000 / sr_w)).astype("float32")
        result = wmodel.transcribe(audio_w)
        hyp = result["text"].strip().lower()
        out["wer"] = round(min(1.0, max(0.0, jiwer_wer(prompt.lower(), hyp))), 4)
    except Exception as e:
        out["error"] = (out["error"] or "") + f" | WER: {e}"

    # MOS — predicted audio quality
    try:
        from speechmos import dnsmos
        import soundfile as sf3
        from scipy import signal as _sig3
        audio_m, sr_m = sf3.read(audio_path, dtype="float32")
        if audio_m.ndim > 1:
            audio_m = audio_m.mean(axis=1)
        if sr_m != 16000:
            audio_m = _sig3.resample(audio_m, int(len(audio_m) * 16000 / sr_m)).astype("float32")
        r = dnsmos.run(audio_m, 16000)
        row = r.iloc[0].to_dict() if hasattr(r, "iloc") else r
        mos_val = float(row.get("ovrl_mos", row.get("OVRL", row.get("mos", 0))))
        out["mos"] = round(min(5.0, max(1.0, mos_val)), 4)
    except Exception as e:
        out["error"] = (out["error"] or "") + f" | MOS: {e}"

    return out
