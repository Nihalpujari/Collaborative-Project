"""
INTEGRATED PIPELINE: Text Generation → Audio Conversion
5 Prompts × 3 Text Models × 2 Audio Models
With text cleaning + retry logic for quota limits
"""

import requests, base64, time, re
from pathlib import Path

# ── Load config + prompts ──────────────────────────────────────────────────
config_vars = {}
exec(open("config.py").read(), config_vars)
TEXT_MODELS  = config_vars.get("TEXT_MODELS", {})
AUDIO_MODELS = config_vars.get("AUDIO_MODELS", {})

from prompts import get_prompts
prompts = get_prompts()[:5]

for model in ["mistral", "groq", "cerebras"]:
    for audio in ["gemini_tts", "deepgram"]:
        Path(f"outputs/pipeline/{model}/{audio}").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/pipeline/{model}/text").mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("INTEGRATED PIPELINE — 5 Prompts × 3 Text × 2 Audio")
print("="*70 + "\n")

# ── Text cleaner ───────────────────────────────────────────────────────────
def clean_for_tts(text):
    if not text: return ""
    text = re.sub(r'\|[^\n]*\|', '', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    if len(text) > 500:
        cut = text[:500]
        last_end = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
        text = cut[:last_end+1] if last_end > 100 else cut
    return text.strip()

# ── Text generation ────────────────────────────────────────────────────────
def generate_text(model_key, prompt):
    try:
        cfg   = TEXT_MODELS.get(model_key, {})
        key   = cfg.get("api_key", "")
        url   = cfg.get("api_url", "")
        model = cfg.get("model_name", "")
        if not key or key.startswith("YOUR_"): return None
        r = requests.post(url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300},
            timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except: pass
    return None

# ── Gemini TTS with retry ──────────────────────────────────────────────────
GEMINI_DELAY = 5  # seconds between calls to stay under quota

def speak_gemini(text, out_path, retries=3):
    cfg = AUDIO_MODELS.get("gemini_tts", {})
    key = cfg.get("api_key", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={key}"

    for attempt in range(retries):
        try:
            r = requests.post(url, json={
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}}
                }
            }, timeout=60)

            if r.status_code == 200:
                audio_bytes = base64.b64decode(
                    r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
                Path(out_path).write_bytes(audio_bytes)
                return len(audio_bytes) // 1024

            elif r.status_code == 429:
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f"\n      ⏳ Quota hit — waiting {wait}s before retry {attempt+1}/{retries}...", end="", flush=True)
                time.sleep(wait)
                continue

            else:
                return f"Error {r.status_code}: {r.text[:60]}"

        except Exception as e:
            return f"Exception: {str(e)[:60]}"

    return "Failed after retries (quota exhausted)"

# ── Deepgram ───────────────────────────────────────────────────────────────
def speak_deepgram(text, out_path):
    try:
        cfg = AUDIO_MODELS.get("deepgram", {})
        key = cfg.get("api_key", "")
        r = requests.post(
            "https://api.deepgram.com/v1/speak?model=aura-asteria-en",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            json={"text": text}, timeout=30)
        if r.status_code == 200:
            Path(out_path).write_bytes(r.content)
            return len(r.content) // 1024
        return f"Error {r.status_code}: {r.text[:60]}"
    except Exception as e:
        return f"Exception: {str(e)[:60]}"

# ── Main loop ──────────────────────────────────────────────────────────────
results = {}

for idx, prompt_text in enumerate(prompts):
    pid = idx + 1
    print(f"{'='*70}")
    print(f"PROMPT {pid}: {prompt_text[:60]}...")
    print(f"{'='*70}\n")

    for text_model in ["mistral", "groq", "cerebras"]:
        print(f"  ── {text_model.upper()} ──")

        # Step 1: Generate text
        print(f"    Step 1 — Generating text...", end="", flush=True)
        generated = generate_text(text_model, prompt_text)

        if generated:
            print(f" ✓ ({len(generated)} chars)")
            cleaned = clean_for_tts(generated)
            print(f"    Cleaned: {len(cleaned)} chars → \"{cleaned[:65]}...\"")
            Path(f"outputs/pipeline/{text_model}/text/prompt_{pid}.txt").write_text(generated, encoding="utf-8")

            # Step 2a: Gemini TTS (with delay between calls)
            time.sleep(GEMINI_DELAY)
            print(f"    Step 2a — Gemini TTS...", end="", flush=True)
            result = speak_gemini(cleaned, f"outputs/pipeline/{text_model}/gemini_tts/prompt_{pid}.wav")
            if isinstance(result, int):
                print(f" ✓ ({result} KB)")
                results[f"{text_model}_gemini"] = results.get(f"{text_model}_gemini", 0) + 1
            else:
                print(f" ✗  {result}")

            # Step 2b: Deepgram
            time.sleep(1)
            print(f"    Step 2b — Deepgram...", end="", flush=True)
            result = speak_deepgram(cleaned, f"outputs/pipeline/{text_model}/deepgram/prompt_{pid}.mp3")
            if isinstance(result, int):
                print(f" ✓ ({result} KB)")
                results[f"{text_model}_deepgram"] = results.get(f"{text_model}_deepgram", 0) + 1
            else:
                print(f" ✗  {result}")

        else:
            print(f" ✗ (generation failed — retrying once...)")
            time.sleep(3)
            generated = generate_text(text_model, prompt_text)
            if generated:
                print(f"    Retry ✓ ({len(generated)} chars)")
            else:
                print(f"    Retry ✗ — skipping audio for this prompt")

        print()

# ── Summary ────────────────────────────────────────────────────────────────
print("="*70)
print("PIPELINE SUMMARY")
print("="*70)
print(f"\n  {'Combination':<26} {'Success':>8}  Status")
print(f"  {'-'*50}")
for text_m in ["mistral", "groq", "cerebras"]:
    for audio_m in ["gemini_tts", "deepgram"]:
        key = f"{text_m}_{audio_m}"
        ok  = results.get(key, 0)
        icon = "✅" if ok == 5 else "⚠️ " if ok > 0 else "❌"
        print(f"  {text_m} → {audio_m:<16} {ok}/5      {icon}")

total = sum(results.values())
print(f"\n  Total: {total}/30")
print("="*70 + "\n")