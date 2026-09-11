"""
STEP 1 — Generate Text + Image + Audio for all 500 prompts
==========================================================

Uses Cloudflare Workers AI:
  Text  : @cf/meta/llama-3.1-8b-instruct
  Image : @cf/black-forest-labs/flux-1-schnell
  Audio : @cf/myshell-ai/melotts

Saves to:
  outputs/text/prompt_N.txt
  outputs/images/prompt_N.png
  outputs/audio/prompt_N.wav

Resumes automatically — skips prompts already generated.
"""

import os
import time
import base64
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# -- config ------------------------------------------------------------------
HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN")

if not ACCOUNT_ID or ACCOUNT_ID == "your_account_id_here":
    raise ValueError("Set CLOUDFLARE_ACCOUNT_ID in .env file")
if not API_TOKEN or API_TOKEN == "your_api_token_here":
    raise ValueError("Set CLOUDFLARE_API_TOKEN in .env file")

BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"
HEADERS  = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

TEXT_DIR  = HERE / "outputs/text"
IMAGE_DIR = HERE / "outputs/images"
AUDIO_DIR = HERE / "outputs/audio"

PROMPTS_CSV = HERE / "prompts_500.csv"
SLEEP_BETWEEN = 1.0   # seconds between prompts (avoids rate limit)

# -- load prompts ------------------------------------------------------------
df = pd.read_csv(PROMPTS_CSV)
print(f"Loaded {len(df)} prompts")
print(f"Output dirs: {TEXT_DIR}\n")


def generate_text(prompt_id, prompt):
    txt_path = TEXT_DIR / f"prompt_{prompt_id}.txt"
    if txt_path.exists():
        return True, "SKIP"
    try:
        r = requests.post(
            f"{BASE_URL}/@cf/meta/llama-3.1-8b-instruct",
            headers=HEADERS,
            json={
                "messages": [{"role": "user", "content":
                    f"Describe this scene in vivid detail in 3-4 sentences: {prompt}"}],
                "max_tokens": 200,
            },
            timeout=60,
        )
        if r.status_code == 200:
            text = r.json().get("result", {}).get("response", "").strip()
            if text:
                txt_path.write_text(text, encoding="utf-8")
                return True, text[:60]
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:60]


def generate_image(prompt_id, prompt):
    img_path = IMAGE_DIR / f"prompt_{prompt_id}.png"
    if img_path.exists():
        return True, "SKIP"
    try:
        r = requests.post(
            f"{BASE_URL}/@cf/black-forest-labs/flux-1-schnell",
            headers=HEADERS,
            json={"prompt": prompt},
            timeout=120,
        )
        if r.status_code == 200:
            img_b64 = r.json().get("result", {}).get("image", "")
            if img_b64:
                img_path.write_bytes(base64.b64decode(img_b64))
                return True, f"{len(img_b64)} b64 chars"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:60]


def generate_audio(prompt_id, prompt):
    aud_path = AUDIO_DIR / f"prompt_{prompt_id}.wav"
    if aud_path.exists():
        return True, "SKIP"
    try:
        r = requests.post(
            f"{BASE_URL}/@cf/myshell-ai/melotts",
            headers=HEADERS,
            json={"prompt": prompt},
            timeout=60,
        )
        if r.status_code == 200:
            aud_b64 = r.json().get("result", {}).get("audio", "")
            if aud_b64:
                aud_path.write_bytes(base64.b64decode(aud_b64))
                return True, f"{len(aud_b64)} b64 chars"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:60]


# -- main loop ---------------------------------------------------------------
print("=" * 60)
print("  STEP 1 — Generating 500 Prompt Outputs")
print("=" * 60 + "\n")

success = 0
failed  = []
n       = len(df)

for i, row in df.iterrows():
    pid   = int(row["prompt_id"])
    pmpt  = str(row["prompt"])
    print(f"[{pid:03d}/{n}] {pmpt[:55]}...")

    ok_t, msg_t = generate_text(pid, pmpt)
    ok_i, msg_i = generate_image(pid, pmpt)
    ok_a, msg_a = generate_audio(pid, pmpt)

    status = "OK" if (ok_t and ok_i and ok_a) else "PARTIAL"
    print(f"        text={msg_t[:30]}  img={'OK' if ok_i else 'FAIL'}  audio={'OK' if ok_a else 'FAIL'}")

    if ok_t and ok_i and ok_a:
        success += 1
    else:
        failed.append(pid)

    if not (ok_t and ok_i and ok_a):
        time.sleep(2.0)   # longer wait on failure
    else:
        time.sleep(SLEEP_BETWEEN)

print("\n" + "=" * 60)
print(f"  Done.  Success: {success}/{n}   Failed: {len(failed)}")
if failed:
    print(f"  Failed prompt IDs: {failed[:20]}")
print(f"\n  Next step:")
print(f"    python step2_full_pipeline.py")
