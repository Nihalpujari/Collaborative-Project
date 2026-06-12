"""
Test Cloudflare Workers AI - Single API key for all 3 tasks:
1. Text Generation  (@cf/meta/llama-3.1-8b-instruct)
2. Image Generation (@cf/black-forest-labs/flux-1-schnell)
3. Audio Generation (@cf/myshell-ai/melotts)
"""

import requests
import base64
from pathlib import Path

ACCOUNT_ID = "YOUR_CLOUDFLARE_ACCOUNT_ID"
API_TOKEN = "YOUR_CLOUDFLARE_API_TOKEN"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

PROMPT = "An elegant owl librarian wearing reading glasses, sitting in a grand library."

CF_TEST = Path(__file__).parent / "outputs/cf_test"
CF_TEST.mkdir(parents=True, exist_ok=True)


# TEXT
print("=" * 60)
print("TEST 1: TEXT GENERATION (@cf/meta/llama-3.1-8b-instruct)")
print("=" * 60)
try:
    r = requests.post(
        f"{BASE_URL}/@cf/meta/llama-3.1-8b-instruct",
        headers=HEADERS,
        json={"messages": [{"role": "user", "content": f"Describe this scene in detail: {PROMPT}"}], "max_tokens": 200},
        timeout=60
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        text = r.json().get("result", {}).get("response", "")
        print(f"PASS - Generated {len(text)} chars")
        print(f"Preview: {text[:200]}")
        with open(CF_TEST / "text_output.txt", "w", encoding="utf-8") as f:
            f.write(f"PROMPT: {PROMPT}\n\nGENERATED:\n{text}")
        print(f"Saved: {CF_TEST / 'text_output.txt'}")
    else:
        print(f"FAIL - {r.text[:200]}")
except Exception as e:
    print(f"FAIL - {e}")


# IMAGE
print()
print("=" * 60)
print("TEST 2: IMAGE GENERATION (@cf/black-forest-labs/flux-1-schnell)")
print("=" * 60)
try:
    r = requests.post(
        f"{BASE_URL}/@cf/black-forest-labs/flux-1-schnell",
        headers=HEADERS,
        json={"prompt": PROMPT},
        timeout=120
    )
    ct = r.headers.get("content-type", "")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        img_b64 = r.json().get("result", {}).get("image", "")
        if img_b64:
            img_data = base64.b64decode(img_b64)
            with open(CF_TEST / "image_output.png", "wb") as f:
                f.write(img_data)
            print(f"PASS - Image saved! {len(img_data)} bytes")
            print(f"Saved: {CF_TEST / 'image_output.png'}")
        else:
            print(f"FAIL - No image in response")
    else:
        print(f"FAIL - {r.text[:200]}")
except Exception as e:
    print(f"FAIL - {e}")


# AUDIO
print()
print("=" * 60)
print("TEST 3: AUDIO GENERATION (@cf/myshell-ai/melotts)")
print("=" * 60)
try:
    r = requests.post(
        f"{BASE_URL}/@cf/myshell-ai/melotts",
        headers=HEADERS,
        json={"prompt": PROMPT},
        timeout=60
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        audio_b64 = r.json().get("result", {}).get("audio", "")
        if audio_b64:
            audio_data = base64.b64decode(audio_b64)
            with open(CF_TEST / "audio_output.wav", "wb") as f:
                f.write(audio_data)
            print(f"PASS - Audio saved! {len(audio_data)} bytes")
            print(f"Saved: {CF_TEST / 'audio_output.wav'}")
        else:
            print(f"FAIL - No audio in response")
    else:
        print(f"FAIL - {r.text[:200]}")
except Exception as e:
    print(f"FAIL - {e}")


# SUMMARY
print()
print("=" * 60)
print("CONCLUSION: Cloudflare Workers AI - 1 API key, all 3 tasks!")
print("=" * 60)
