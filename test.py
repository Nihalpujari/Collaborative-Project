"""
Standalone Audio Test - Works on Windows
"""

import requests
import sys
from pathlib import Path

# Find config.py in current directory
config_path = Path("config.py")
if not config_path.exists():
    print(f"❌ config.py not found!")
    print(f"   Looking in: {Path.cwd()}")
    sys.exit(1)

# Load config
config_vars = {}
exec(open(config_path).read(), config_vars)
AUDIO_MODELS = config_vars.get("AUDIO_MODELS", {})

print("\n" + "="*70)
print("AUDIO TEST - WINDOWS")
print("="*70 + "\n")

text = "The quick brown fox jumps over the lazy dog"
output_dir = Path("outputs/audio")
output_dir.mkdir(parents=True, exist_ok=True)

# TEST 1: ELEVENLABS
print("1. Testing ElevenLabs...")
try:
    cfg = AUDIO_MODELS.get("elevenlabs", {})
    key = cfg.get("api_key", "")
    print(f"   Key configured: {bool(key and not key.startswith('YOUR_'))}")
    
    if key and not key.startswith("YOUR_"):
        r = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB",
            headers={"xi-api-key": key},
            json={"text": text, "model_id": "eleven_multilingual_v2", 
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print("   ✅ SUCCESS\n")
        else:
            print(f"   ❌ FAILED: {r.text[:100]}\n")
    else:
        print("   ❌ No valid API key\n")
except Exception as e:
    print(f"   ❌ ERROR: {str(e)}\n")

# TEST 2: FISH AUDIO
print("2. Testing Fish Audio...")
try:
    cfg = AUDIO_MODELS.get("fish_audio", {})
    key = cfg.get("api_key", "")
    url = cfg.get("api_url", "")
    model = cfg.get("model_name", "")
    
    print(f"   Key configured: {bool(key and not key.startswith('YOUR_'))}")
    print(f"   URL: {url if url else 'NOT SET'}")
    print(f"   Model: {model if model else 'NOT SET'}")
    
    if key and not key.startswith("YOUR_") and url and model:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"text": text, "model": model, "voice": "default"},
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print("   ✅ SUCCESS\n")
        else:
            print(f"   ❌ FAILED: {r.text[:100]}\n")
    else:
        print("   ❌ Missing config fields\n")
except Exception as e:
    print(f"   ❌ ERROR: {str(e)}\n")

# TEST 3: CAMB AI
print("3. Testing Camb AI...")
try:
    cfg = AUDIO_MODELS.get("camb_ai", {})
    key = cfg.get("api_key", "")
    url = cfg.get("api_url", "")
    model = cfg.get("model_name", "")
    
    print(f"   Key configured: {bool(key and not key.startswith('YOUR_'))}")
    print(f"   URL: {url if url else 'NOT SET'}")
    print(f"   Model: {model if model else 'NOT SET'}")
    
    if key and not key.startswith("YOUR_") and url and model:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"text": text, "model": model, "voice": "default"},
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print("   ✅ SUCCESS\n")
        else:
            print(f"   ❌ FAILED: {r.text[:100]}\n")
    else:
        print("   ❌ Missing config fields\n")
except Exception as e:
    print(f"   ❌ ERROR: {str(e)}\n")

print("="*70 + "\n")