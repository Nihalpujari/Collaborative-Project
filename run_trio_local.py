"""Local launcher — loads api_keys.py, probes CLOUDFLARE_POOL for a live account, then runs hf_app.py."""
import os, sys, importlib.util, requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

spec = importlib.util.spec_from_file_location("_keys", PROJECT_ROOT / "api_keys.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Pick a Cloudflare account with quota
def _probe(account_id, api_token):
    try:
        url = (f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
               f"/ai/run/@cf/meta/llama-3.1-8b-instruct")
        r = requests.post(url,
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=10)
        return r.status_code != 429
    except Exception:
        return False

pool = getattr(mod, "CLOUDFLARE_POOL", [])
chosen_id    = getattr(mod, "CLOUDFLARE_ACCOUNT_ID", "")
chosen_token = getattr(mod, "CLOUDFLARE_API_TOKEN",  "")

if pool:
    print(f"Probing {len(pool)} Cloudflare accounts...")
    for entry in pool:
        aid, tok = entry["account_id"], entry["api_token"]
        ok = _probe(aid, tok)
        status = "OK" if ok else "quota/error"
        print(f"  {entry['label']}: {status}")
        if ok:
            chosen_id, chosen_token = aid, tok
            print(f"  -> using {entry['label']}")
            break
    else:
        print("  All accounts exhausted -- using fallback (may hit 429)")

os.environ["CLOUDFLARE_ACCOUNT_ID"] = chosen_id
os.environ["CLOUDFLARE_API_TOKEN"]  = chosen_token
os.environ.setdefault("GROQ_KEY",       getattr(mod, "GROQ_KEY",       "") or "")
os.environ.setdefault("GEMINI_API_KEY", getattr(mod, "GEMINI_API_KEY", "") or "")
os.environ.setdefault("CEREBRAS_KEY",   getattr(mod, "CEREBRAS_KEY",   "") or "")
os.environ.setdefault("MISTRAL_KEY",    getattr(mod, "MISTRAL_KEY",    "") or "")

# Remove project root from sys.path so secrets.py does not shadow stdlib secrets
project_str = str(PROJECT_ROOT)
sys.path = [p for p in sys.path if Path(p).resolve() != PROJECT_ROOT.resolve()]

print("Starting Trio -> http://127.0.0.1:7860")

app_path = PROJECT_ROOT / "hf_app.py"
exec(compile(app_path.read_text(), str(app_path), "exec"), {"__file__": str(app_path)})
