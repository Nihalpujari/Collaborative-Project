"""Local launcher — loads api_keys.py then runs hf_app.py."""
import os, sys, importlib.util
from pathlib import Path

# Project root (where api_keys.py lives)
PROJECT_ROOT = Path(r"D:\nihal\Collaborative-Project")

for name in ("api_keys.py", "secrets.py"):
    keys_path = PROJECT_ROOT / name
    if keys_path.exists():
        break

spec = importlib.util.spec_from_file_location("_keys", keys_path)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

os.environ.setdefault("CLOUDFLARE_ACCOUNT_ID", getattr(mod, "CLOUDFLARE_ACCOUNT_ID", "") or "")
os.environ.setdefault("CLOUDFLARE_API_TOKEN",  getattr(mod, "CLOUDFLARE_API_TOKEN",  "") or "")
os.environ.setdefault("GROQ_KEY",              getattr(mod, "GROQ_KEY",              "") or "")
os.environ.setdefault("GEMINI_API_KEY",        getattr(mod, "GEMINI_API_KEY",        "") or "")

print(f"Credentials loaded from {keys_path.name}")
print("Starting Trio -> http://127.0.0.1:7860")

app_path = Path(__file__).parent / "hf_app.py"
exec(compile(app_path.read_text(), str(app_path), "exec"), {"__file__": str(app_path)})
