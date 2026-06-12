"""
============================================================
  MULTIMODAL AI CONTENT GENERATOR  ·  multi-agent pipeline
============================================================

ONE user prompt  →  three coordinated outputs (text + image + audio).

Architecture (mirrors the LangGraph design in Project 2):

    USER PROMPT
        │
        ▼
    ┌─────────────────────────────────────┐
    │  MAIN AGENT (Supervisor / Router)   │
    │                                     │
    │   1.  PromptEnhancer Sub-Agent      │ ← uses LLM to derive 3 modality prompts
    │            ┌────┬────┐              │
    │            ▼    ▼    ▼              │
    │         text  image audio           │
    │        prompt prompt prompt         │
    │            │    │    │              │
    │   2. ┌─────┴─┐  │  ┌─┴─────┐        │
    │      ▼       ▼  ▼  ▼       ▼        │
    │     TEXT   IMAGE     AUDIO          │ ← parallel Sub-Agents
    │     Kimi   FLUX      Kokoro         │
    │      │      │         │             │
    │      ▼      ▼         ▼             │
    │   .txt    .png      .flac           │
    │      │      │         │             │
    │      └──────┴────┬────┘             │
    │                  ▼                  │
    │           BUNDLE & RESULTS          │
    └─────────────────────────────────────┘

All three sub-agents run under ONE Hugging Face token via Inference Providers.

SETUP:
    pip install huggingface_hub openai Pillow

    export HF_TOKEN="hf_YOUR_TOKEN"

RUN (two ways):

    # Interactive — it asks for your prompt
    python3 multimodal_pipeline.py

    # Non-interactive — pass your prompt as an argument
    python3 multimodal_pipeline.py "Launch campaign for a sustainable coffee brand"

OUTPUTS (timestamped folder, e.g. ./runs/2026-06-10_14-32-05/):
    01_enhanced_prompts.json   ·  the 3 derived prompts (provenance)
    02_text.txt                ·  generated copy
    03_image.png               ·  generated image
    04_audio.flac              ·  generated narration
    99_results.json            ·  full benchmark: latency, model, scores
"""

import os, sys, time, json, pathlib, datetime, concurrent.futures as cf

# ----------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Set HF_TOKEN env var first (https://huggingface.co/settings/tokens)"

# Model choices — all open source, all routed via the same HF token.
MODEL_TEXT  = "moonshotai/Kimi-K2-Instruct-0905"
MODEL_IMAGE = "black-forest-labs/FLUX.1-schnell"
MODEL_AUDIO = "hexgrad/Kokoro-82M"

# Get user prompt — CLI arg, or interactive
if len(sys.argv) > 1:
    USER_PROMPT = " ".join(sys.argv[1:])
else:
    print("Multimodal Content Generator — open-source pipeline")
    print("-" * 50)
    USER_PROMPT = input("Enter your prompt (one line, plain English):\n> ").strip()
    if not USER_PROMPT:
        USER_PROMPT = "Launch campaign for a sustainable coffee brand"
        print(f"(Empty — using default: {USER_PROMPT})")

# Output folder (timestamped so successive runs don't overwrite)
run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = pathlib.Path("runs") / run_id
OUT.mkdir(parents=True, exist_ok=True)
print(f"\nRun folder: {OUT}/\n")

results = {"user_prompt": USER_PROMPT, "run_id": run_id,
           "models": {"text": MODEL_TEXT, "image": MODEL_IMAGE, "audio": MODEL_AUDIO}}

# ----------------------------------------------------------
# 1. PROMPT ENHANCER SUB-AGENT
#    Uses the LLM itself to derive three modality-specific prompts
#    from the single user prompt.
# ----------------------------------------------------------
def prompt_enhancer(user_prompt: str) -> dict:
    """Return a dict with three keys: text_prompt, image_prompt, audio_prompt."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    system = (
        "You are a Prompt Enhancer for a multimodal AI content generator. "
        "Given a single user idea, you return three specialised prompts in JSON, "
        "one optimised for each modality.\n\n"
        "Rules:\n"
        "1. 'text_prompt': instructions for an LLM to write ~100 words of marketing "
        "copy. Include tone, structure, audience, CTA.\n"
        "2. 'image_prompt': visual description for a text-to-image model (FLUX). "
        "Concrete subject, composition, style, lighting, camera. No abstract words.\n"
        "3. 'audio_prompt': the SHORT narration script for a TTS model (~40-60 words, "
        "speakable, no markdown). This is what will be read aloud verbatim.\n\n"
        "Return ONLY a JSON object with exactly these three keys."
    )
    print(f"[1/4] PROMPT ENHANCER  ·  {MODEL_TEXT}")
    t0 = time.time()
    try:
        completion = client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        raw = completion.choices[0].message.content
        enhanced = json.loads(raw)
        dt = time.time() - t0
        # Validate keys
        for k in ("text_prompt", "image_prompt", "audio_prompt"):
            assert k in enhanced, f"Missing key {k} in enhancer output"
        (OUT / "01_enhanced_prompts.json").write_text(json.dumps(enhanced, indent=2))
        print(f"      OK  ·  {dt:.2f}s  →  01_enhanced_prompts.json")
        results["prompt_enhancer"] = {"ok": True, "latency_s": round(dt, 2),
                                      "model": MODEL_TEXT}
        return enhanced
    except Exception as e:
        # Robust fallback: just reuse the user prompt for each modality
        print(f"      FAIL  ·  {type(e).__name__}: {e}  (using fallback)")
        results["prompt_enhancer"] = {"ok": False, "error": str(e), "used_fallback": True}
        return {
            "text_prompt":  f"Write a 100-word marketing paragraph about: {user_prompt}. "
                            "Professional tone, end with a CTA.",
            "image_prompt": f"{user_prompt}, cinematic, modern, glassmorphism, 4k",
            "audio_prompt": f"Introducing our new offering — {user_prompt}. "
                            "Discover what makes it different.",
        }

# ----------------------------------------------------------
# 2. SUB-AGENTS  (each runs independently — parallelised below)
# ----------------------------------------------------------
def text_agent(prompt: str) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
    print(f"[2/4] TEXT SUB-AGENT   ·  {MODEL_TEXT}")
    t0 = time.time()
    completion = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = completion.choices[0].message.content
    dt = time.time() - t0
    (OUT / "02_text.txt").write_text(text)
    print(f"      OK  ·  {dt:.2f}s  ·  {len(text.split())} words  →  02_text.txt")
    return {"ok": True, "latency_s": round(dt, 2), "words": len(text.split()),
            "file": "02_text.txt", "model": MODEL_TEXT}

def image_agent(prompt: str) -> dict:
    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=HF_TOKEN)
    print(f"[3/4] IMAGE SUB-AGENT  ·  {MODEL_IMAGE}")
    t0 = time.time()
    image = client.text_to_image(prompt, model=MODEL_IMAGE)
    dt = time.time() - t0
    image.save(OUT / "03_image.png")
    size = (OUT / "03_image.png").stat().st_size
    print(f"      OK  ·  {dt:.2f}s  ·  {size:,} bytes  →  03_image.png")
    return {"ok": True, "latency_s": round(dt, 2), "bytes": size,
            "file": "03_image.png", "model": MODEL_IMAGE}

def audio_agent(prompt: str) -> dict:
    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=HF_TOKEN)
    print(f"[4/4] AUDIO SUB-AGENT  ·  {MODEL_AUDIO}")
    t0 = time.time()
    audio = client.text_to_speech(prompt, model=MODEL_AUDIO)
    dt = time.time() - t0
    (OUT / "04_audio.flac").write_bytes(audio)
    size = (OUT / "04_audio.flac").stat().st_size
    print(f"      OK  ·  {dt:.2f}s  ·  {size:,} bytes  →  04_audio.flac")
    return {"ok": True, "latency_s": round(dt, 2), "bytes": size,
            "file": "04_audio.flac", "model": MODEL_AUDIO}

# ----------------------------------------------------------
# 3. MAIN AGENT (Supervisor) orchestrates the run
# ----------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 60)
    print(f"  MAIN AGENT  ·  user prompt: {USER_PROMPT}")
    print("=" * 60)

    # --- Step 1: enhance -----------------------------------
    enhanced = prompt_enhancer(USER_PROMPT)

    # --- Step 2: run sub-agents in parallel ----------------
    print()
    sub_results = {"text": None, "image": None, "audio": None}
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(text_agent,  enhanced["text_prompt"]):  "text",
            ex.submit(image_agent, enhanced["image_prompt"]): "image",
            ex.submit(audio_agent, enhanced["audio_prompt"]): "audio",
        }
        for fut in cf.as_completed(futures):
            modality = futures[fut]
            try:
                sub_results[modality] = fut.result()
            except Exception as e:
                print(f"      FAIL  ·  {modality}: {type(e).__name__}: {e}")
                sub_results[modality] = {"ok": False, "error": str(e)}
    results["sub_agents"] = sub_results

    # --- Step 3: bundle & report ---------------------------
    results["total_latency_s"] = round(time.time() - t_start, 2)
    (OUT / "99_results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print(f"  DONE in {results['total_latency_s']}s")
    print(f"  Files in {OUT}/")
    for p in sorted(OUT.iterdir()):
        print(f"    · {p.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
