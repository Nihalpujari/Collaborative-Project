"""
Stage 2 - LLM Judge Ratings

Uses Gemini to rate each prompt's multimodal output
(text + image + audio transcript) with an overall quality score 1-10.

These ratings become the target (y) for Linear Regression in Stage 3
to learn weights w1, w2, w3 for Q_text, Q_image, Q_audio.

Saves: outputs/scores/llm_ratings.csv
"""

import sys
import os
import time
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# -- load API key ------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)
MODEL  = "gemini-3.5-flash-lite"

# -- paths -------------------------------------------------------------------
BASE       = Path(__file__).parent.parent
TEXT_DIR   = BASE / "outputs/text"
IMAGE_DIR  = BASE / "outputs/images"
SCORES_CSV = BASE / "outputs/scores/quality_scores.csv"
OUT_DIR    = BASE / "outputs/scores"

# -- rating prompt -----------------------------------------------------------
JUDGE_PROMPT = """You are an expert evaluator of AI-generated multimodal content.

You will be shown:
1. An original text prompt
2. AI-generated text based on that prompt
3. An AI-generated image based on that prompt
4. A transcript of AI-generated speech audio based on that prompt

Rate the OVERALL quality of all three outputs together on a scale of 1 to 10.

Scoring guide:
  9-10 = All three outputs are excellent and highly consistent with the prompt
  7-8  = Good quality, minor issues in one modality
  5-6  = Average, noticeable issues but still relevant
  3-4  = Poor, significant issues in one or more modalities
  1-2  = Very poor, outputs largely irrelevant to the prompt

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{"overall_score": <1-10>, "text_score": <1-10>, "image_score": <1-10>, "audio_score": <1-10>, "reasoning": "<one sentence>"}
"""


# -- helpers -----------------------------------------------------------------

def build_user_message(prompt, generated_text, transcript):
    return f"""Original Prompt: "{prompt}"

Generated Text:
{generated_text}

Audio Transcript (what the TTS narrated):
{transcript}

Please evaluate the image shown alongside this text and audio content."""


def parse_score(response_text):
    """Extract JSON scores from Gemini response."""
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Fallback: extract just overall_score
    match = re.search(r'overall_score["\s:]+(\d+(?:\.\d+)?)', response_text)
    if match:
        return {"overall_score": float(match.group(1)), "reasoning": "parsed fallback"}

    return None


def rate_prompt(idx, prompt, transcript):
    """Send one prompt's outputs to Gemini and get ratings."""
    text_path  = TEXT_DIR  / f"prompt_{idx}.txt"
    image_path = IMAGE_DIR / f"prompt_{idx}.png"

    if not image_path.exists():
        return None

    generated_text = text_path.read_text(encoding="utf-8").strip()
    user_message   = build_user_message(prompt, generated_text, transcript)
    full_prompt    = JUDGE_PROMPT + "\n\n" + user_message

    image = Image.open(image_path).convert("RGB")

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[full_prompt, image],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            result = parse_score(response.text)
            if result:
                return result
            else:
                print(f"    [WARN] Could not parse: {response.text[:120]}")
                return None
        except Exception as e:
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                wait = 10 * (attempt + 1)
                print(f"    [RETRY {attempt+1}/5] Server busy, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] {msg[:120]}")
                return None
    print(f"    [FAILED] Max retries reached")
    return None


# -- main --------------------------------------------------------------------

def main():
    if not SCORES_CSV.exists():
        print(f"ERROR: {SCORES_CSV} not found. Run quality_scores.py first.")
        return

    df_scores = pd.read_csv(SCORES_CSV)

    print("=" * 70)
    print(f"  STAGE 2 - LLM Judge Ratings ({MODEL})")
    print(f"  Rating {len(df_scores)} prompts...")
    print("=" * 70 + "\n")

    results = []

    for _, row in df_scores.iterrows():
        idx        = int(row["prompt_id"])
        prompt     = row["prompt"]
        transcript = row.get("transcript", "")

        print(f"  [{idx:02d}/{len(df_scores)}] {prompt[:55]}...")

        rating = rate_prompt(idx, prompt, transcript)

        if rating:
            overall = float(rating.get("overall_score", 0))
            text_s  = float(rating.get("text_score",    0))
            image_s = float(rating.get("image_score",   0))
            audio_s = float(rating.get("audio_score",   0))
            reason  = rating.get("reasoning", "")

            print(f"         Overall={overall}  Text={text_s}  Image={image_s}  Audio={audio_s}")
            print(f"         {reason[:80]}")

            results.append({
                "prompt_id":   idx,
                "prompt":      prompt[:80],
                "llm_overall": overall,
                "llm_text":    text_s,
                "llm_image":   image_s,
                "llm_audio":   audio_s,
                "reasoning":   reason,
            })
        else:
            print(f"         [FAILED - skipped]")

        time.sleep(1.0)

    df_out = pd.DataFrame(results)
    out_path = OUT_DIR / "llm_ratings.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Prompts rated : {len(results)}")
    for col, label in [("llm_overall","Overall"), ("llm_text","Text"), ("llm_image","Image"), ("llm_audio","Audio")]:
        print(f"  {label:<12} avg={df_out[col].mean():.2f}  min={df_out[col].min():.1f}  max={df_out[col].max():.1f}")

    print("\n  TOP 3:")
    for _, r in df_out.nlargest(3, "llm_overall").iterrows():
        print(f"    [{int(r['prompt_id'])}] {r['prompt'][:55]}  → {r['llm_overall']}/10")

    print("\n  BOTTOM 3:")
    for _, r in df_out.nsmallest(3, "llm_overall").iterrows():
        print(f"    [{int(r['prompt_id'])}] {r['prompt'][:55]}  → {r['llm_overall']}/10")


if __name__ == "__main__":
    main()
