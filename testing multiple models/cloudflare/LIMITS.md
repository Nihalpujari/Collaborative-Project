# Cloudflare Workers AI — Quotas & Cost Limits

Owner: Anuj (Cloudflare API & Requirements)
Source: https://developers.cloudflare.com/workers-ai/platform/pricing/

---

## Free Tier

**10,000 Neurons per day, at no charge.** Resets daily (UTC). After that,
paid usage costs **$0.011 per 1,000 Neurons**.

"Neurons" are Cloudflare's compute-unit currency — every model call costs a
different number of neurons depending on input/output size.

---

## Per-model neuron cost (the 3 models we use)

| Model | Modality | Cost |
|---|---|---|
| `@cf/meta/llama-3.1-8b-instruct` | Text | 25,608 neurons / 1M input tokens · 75,147 neurons / 1M output tokens |
| `@cf/black-forest-labs/flux-1-schnell` | Image | 4.80 neurons / 512×512 tile + 9.60 neurons / diffusion step |
| `@cf/myshell-ai/melotts` | Audio (TTS) | 18.63 neurons / minute of generated audio |

---

## Estimated cost for a full 56-prompt benchmark run

These are **estimates** based on typical prompt/response sizes we observed
while testing (see below) — actual usage will vary a bit per prompt.

| Modality | Assumption | Neurons/prompt | × 56 prompts |
|---|---|---|---|
| Text | ~50 input tokens, ~200 output tokens | ~16 | ~900 |
| Image | 512×512, 4 steps (flux-schnell default) | ~43 | ~2,400 |
| Audio | ~1 min of generated speech per prompt | ~19 | ~1,050 |
| **Total** | | | **~4,350 neurons** |

**Conclusion: one full 56-prompt × 3-modality run costs ~4,350 neurons —
well inside the 10,000/day free tier.** You can run it roughly **twice a
day** for free before hitting the cap.

---

## Practical guidance for the team

1. **Don't loop the full benchmark repeatedly while debugging.** Every
   re-run burns quota. `cloudflare_benchmark.py` already skips
   already-generated files — use that caching, don't delete `outputs/`
   just to "start fresh" unless you actually changed the prompts or model.
2. **If you hit a `429` status code, that means the daily quota is
   exhausted.** The benchmark script already detects this
   (`QUOTA_EXHAUSTED` flag) and stops cleanly instead of erroring out. Just
   wait until the next UTC day.
3. **`melotts` (audio) is flaky, not quota-related.** It intermittently
   returns `500 Internal Server Error` on ~1 in 4 calls, regardless of
   prompt — this is a Cloudflare-side reliability issue, confirmed by
   testing the exact same request repeatedly. Fixed in
   `cloudflare_benchmark.py` with a 3-attempt retry + backoff — make sure
   you're running the updated version.
4. **Costs scale with output length**, not prompt count. A model asked to
   write long-form text will burn far more neurons than a short answer.
   Keep `max_tokens` capped in `generate_text` calls if we need to stretch
   quota further.

---

## Credentials

Stored in `config.py` (gitignored, never commit). Format:

```python
CLOUDFLARE = {
    "account_id": "...",
    "api_token":  "...",
}
```

Only Anuj currently holds the live token. If teammates need to run the
Cloudflare scripts locally, ask Anuj for the values rather than generating
your own token — this avoids splitting the shared 10,000/day quota across
multiple accounts and losing track of usage.
