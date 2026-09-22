"""
Trio — multimodal AI generator for Hugging Face Spaces
Reads credentials from HF Secrets (Settings → Variables and secrets).
"""

import os, base64, time, io, math, tempfile, struct, re, json, requests, secrets as _tok
from collections import OrderedDict
from pathlib import Path

import gradio as gr
from PIL import Image

# ── Credentials ───────────────────────────────────────────────────────────────
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN  = os.environ.get("CLOUDFLARE_API_TOKEN", "")
GROQ_KEY      = os.environ.get("GROQ_KEY")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY")

CF_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"
CF_HEADERS  = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}

# ── Model catalogues ──────────────────────────────────────────────────────────
TEXT_MODELS = {
    "Llama 4 Scout (Cloudflare)":  {"provider": "cloudflare", "id": "@cf/meta/llama-4-scout-17b-16e-instruct"},
    "Llama 3.1 8B (Cloudflare)":   {"provider": "cloudflare", "id": "@cf/meta/llama-3.1-8b-instruct"},
    "Kimi K2.6 (Cloudflare)":      {"provider": "cloudflare", "id": "@cf/moonshotai/kimi-k2.6"},
    "GPT-OSS 20B (Cloudflare)":    {"provider": "cloudflare", "id": "@cf/openai/gpt-oss-20b"},
    "Llama 3.3 70B (Groq)":        {"provider": "groq",       "id": "llama-3.3-70b-versatile"},
    "Llama 3.1 8B Instant (Groq)": {"provider": "groq",       "id": "llama-3.1-8b-instant"},
    "Gemini Flash (Google)":       {"provider": "gemini",     "id": "gemini-flash-latest"},
}
IMAGE_MODELS = {
    "Flux 1 Schnell (Cloudflare)":        {"provider": "cloudflare", "id": "@cf/black-forest-labs/flux-1-schnell"},
    "Flux 2 Dev (Cloudflare)":            {"provider": "cloudflare", "id": "@cf/black-forest-labs/flux-2-dev"},
    "Leonardo Lucid Origin (Cloudflare)": {"provider": "cloudflare", "id": "@cf/leonardo/lucid-origin"},
    "Leonardo Phoenix 1.0 (Cloudflare)":  {"provider": "cloudflare", "id": "@cf/leonardo/phoenix-1.0"},
    "Gemini Flash Image (Google)":        {"provider": "gemini",     "id": "gemini-2.5-flash-image"},
}
AUDIO_MODELS = {
    "Deepgram Aura-1 (Cloudflare)": {"provider": "cloudflare", "id": "@cf/deepgram/aura-1"},
    "Deepgram Aura-2 (Cloudflare)": {"provider": "cloudflare", "id": "@cf/deepgram/aura-2-es"},
    "MeloTTS (Cloudflare)":         {"provider": "cloudflare", "id": "@cf/myshell-ai/melotts"},
    "Gemini TTS (Google)":          {"provider": "gemini",     "id": "gemini-2.5-flash-preview-tts"},
}

# A pool rather than a fixed row: six are drawn from it at random, so the
# suggestions change on every refresh instead of showing the same six forever.
# Kept deliberately varied - places, creatures, ideas, single moments - because
# all three models have to do something with whichever one is picked.
EXAMPLE_PROMPTS = [
    "A lighthouse keeper's last night on duty",
    "Tokyo ramen shop, 2 a.m., raining",
    "How octopuses taste with their arms",
    "A library grown from living trees",
    "Mars colony breakfast, year 40",
    "The last payphone in America",
    "A beekeeper explaining winter to the hive",
    "A night train crossing Siberia in February",
    "Why flamingos stand on one leg",
    "The last bookshop in a flooded city",
    "A blacksmith's forge at dawn",
    "How ravens remember human faces",
    "An abandoned amusement park in autumn",
    "The moment before a thunderstorm breaks",
    "A watchmaker's bench, magnified",
    "Deep sea creatures that make their own light",
    "A street market in Marrakesh at dusk",
    "The physics of a perfect skipping stone",
    "A shepherd's hut in the Scottish Highlands",
    "Why some trees share nutrients underground",
    "A jazz club that never closes",
    "The first frost on a spiderweb",
    "An astronaut's first hour back on Earth",
    "How salmon find the river they were born in",
    "A candlelit monastery scriptorium",
    "The last ice cream van of summer",
    "Fog rolling into San Francisco Bay",
    "A potter centering clay on the wheel",
    "Termites building a cathedral of mud",
    "The quietest room in the world",
]

#: how many suggestion chips to show at once
EXAMPLE_COUNT = 6

# ── Scoring params ────────────────────────────────────────────────────────────
LAMBDA    = 0.5
LR_PARAMS = {
    "mu_good": [0.8400, 0.4656, 0.8723],
    "sd_good": [0.0088, 0.0255, 0.0153],
    "mu_bad":  [0.8372, 0.4585, 0.8706],
    "sd_bad":  [0.0097, 0.0245, 0.0153],
    "r": 0.1604, "pairwise": 57.4,
}

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'
    '&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400'
    '&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">'
)

# ── Trio CSS ──────────────────────────────────────────────────────────────────
TRIO_CSS = """
/* ===== Trio design system ===== */
.tp,.tp *,.tp *::before,.tp *::after{box-sizing:border-box}
.tp p,.tp h1,.tp h2,.tp h3,.tp figure,.tp ul{margin:0;padding:0}
.tp ul{list-style:none}
.tp{
  --ink:#17150F;--ink-70:#4A463B;--ink-45:#7C7767;
  --rule:#DCD5C4;--rule-str:#C3BAA4;
  --paper:#F3EEE1;--card:#FCFAF3;
  --accent:#CE3B16;--accent-soft:#F7E3DB;
  --voice:#1C5E54;--voice-soft:#DDE9E5;
  --wait:#9A6B00;--wait-soft:#F5E8CB;
  --r:3px;--shadow:3px 3px 0 var(--rule-str);
  container-type:inline-size;
  font-family:"Space Grotesk",system-ui,sans-serif;
  color:var(--ink);background:var(--paper);
  border:1px solid var(--rule-str);border-radius:var(--r);
  padding:16px;-webkit-font-smoothing:antialiased;
}
.tp-page-root{border:none !important;box-shadow:none !important;border-radius:0 !important;padding:20px 16px !important;width:100% !important;box-sizing:border-box !important}
.tp .mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
.tp .prose{font-family:"Source Serif 4",Georgia,serif}
/* ---------- Masthead ---------- */
.tp-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding-bottom:12px;border-bottom:1px solid var(--rule)}
.tp-mark{font-size:19px;font-weight:700;letter-spacing:-0.02em;line-height:1}
.tp-mark span{color:var(--accent)}
.tp-flow{display:flex;align-items:center;gap:7px;font-size:11.5px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-70)}
.tp-flow b{font-weight:500;color:var(--ink)}
.tp-arrow{color:var(--ink-45)}
.tp-out{display:inline-flex;align-items:center;gap:5px;padding:3px 7px;border:1px solid var(--rule-str);border-radius:2px;background:var(--card);font-weight:500;color:var(--ink)}
.tp-out i{width:6px;height:6px;border-radius:50%;background:var(--ink);font-style:normal}
.tp-out.is-text i{background:var(--accent)}
.tp-out.is-image i{background:var(--ink)}
.tp-out.is-audio i{background:var(--voice)}
.tp-head-right{margin-left:auto}
/* ---------- Prompt row ---------- */
.tp-form{padding:14px 0 10px}
.tp-row{display:flex;gap:8px;flex-wrap:wrap}
.tp-field{flex:1 1 240px;min-width:0;display:flex;align-items:center;background:var(--card);border:1.5px solid var(--ink);border-radius:var(--r);box-shadow:var(--shadow);transition:box-shadow .18s ease,transform .18s ease}
.tp-field:focus-within{box-shadow:3px 3px 0 var(--accent)}
.tp-input{flex:1;min-width:0;border:0;background:none;outline:none;font:500 16px/1.3 "Space Grotesk",sans-serif;color:var(--ink);padding:13px 14px}
.tp-input::placeholder{color:var(--ink-45);font-weight:500}
.tp-go{flex:0 0 auto;border:1.5px solid var(--ink) !important;border-radius:var(--r) !important;background:var(--accent) !important;color:#fff !important;cursor:pointer !important;font:600 15px/1 "Space Grotesk",sans-serif !important;letter-spacing:.01em;padding:0 20px !important;min-height:48px;box-shadow:var(--shadow) !important;display:inline-flex !important;align-items:center;gap:8px;transition:transform .15s ease,box-shadow .15s ease,background .15s ease}
.tp-go:hover{transform:translate(-1px,-1px) !important;box-shadow:4px 4px 0 var(--rule-str) !important;background:var(--accent) !important}
.tp-go:active{transform:translate(2px,2px) !important;box-shadow:1px 1px 0 var(--rule-str) !important}
.tp-go:disabled{background:var(--ink-45) !important;cursor:default !important;box-shadow:var(--shadow) !important;transform:none !important}
.tp-go .dot3{display:inline-flex;gap:3px}
.tp-go .dot3 i{width:4px;height:4px;border-radius:50%;background:#fff;animation:tp-blink 1.05s infinite;font-style:normal}
.tp-go .dot3 i:nth-child(2){animation-delay:.15s}
.tp-go .dot3 i:nth-child(3){animation-delay:.3s}
@keyframes tp-blink{0%,100%{opacity:.28}45%{opacity:1}}
/* ---------- Example chips ---------- */
.tp-egs{padding-top:12px}
.tp-egs-label{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-45);margin-bottom:8px;display:block}
.tp-chips{display:flex;flex-wrap:wrap;gap:7px}
.tp-chip{display:inline-flex !important;align-items:center;gap:7px;padding:9px 13px !important;min-height:40px !important;background:var(--card) !important;color:var(--ink) !important;cursor:pointer !important;border:1px solid var(--rule-str) !important;border-radius:100px !important;font:500 13.5px/1.1 "Space Grotesk",sans-serif !important;text-align:left;transition:transform .16s ease,border-color .16s ease,background .16s ease,box-shadow .16s ease;animation:tp-chip-in .34s cubic-bezier(.2,.7,.3,1) backwards}
.tp-chip::before{content:"\\21B5";font-size:12px;color:var(--accent);opacity:.65;transition:opacity .16s ease,transform .16s ease}
.tp-chip:hover,.tp-chip:focus-visible{border-color:var(--ink) !important;background:var(--accent-soft) !important;transform:translateY(-2px) !important;box-shadow:0 3px 0 var(--rule-str) !important;outline:none !important;color:var(--ink) !important}
.tp-chip:hover::before{opacity:1;transform:translateX(2px)}
.tp-chip:active{transform:translateY(0) !important;box-shadow:none !important}
.tp-chip:nth-child(1){animation-delay:.02s}.tp-chip:nth-child(2){animation-delay:.07s}
.tp-chip:nth-child(3){animation-delay:.12s}.tp-chip:nth-child(4){animation-delay:.17s}
.tp-chip:nth-child(5){animation-delay:.22s}.tp-chip:nth-child(6){animation-delay:.27s}
@keyframes tp-chip-in{from{opacity:0;transform:translateY(6px)}}
/* ---------- Model settings ---------- */
.tp-settings{margin-top:14px;border-top:1px solid var(--rule);padding-top:10px}
.tp-settings summary{cursor:pointer;list-style:none;display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--ink-70);padding:6px 2px;min-height:32px}
.tp-settings summary::-webkit-details-marker{display:none}
.tp-settings summary::before{content:"+";font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--accent);transition:transform .2s ease}
.tp-settings[open] summary::before{content:"\\2013"}
.tp-settings summary:hover{color:var(--ink)}
.tp-opts{display:grid;gap:14px;padding:10px 2px 4px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.tp-opt-h{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-45);margin-bottom:7px}
.tp-sel{display:block;width:100%;padding:8px 10px;background:var(--card);border:1px solid var(--rule-str);border-radius:var(--r);font:500 13px/1.4 "Space Grotesk",sans-serif;color:var(--ink);cursor:pointer}
.tp-sel:focus{border-color:var(--ink);outline:none}
/* ---------- Results ---------- */
.tp-results{margin-top:14px;display:grid;gap:10px;grid-template-columns:1fr}
@container (min-width:720px){.tp-results{grid-template-columns:1.15fr 1fr 1fr}}
.tp-card{position:relative;overflow:hidden;display:flex;flex-direction:column;background:var(--card);border:1px solid var(--rule-str);border-radius:var(--r);animation:tp-card-in .26s cubic-bezier(.2,.7,.3,1) backwards}
@keyframes tp-card-in{from{opacity:0;transform:translateY(8px)}}
.tp-card.is-done{border-color:var(--ink);box-shadow:var(--shadow)}
.tp-card-h{display:flex;align-items:center;gap:8px;padding:9px 11px;border-bottom:1px solid var(--rule);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-70)}
.tp-card-h b{font-weight:500;color:var(--ink)}
.tp-card-h .tp-acts{margin-left:auto;display:flex;gap:6px}
.tp-tick{color:var(--voice);font-size:12px;letter-spacing:0}
.tp-lane{position:absolute;inset:0 0 auto 0;height:3px;background:var(--rule)}
.tp-lane::after{content:"";display:block;height:100%;width:100%;background:var(--accent);transform-origin:0 50%;animation:tp-fill 2s cubic-bezier(.25,.6,.2,1) forwards}
.tp-lane.lane-img::after{background:var(--ink);animation-duration:1.5s}
.tp-lane.lane-audio::after{background:var(--voice);animation-duration:1.7s}
@keyframes tp-fill{from{transform:scaleX(0)}70%{transform:scaleX(.82)}to{transform:scaleX(1)}}
.tp-eta{margin-left:auto;font-size:10.5px;letter-spacing:.04em;color:var(--ink-45)}
.tp-body{padding:12px;flex:1}
.tp-text{font-size:15.5px;line-height:1.55;color:var(--ink);text-wrap:pretty}
.tp-text+.tp-text{margin-top:.7em}
.tp-skel{display:grid;gap:9px}
.tp-skel i{display:block;height:11px;border-radius:2px;background:linear-gradient(90deg,var(--rule) 0%,#EFE9DA 45%,var(--rule) 90%);background-size:220% 100%;animation:tp-sweep 1.15s linear infinite;font-style:normal}
.tp-skel i:nth-child(2){animation-delay:.1s}.tp-skel i:nth-child(3){animation-delay:.2s}
.tp-skel i:nth-child(4){animation-delay:.3s;width:62%}
@keyframes tp-sweep{to{background-position:-220% 0}}
.tp-figure{aspect-ratio:1/1;border:1px solid var(--rule);border-radius:2px;overflow:hidden;background:repeating-linear-gradient(135deg,#E9E2D2 0 9px,#E3DBC9 9px 18px);display:grid;place-items:center}
.tp-figure img{width:100%;height:100%;object-fit:cover;display:block}
.tp-figure span{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-45)}
.tp-figure.is-loading{position:relative}
.tp-figure.is-loading::after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent 20%,rgba(255,255,255,.75) 50%,transparent 80%);background-size:250% 100%;animation:tp-sweep 1.3s linear infinite}
.tp-wave{flex:1;display:flex;align-items:center;gap:2.5px;height:40px;min-width:0}
.tp-wave i{flex:1;background:var(--voice);opacity:.38;border-radius:1px;height:30%;font-style:normal}
.tp-wave i:nth-child(3n){height:68%}.tp-wave i:nth-child(3n+1){height:44%}
.tp-wave i:nth-child(4n){height:88%}.tp-wave i:nth-child(5n){height:22%}
.tp-wave.is-live i{animation:tp-bounce 1.05s ease-in-out infinite;opacity:.85}
.tp-wave.is-live i:nth-child(2n){animation-delay:.09s}.tp-wave.is-live i:nth-child(3n){animation-delay:.18s}
.tp-wave.is-live i:nth-child(5n){animation-delay:.27s}.tp-wave.is-live i:nth-child(7n){animation-delay:.36s}
@keyframes tp-bounce{0%,100%{transform:scaleY(.35)}50%{transform:scaleY(1)}}
.tp-time{font-size:11px;color:var(--ink-45)}
.tp-btn{display:inline-flex !important;align-items:center;gap:6px;cursor:pointer !important;min-height:34px !important;padding:6px 11px !important;border:1px solid var(--rule-str) !important;border-radius:2px !important;background:#fff !important;color:var(--ink) !important;font:500 11.5px/1 "IBM Plex Mono",monospace !important;letter-spacing:.02em;transition:border-color .15s ease,background .15s ease,transform .15s ease;text-decoration:none !important}
.tp-btn:hover{border-color:var(--ink) !important;background:var(--accent-soft) !important;color:var(--ink) !important}
.tp-btn:active{transform:translateY(1px) !important}
.tp-card.is-failed{border-color:var(--wait);border-style:dashed;background:#FDFBF4}
.tp-card.is-failed .tp-card-h{color:var(--wait);border-bottom-color:var(--wait-soft)}
.tp-fail{display:grid;gap:10px;justify-items:start}
.tp-fail p{font-size:13.5px;line-height:1.45;color:var(--ink-70)}
.tp-fail p b{color:var(--ink);font-weight:600}
.tp-foot{margin-top:14px;font-size:11px;color:var(--ink-45);display:flex;gap:10px;flex-wrap:wrap}
/* ---------- Coherence ---------- */
.tp-coh{margin-top:10px;border-top:1px solid var(--rule)}
.tp-coh>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:9px 2px;min-height:42px}
.tp-coh>summary::-webkit-details-marker{display:none}
.tp-coh-k{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-45)}
.tp-coh-n{font:700 21px/1 "Space Grotesk",sans-serif;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tp-coh-read{font-family:"Source Serif 4",Georgia,serif;font-style:italic;font-size:14.5px;color:var(--ink-70)}
.tp-coh-more{margin-left:auto;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);display:inline-flex;gap:5px;align-items:center}
.tp-coh-more::after{content:"\\25BE";transition:transform .2s ease}
.tp-coh[open] .tp-coh-more::after{transform:rotate(180deg)}
.tp-coh[open] .tp-coh-more span::after{content:" breakdown"}
.tp-coh>summary:hover .tp-coh-more{text-decoration:underline}
.tp-seg{display:inline-block;flex:0 0 auto;width:98px;height:8px;overflow:hidden;background:repeating-linear-gradient(90deg,var(--rule) 0 8px,transparent 8px 10px)}
.tp-seg i{display:block;height:100%;width:var(--w);background:repeating-linear-gradient(90deg,var(--ink) 0 8px,transparent 8px 10px);animation:tp-grow .24s cubic-bezier(.2,.7,.3,1) backwards;font-style:normal}
.tp-seg.is-mid i{background-image:repeating-linear-gradient(90deg,var(--wait) 0 8px,transparent 8px 10px)}
.tp-seg.is-good i{background-image:repeating-linear-gradient(90deg,var(--voice) 0 8px,transparent 8px 10px)}
.tp-seg.is-off i{background-image:repeating-linear-gradient(90deg,var(--rule-str) 0 8px,transparent 8px 10px)}
@keyframes tp-grow{from{width:0}}
.tp-coh-body{padding:2px 2px 12px;display:grid;gap:7px;animation:tp-coh-in .22s ease-out}
@keyframes tp-coh-in{from{opacity:0;transform:translateY(-4px)}}
.tp-pair{display:flex;align-items:center;gap:10px;font-size:11.5px}
.tp-pair-k{display:inline-flex;align-items:center;gap:6px;min-width:128px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-70)}
.tp-pair-k i{width:6px;height:6px;border-radius:50%;display:inline-block;font-style:normal}
.tp-pair-k .d-text{background:var(--accent)}.tp-pair-k .d-img{background:var(--ink)}
.tp-pair-k .d-audio{background:var(--voice)}.tp-pair-k .d-off{background:var(--rule-str)}
.tp-pair-v{font-variant-numeric:tabular-nums;font-weight:500}
.tp-pair.is-off{color:var(--ink-45)}
.tp-coh-foot{font-size:11.5px;line-height:1.5;color:var(--ink-45);margin-top:3px;max-width:62ch}
.tp-coh-foot b{color:var(--ink-70);font-weight:600}
/* ---------- Method block ---------- */
.tp-method{margin-top:12px;border:1px solid var(--rule-str);border-radius:var(--r);background:var(--card)}
.tp-method>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:8px;padding:11px 12px;min-height:44px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-70);font-family:"IBM Plex Mono",monospace}
.tp-method>summary::-webkit-details-marker{display:none}
.tp-method>summary::before{content:"+";color:var(--accent);font-size:13px}
.tp-method[open]>summary::before{content:"\\2013"}
.tp-method>summary em{font-style:normal;color:var(--ink-45);margin-left:auto;letter-spacing:.04em}
.tp-method-b{border-top:1px solid var(--rule);animation:tp-coh-in .22s ease-out}
.tp-method-intro{padding:11px 12px 4px;font-size:12.5px;line-height:1.5;color:var(--ink-70);max-width:72ch}
.tp-appr{display:grid;grid-template-columns:1fr auto;align-items:center;gap:4px 16px;padding:11px 12px;border-top:1px solid var(--rule)}
.tp-appr.is-best{background:#fff}
.tp-appr-n{font:600 13.5px/1.2 "Space Grotesk",sans-serif;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tp-appr-n small{font-weight:500;color:var(--ink-45);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
.tp-badge{font:500 9.5px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;text-transform:uppercase;background:var(--accent);color:#fff;padding:4px 6px;border-radius:2px}
.tp-f{grid-column:1;font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-70);overflow-x:auto;white-space:nowrap;padding-bottom:1px}
.tp-r{grid-column:2;grid-row:1/span 2;display:flex;align-items:center;gap:10px}
.tp-r b{font:600 13px/1 "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;min-width:52px;text-align:right}
.tp-rbar{width:110px;height:10px;background:var(--rule);border-radius:1px;overflow:hidden}
.tp-rbar i{display:block;height:100%;width:var(--w);background:var(--ink-45);animation:tp-grow .26s cubic-bezier(.2,.7,.3,1) backwards;font-style:normal}
.tp-appr.is-best .tp-rbar i{background:var(--accent);animation-delay:.1s}
.tp-scale{display:flex;justify-content:flex-end;gap:6px;padding:0 12px 10px;font:400 10px/1 "IBM Plex Mono",monospace;color:var(--ink-45);letter-spacing:.06em}
.tp-mnote{padding:0 12px 12px;font-size:12.5px;line-height:1.55;color:var(--ink-70);max-width:74ch}
.tp-honest{margin:0 12px 12px;padding:11px 13px;background:#fff;border:1.5px solid var(--ink);border-left:5px solid var(--accent);border-radius:2px;font-size:13.5px;line-height:1.55;color:var(--ink)}
.tp-honest b{font-weight:600}
.tp-honest span{display:block;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;font-family:"IBM Plex Mono",monospace;color:var(--accent);margin-bottom:5px}
/* ---------- Reduced motion ---------- */
@media (prefers-reduced-motion: reduce){
  .tp *,.tp *::before,.tp *::after{animation-duration:.001ms !important;animation-iteration-count:1 !important;transition-duration:.001ms !important}
  .tp-lane::after{transform:scaleX(1)}
}
"""

# ── Gradio page overrides (injected globally via launch()) ────────────────────
GRADIO_CSS = """
/* Force light colour scheme — Trio is a warm-paper light design */
:root, html { color-scheme: light !important; }
html, body {
  background: #E7E1D2 !important;
  color: #17150F !important;
  margin: 0 !important; padding: 0 !important;
}
gradio-app, .gradio-container {
  background: #E7E1D2 !important;
  /* was a flat 960px, which left ~520px of dead background on each side of a
     1920px window. Fluid instead: grows with the window, capped so the result
     paragraph never becomes an unreadably long line on an ultrawide monitor.
     Tune the 1400px if you want it tighter or fuller. */
  max-width: min(1400px, 94vw) !important;
  margin: 0 auto !important;
  padding: 0 !important;
  width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
}
/* All gr.HTML / block wrappers fill parent width; reset Gradio's dark text */
.block, .prose, .html-container, [data-testid="html"] {
  width: 100% !important; max-width: none !important; min-width: 0 !important;
  box-sizing: border-box !important;
  color: #17150F !important;
}
/* Override every descendant of .prose so dark-theme near-white doesn't leak in */
.prose *, .html-container * { color: inherit; }
/* Trio root forces its own ink (beats any inherited near-white from Gradio) */
.tp, .tp-page-root { color: #17150F !important; background-color: #F3EEE1 !important; }
/* Input field — override Gradio dark-theme input background/text */
.tp-field { background-color: #FCFAF3 !important; }
.tp-input {
  background: transparent !important;
  color: #17150F !important;
  -webkit-text-fill-color: #17150F !important;
  caret-color: #17150F !important;
}
.tp-input::placeholder { color: #7C7767 !important; -webkit-text-fill-color: #7C7767 !important; opacity: 1 !important; }
/* Exceptions — elements that have their own intentional colours */
.tp-go, .tp-go * { color: #fff !important; }
/* Ink variants — explicitly resolve CSS vars as hex to avoid dark-mode bleed */
.tp-flow, .tp-arrow, .tp-out, .tp-egs-label, .tp-opt-h,
.tp-card-h, .tp-eta, .tp-time, .tp-tick,
.tp-coh-k, .tp-coh-read, .tp-pair-k, .tp-coh-foot,
.tp-method > summary, .tp-settings summary { color: #4A463B !important; }
.tp-mark, .tp-flow b, .tp-card-h b, .tp-out.is-text,
.tp-out.is-image, .tp-out.is-audio { color: #17150F !important; }
.tp-mark span, .tp-settings summary::before, .tp-method > summary::before,
.tp-coh-more, .tp-badge { color: #CE3B16 !important; }
.tp-chip { color: #17150F !important; }
.tp-btn { color: #17150F !important; }
.tp-tick { color: #1C5E54 !important; }
/* No vertical gap between stacked blocks */
.gap { gap: 0 !important; }
/* Off-screen bridge components (kept in DOM with visible=True) */
#trio-data-hidden, #trio-btn-hidden {
  position: fixed !important; top: -9999px !important; left: -9999px !important;
  width: 1px !important; height: 1px !important;
  overflow: hidden !important; opacity: 0 !important; pointer-events: none !important;
  z-index: -1 !important;
}
/* Results area hidden by default — JS overrides via inline style when needed */
#trio-results-area { min-height: 4px !important; display: none; }
/* Again-btn starts hidden; ID specificity beats .tp-go class's display:inline-flex !important */
#trio-again-btn { display: none !important; }
/* Hide Gradio footer */
footer, .built-with { display: none !important; }

/* Tricoherence verdict pill, in place of a 0-1 bar: S(p) is an unbounded
   log-ratio, so a percentage meter would be meaningless for it. */
.tp-verdict {
  font: 500 9.5px/1 "IBM Plex Mono", monospace; letter-spacing: .12em;
  text-transform: uppercase; padding: 4px 7px; border-radius: 2px;
  border: 1px solid var(--rule-str); color: var(--ink-45); background: #fff;
  white-space: nowrap;
}
.tp-verdict.is-good { border-color: var(--voice); color: var(--voice); background: var(--voice-soft); }
.tp-verdict.is-bad  { border-color: var(--wait);  color: var(--wait);  background: var(--wait-soft); }
.tp-appr-n { align-items: baseline; }


/* typewriter caret - the design's blink keyframes, borrowed from .dot3 */
.tp-caret::after {
  content: ""; display: inline-block; width: 7px; height: 1.05em;
  margin-left: 2px; background: var(--accent); vertical-align: -2px;
  animation: tp-blink 1s steps(1) infinite;
}
/* while typing, hold the final height so the card does not grow line by line
   and shove the coherence panel down the page on every frame */
.tp-body[data-tw][data-typing] { min-height: var(--tw-h, auto); }


/* ── Result-page detail (local build) ────────────────────────────────────── */

/* prompt echo: the form is hidden on the results page, so this is the only
   thing telling you what was asked */
.tp-echo {
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  margin: 2px 0 14px; padding-bottom: 12px;
  border-bottom: 1px solid var(--rule);
  font-family: "Source Serif 4", Georgia, serif; font-size: 17px; color: var(--ink);
}
.tp-echo-k {
  font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-45); flex: 0 0 auto;
}

/* model name in a card header, pushed left of the action buttons */
.tp-model {
  font-size: 10px; letter-spacing: .04em; color: var(--ink-45);
  margin-left: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 14ch;
}
@container (min-width: 720px) { .tp-model { max-width: 20ch; } }

/* Whisper transcript under the audio player */
.tp-transcript { margin-top: 12px; border-top: 1px solid var(--rule); padding-top: 10px; }
.tp-transcript-h {
  font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-45); margin-bottom: 6px;
}
.tp-transcript-b {
  font-size: 13.5px; line-height: 1.55; color: var(--ink-70);
  font-style: italic; text-wrap: pretty;
}

/* staggered entrance: the cards already share one fade-and-rise animation,
   so they all landed at the same instant. Delaying each lane by 110ms lets
   them arrive left to right instead of popping in as a block. */
.tp-results .tp-card:nth-child(1) { animation-delay: 0ms; }
.tp-results .tp-card:nth-child(2) { animation-delay: 110ms; }
.tp-results .tp-card:nth-child(3) { animation-delay: 220ms; }

@media (prefers-reduced-motion: reduce) {
  .tp-results .tp-card { animation-delay: 0ms !important; }
}


/* ── Empty-state preview + a Gradio artefact fix (local build) ───────────── */

/* Gradio wraps the off-screen bridge textbox in its own .form div. Only the
   textbox was moved to -9999px, so the wrapper stayed behind as a 2px bar in
   Gradio's dark-theme colour, drawn across the page under the card. */
#trio-data-hidden, #trio-btn-hidden { border: 0 !important; background: none !important; }
.gradio-container .form:has(#trio-data-hidden),
.gradio-container .form:has(#trio-btn-hidden) {
  position: fixed !important; top: -9999px !important; left: -9999px !important;
  border: 0 !important; background: none !important; box-shadow: none !important;
}

.tp-ghost { margin-top: 18px; border-top: 1px solid var(--rule); padding-top: 12px; }
.tp-ghost-h {
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-45); margin-bottom: 10px;
}
/* inert on purpose: dashed and flat, so it never reads as "loading" */
.tp-ghost-card { border-style: dashed; background: transparent; opacity: .72; }
.tp-ghost-card .tp-card-h { border-bottom-color: var(--rule); color: var(--ink-45); }
.tp-ghost-card .tp-card-h b { color: var(--ink-45); font-weight: 500; }
.tp-ghost-card .tp-figure { opacity: .55; }
.tp-ghost-card .tp-wave { opacity: .45; }
.tp-ghost-lines { display: grid; gap: 9px; }
.tp-ghost-lines i {
  display: block; height: 10px; border-radius: 2px; background: var(--rule);
}
.tp-ghost-lines i:nth-child(2) { width: 92%; }
.tp-ghost-lines i:nth-child(3) { width: 97%; }
.tp-ghost-lines i:nth-child(4) { width: 58%; }
/* the three previews have different content heights (a square image is much
   taller than four skeleton lines), so pin every caption to the bottom of its
   card - otherwise the row reads as three mismatched boxes */
.tp-ghost-card .tp-body { display: flex; flex-direction: column; }
.tp-ghost-cap {
  margin-top: auto; padding-top: 10px; font-size: 11.5px; color: var(--ink-45);
  font-family: "IBM Plex Mono", monospace;
}

"""

# ── JavaScript for the static Trio UI ────────────────────────────────────────
# Note: this is a plain string, not f-string, so { } are literal JS braces
TRIO_JS = r"""
if (!window._trioSetup) {
  window._trioSetup = true;

  // Current page: '01'=form, '02'=generating, '03'=complete, '04'=partial/fail
  window._trioPage = '01';

  // Re-draw the suggestion chips from the full pool. Gradio builds the static
  // HTML once at import, so a server-side sample would only change when the
  // process restarts - doing it here means a plain refresh gives new ones.
  window.trioShuffleChips = function() {
    var row = document.querySelector('.tp-chips');
    if (!row || !row.dataset.pool) return;
    var pool;
    try { pool = JSON.parse(row.dataset.pool); } catch (e) { return; }
    if (!Array.isArray(pool) || !pool.length) return;

    var n = Math.min(parseInt(row.dataset.count, 10) || 6, pool.length);
    var picked = pool.slice();
    // Fisher-Yates over just the first n slots
    for (var i = 0; i < n; i++) {
      var j = i + Math.floor(Math.random() * (picked.length - i));
      var t = picked[i]; picked[i] = picked[j]; picked[j] = t;
    }
    picked = picked.slice(0, n);

    row.textContent = '';
    picked.forEach(function(text) {
      var b = document.createElement('button');
      b.className = 'tp-chip';
      b.type = 'button';
      b.dataset.p = text;
      b.textContent = text;              // textContent, so no markup can slip in
      b.addEventListener('click', function() { trioRunChip(text); });
      row.appendChild(b);
    });
  };

  // ── Typewriter reveal for the generated paragraph ───────────────────────
  // The model returns the whole paragraph at once, so this is a reveal, not a
  // token stream: the markup is already in the DOM and we walk its text nodes
  // blanking and then restoring them. Doing it that way keeps the headings,
  // bold runs and paragraph breaks that markdown produced - typing into
  // textContent would flatten all of it.
  window._trioTyped = null;

  window.trioTypewriter = function(el) {
    if (!el) return;

    // collect every text node, in order, and remember what it said
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    var parts = [], node;
    while ((node = walker.nextNode())) {
      if (node.nodeValue && node.nodeValue.length) {
        parts.push({ node: node, text: node.nodeValue });
      }
    }
    if (!parts.length) return;

    var total = parts.reduce(function(n, p) { return n + p.text.length; }, 0);

    // reduced motion, or a suspiciously huge paragraph: just show it
    var reduce = window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || total > 6000) return;

    // pin the finished height first, so the card does not grow line by line
    // and push everything below it down on every frame
    el.style.setProperty('--tw-h', el.getBoundingClientRect().height + 'px');
    el.setAttribute('data-typing', '');

    parts.forEach(function(p) { p.node.nodeValue = ''; });

    // Fixed duration rather than a fixed character rate: paragraphs vary from
    // ~400 to ~1200 characters, and a constant rate makes the long ones crawl.
    // TYPE_MS is the knob - raise it to slow the reveal down, lower to speed up.
    // Short paragraphs finish sooner than this, because the per-frame budget
    // bottoms out at one character.
    var TYPE_MS = 3200, FRAME = 16;
    var perFrame = Math.max(1, Math.ceil(total / (TYPE_MS / FRAME)));

    var pi = 0, ci = 0;
    function tick() {
      var budget = perFrame;
      while (budget > 0 && pi < parts.length) {
        var p = parts[pi];
        var take = Math.min(budget, p.text.length - ci);
        ci += take; budget -= take;
        p.node.nodeValue = p.text.slice(0, ci);

        // caret rides the element currently being filled
        var host = p.node.parentElement;
        if (host && host !== window._trioCaretHost) {
          if (window._trioCaretHost) window._trioCaretHost.classList.remove('tp-caret');
          host.classList.add('tp-caret');
          window._trioCaretHost = host;
        }
        if (ci >= p.text.length) { pi++; ci = 0; }
      }
      if (pi < parts.length) {
        requestAnimationFrame(tick);
      } else {
        if (window._trioCaretHost) {
          window._trioCaretHost.classList.remove('tp-caret');
          window._trioCaretHost = null;
        }
        el.removeAttribute('data-typing');
        el.style.removeProperty('--tw-h');
      }
    }
    requestAnimationFrame(tick);
  };

  // Watch the results area. generate() yields twice - once when the three
  // lanes finish and again ~25s later with the scores - and the second yield
  // replaces the same markup. Without remembering what was already typed, the
  // paragraph would type itself a second time when the coherence panel lands.
  window.trioWatchResults = function() {
    var area = document.querySelector('#trio-results-area');
    if (!area || area._trioObserved) return;
    area._trioObserved = true;
    new MutationObserver(function() {
      var el = area.querySelector('[data-tw]');
      if (!el) return;
      var text = el.textContent.trim();
      if (!text || text === window._trioTyped) return;
      window._trioTyped = text;
      window.trioTypewriter(el);
    }).observe(area, { childList: true, subtree: true });
  };

  // ── Elapsed timer ───────────────────────────────────────────────────────
  // The status used to show three blinking dots, which look identical at one
  // second and at thirty. A counting clock tells you it is still working and
  // roughly how long this one is taking.
  window._trioTimer = null;

  window.trioStartTimer = function(label) {
    window.trioStopTimer();
    var el = document.getElementById('trio-status');
    if (!el) return;
    var t0 = Date.now();
    var render = function() {
      var s = (Date.now() - t0) / 1000;
      el.textContent = (label ? label + ' ' : '') + s.toFixed(1) + 's';
    };
    render();
    window._trioTimer = setInterval(render, 100);
  };

  window.trioStopTimer = function() {
    if (window._trioTimer) { clearInterval(window._trioTimer); window._trioTimer = null; }
  };

  // ── Retry just the audio lane ───────────────────────────────────────────
  // The old button called trioGenerate(), which re-ran all three models: the
  // label was wrong, it cost triple the neurons, and the image changed under
  // someone who was happy with it. The server keeps the finished text and
  // image against this token and only calls the voice model.
  window.trioRetryAudio = function() {
    var meta = document.getElementById('trio-gen-meta');
    var token = meta && meta.getAttribute('data-token');
    if (!token) { trioGenerate(); return; }   // nothing cached: full run

    var ta = document.querySelector('#trio-data-hidden textarea');
    if (!ta) return;
    var data = {
      retry: 'audio',
      token: token,
      audio_model: (document.getElementById('trio-audio-sel') || {value: ''}).value || '',
      scoring: false
    };
    var setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, JSON.stringify(data));
    ta.dispatchEvent(new Event('input', {bubbles: true}));

    var goBtn = document.getElementById('trio-go');
    if (goBtn) goBtn.disabled = true;
    window.trioStartTimer('retrying audio');

    setTimeout(function() {
      var btn = document.querySelector('#trio-btn-hidden');
      if (btn) btn.click();
    }, 60);
  };

  window.trioSync = function() {
    var data = {
      prompt: (document.getElementById('trio-prompt-field') || {value: ''}).value || '',
      text_model: (document.getElementById('trio-text-sel') || {value: ''}).value || '',
      image_model: (document.getElementById('trio-image-sel') || {value: ''}).value || '',
      audio_model: (document.getElementById('trio-audio-sel') || {value: ''}).value || '',
      scoring: true
    };
    var ta = document.querySelector('#trio-data-hidden textarea');
    if (!ta) return false;
    var setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, JSON.stringify(data));
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    return true;
  };

  // Page 01 — show form, hide results (back-nav from results pages)
  window.trioGoToForm = function() {
    window._trioPage = '01';
    var formView = document.getElementById('tp-view-form');
    var resultsArea = document.querySelector('#trio-results-area');
    var againBtn = document.getElementById('trio-again-btn');
    var goBtn = document.getElementById('trio-go');
    var statusEl = document.getElementById('trio-status');
    if (formView) formView.style.display = 'block';
    if (resultsArea) resultsArea.style.display = 'none';
    if (againBtn) againBtn.style.removeProperty('display');
    if (goBtn) { goBtn.disabled = false; }
    window.trioStopTimer();
    if (statusEl) statusEl.innerHTML = '~2s · free';
    setTimeout(function() {
      var inp = document.getElementById('trio-prompt-field');
      // select, not just focus: coming back from a result you almost always
      // want a different prompt, so typing should replace the old one
      if (inp) { inp.focus(); inp.select(); }
    }, 60);
  };

  // Page 02 → 03/04 — hide form, show results (generation flow)
  window.trioGenerate = function() {
    // If already on results page, "Generate again" = go back to form (Page 01)
    if (window._trioPage === '03' || window._trioPage === '04') {
      trioGoToForm();
      return;
    }
    if (!trioSync()) { setTimeout(trioGenerate, 100); return; }
    var goBtn = document.getElementById('trio-go');
    var againBtn = document.getElementById('trio-again-btn');
    var statusEl = document.getElementById('trio-status');
    var formView = document.getElementById('tp-view-form');
    var resultsArea = document.querySelector('#trio-results-area');
    window._trioPage = '02';
    if (goBtn) { goBtn.disabled = true; }
    if (againBtn) againBtn.style.removeProperty('display');
    window.trioStartTimer('');
    // Page 02: hide form completely, show results area (Gradio injects loading HTML)
    if (formView) formView.style.display = 'none';
    if (resultsArea) resultsArea.style.display = 'block';
    setTimeout(function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 80);
    setTimeout(function() {
      var btn = document.querySelector('#trio-btn-hidden');
      if (btn) btn.click();
    }, 60);
  };

  // draw a fresh set as soon as the DOM is ready, and start watching results
  function trioBoot() { window.trioShuffleChips(); window.trioWatchResults(); }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', trioBoot);
  } else {
    trioBoot();
  }

  window.trioRunChip = function(text) {
    var inp = document.getElementById('trio-prompt-field');
    if (inp) inp.value = text;
    trioGenerate();
  };

  // Poll every 250ms: gen_meta appearing signals generation done → transition to Page 03/04
  setInterval(function() {
    var goBtn = document.getElementById('trio-go');
    if (!goBtn || !goBtn.disabled) return;
    var meta = document.getElementById('trio-gen-meta');
    if (!meta) return;
    goBtn.disabled = false;
    window.trioStopTimer();
    var againBtn = document.getElementById('trio-again-btn');
    var statusEl = document.getElementById('trio-status');
    var done  = parseInt(meta.getAttribute('data-done')  || '0', 10);
    var total = parseInt(meta.getAttribute('data-total') || '3', 10);
    var time  = parseFloat(meta.getAttribute('data-time') || '0');
    if (done >= total) {
      window._trioPage = '03';
      if (statusEl) statusEl.textContent = 'done in ' + time.toFixed(1) + 's';
      if (againBtn) { againBtn.style.setProperty('display', 'inline-flex', 'important'); againBtn.innerHTML = 'Generate again &#8594;'; }
    } else if (done > 0) {
      window._trioPage = '04';
      if (statusEl) statusEl.textContent = done + ' of ' + total + ' ready';
      if (againBtn) { againBtn.style.setProperty('display', 'inline-flex', 'important'); againBtn.innerHTML = 'Generate again &#8594;'; }
    } else {
      window._trioPage = '04';
      if (statusEl) statusEl.textContent = 'failed';
      if (againBtn) { againBtn.style.setProperty('display', 'inline-flex', 'important'); againBtn.innerHTML = 'Try again &#8594;'; }
    }
  }, 250);
}
"""

# ── Build static Trio UI HTML (no f-strings — all concatenation) ──────────────
# ── Empty-state preview (added for the Mac/local build) ───────────────────────
def _ghost_lanes_html():
    """Outlined, inert versions of the three result cards for the landing page.

    The form on its own leaves roughly 400px of dead space at the bottom of a
    900px window, which reads as unfinished. These sit inside #tp-view-form, so
    trioGenerate() hides them with the rest of the form and trioGoToForm()
    brings them back - no JS changes needed.

    Built only from classes the design already defines, and deliberately NOT
    animated: .tp-skel and .tp-wave.is-live both pulse, which would suggest
    something is loading when nothing has started.
    """
    return (
        '<div class="tp-ghost" aria-hidden="true">'
        '<div class="tp-ghost-h mono">What you get back</div>'
        '<div class="tp-results">'
        '<article class="tp-card tp-ghost-card">'
        '<div class="tp-card-h"><i class="tp-dot d-text"></i><b>Text</b></div>'
        '<div class="tp-body">'
        '<div class="tp-ghost-lines"><i></i><i></i><i></i><i></i></div>'
        '<p class="tp-ghost-cap">a short paragraph</p>'
        '</div></article>'
        '<article class="tp-card tp-ghost-card">'
        '<div class="tp-card-h"><i class="tp-dot d-img"></i><b>Image</b></div>'
        '<div class="tp-body">'
        '<div class="tp-figure"><span>your picture</span></div>'
        '<p class="tp-ghost-cap">one square image</p>'
        '</div></article>'
        '<article class="tp-card tp-ghost-card">'
        '<div class="tp-card-h"><i class="tp-dot d-audio"></i><b>Audio</b></div>'
        '<div class="tp-body">'
        '<div class="tp-wave">' + ('<i></i>' * 28) + '</div>'
        '<p class="tp-ghost-cap">the paragraph, read aloud</p>'
        '</div></article>'
        '</div></div>'
    )


def _build_static_html():
    # Use data-p attribute instead of inline json in onclick to avoid
    # double-quote conflicts that break the HTML attribute parser
    def _html_attr(s):
        return s.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#39;')

    # a random six for the initial render; trioShuffleChips() re-draws them on
    # every page load so a refresh really does change the suggestions
    import random as _random
    _shown = _random.sample(EXAMPLE_PROMPTS, min(EXAMPLE_COUNT, len(EXAMPLE_PROMPTS)))
    chips = "".join(
        '<button class="tp-chip" type="button"'
        + ' data-p="' + _html_attr(p) + '"'
        + ' onclick="trioRunChip(this.dataset.p)">'
        + p + '</button>'
        for p in _shown
    )
    chip_pool = _html_attr(json.dumps(EXAMPLE_PROMPTS))

    def _sel(elem_id, models):
        opts = "".join(
            '<option value="' + k.replace('"', '&quot;') + '">'
            + k.replace('<', '&lt;').replace('>', '&gt;') + '</option>'
            for k in models
        )
        return (
            '<select id="' + elem_id + '" class="tp-sel" onchange="trioSync()">'
            + opts + '</select>'
        )

    text_sel  = _sel('trio-text-sel',  TEXT_MODELS)
    image_sel = _sel('trio-image-sel', IMAGE_MODELS)
    audio_sel = _sel('trio-audio-sel', AUDIO_MODELS)

    return (
        FONTS_LINK
        + '<style>' + TRIO_CSS + '</style>'
        + '<div class="tp tp-page-root">'
        +   '<header class="tp-head">'
        +     '<div class="tp-mark">Trio<span>.</span></div>'
        +     '<div class="tp-flow">'
        +       '<b>1 prompt</b>'
        +       '<span class="tp-arrow">&#8594;</span>'
        +       '<span class="tp-out is-text"><i></i>text</span>'
        +       '<span class="tp-out is-image"><i></i>image</span>'
        +       '<span class="tp-out is-audio"><i></i>audio</span>'
        +     '</div>'
        +     '<div class="tp-head-right" style="display:flex;align-items:center;gap:12px">'
        +       '<div id="trio-status" class="mono" role="status" aria-live="polite"'
        +            ' style="font-size:11px;color:#7C7767">~2s &#183; free</div>'
        +       '<button id="trio-again-btn" class="tp-go" type="button"'
        +         ' onclick="trioGoToForm()">'
        +         'Generate again &#8594;'
        +       '</button>'
        +     '</div>'
        +   '</header>'
        # Page 01: form — hidden when on Page 02/03/04
        +   '<section id="tp-view-form">'
        +   '<div class="tp-form">'
        +     '<div class="tp-row">'
        +       '<div class="tp-field">'
        +         '<input class="tp-input" id="trio-prompt-field" type="text"'
        +         ' placeholder="Describe anything &#8212; a place, a creature, an idea&#8230;"'
        +         ' oninput="trioSync()"'
        +         ' onkeydown="if(event.key===\'Enter\')trioGenerate()">'
        +       '</div>'
        +       '<button class="tp-go" id="trio-go" type="button" onclick="trioGenerate()">'
        +         'Generate all three &#8594;'
        +       '</button>'
        +     '</div>'
        +     '<div class="tp-egs">'
        +       '<span class="tp-egs-label">Or start with one of these &#8212; click to run it</span>'
        +       '<div class="tp-chips" data-pool="' + chip_pool + '"'
        +            ' data-count="' + str(EXAMPLE_COUNT) + '">' + chips + '</div>'
        +     '</div>'
        +     '<details class="tp-settings">'
        +       '<summary>Model options &#8212; fine as they are</summary>'
        +       '<div class="tp-opts">'
        +         '<div><div class="tp-opt-h">Text</div>'  + text_sel  + '</div>'
        +         '<div><div class="tp-opt-h">Image</div>' + image_sel + '</div>'
        +         '<div><div class="tp-opt-h">Audio</div>' + audio_sel + '</div>'
        +       '</div>'
        +     '</details>'
        +   '</div>'
        +   _ghost_lanes_html()
        +   '</section>'
        # Page 02/03/04: results injected by Gradio into #trio-results-area below
        + '</div>'
    )

STATIC_MAIN_HTML = _build_static_html()

# ── HTML helpers ──────────────────────────────────────────────────────────────
WAVE_BARS = "<i></i>" * 28

def _seg_class(val):
    if val is None: return "is-off"
    if val >= 0.5:  return "is-good"
    if val >= 0.3:  return "is-mid"
    return ""

def _seg_pct(val):
    return f"{int(val * 100)}%" if val is not None else "0%"

def _fmt(v, places=4):
    return "&#8212;" if v is None else f"{v:.{places}f}"


def _bar(v):
    """0-1 bar for a quality score; blank when the lane did not run."""
    if v is None:
        return '<span class="tp-rbar"><i style="--w:0%"></i></span>'
    return f'<span class="tp-rbar"><i style="--w:{max(0.0, min(1.0, v)) * 100:.1f}%"></i></span>'


def _method_block(scores=None):
    """The three per-output quality scores, then the combined Tricoherence score.

    Shows THIS generation's numbers rather than the study's leaderboard: the
    formulas are the ones that produced them, so the panel explains the score
    on screen instead of describing three approaches the app does not run.
    """
    sc = scores or {}
    qt, qi, qa = sc.get("q_text"), sc.get("q_image"), sc.get("q_audio")
    lr = sc.get("lr_score")

    if lr is None:
        verdict, vclass = "&#8212;", ""
    elif lr > 0:
        verdict, vclass = "Good", "is-good"
    else:
        verdict, vclass = "Not-Good", "is-bad"

    lr_txt = "&#8212;" if lr is None else f"{lr:+.3f}"

    return f"""
<details class="tp-method">
  <summary>How we measure this <em>500 prompts</em></summary>
  <div class="tp-method-b">
    <p class="tp-method-intro">Each output is scored against your prompt on its
    own terms, then the three are combined into one number. &#955; = {LAMBDA}
    penalises a pair that disagrees with itself, so a picture that is beautiful
    but off-prompt cannot coast on looks alone.</p>

    <div class="tp-appr">
      <div class="tp-appr-n">Q_text</div>
      <div class="tp-f">BERTScore(generated_text, prompt)</div>
      <div class="tp-r">{_bar(qt)}<b>{_fmt(qt)}</b></div>
    </div>

    <div class="tp-appr">
      <div class="tp-appr-n">Q_image</div>
      <div class="tp-f">avg(CLIP, Aesthetic) &#8722; {LAMBDA}&#183;var(CLIP, Aesthetic)</div>
      <div class="tp-r">{_bar(qi)}<b>{_fmt(qi)}</b></div>
    </div>

    <div class="tp-appr">
      <div class="tp-appr-n">Q_audio</div>
      <div class="tp-f">avg(semantic, wer_inv) &#8722; {LAMBDA}&#183;var(semantic, wer_inv)</div>
      <div class="tp-r">{_bar(qa)}<b>{_fmt(qa)}</b></div>
    </div>

    <div class="tp-appr is-best">
      <div class="tp-appr-n">Tricoherence score
        <span class="tp-badge">Likelihood ratio</span></div>
      <div class="tp-f">S(p) = &#931;&#7522; [ log P(f&#7522;|Good) &#8722; log P(f&#7522;|Not-Good) ]</div>
      <div class="tp-r"><span class="tp-verdict {vclass}">{verdict}</span><b>{lr_txt}</b></div>
    </div>

    <p class="tp-mnote">Gaussians for Good and Not-Good were fitted on 500
    prompts with leave-one-out cross-validation, split at a judge rating of
    3.5 (273 Good, 227 Not-Good). S(p) above zero means the three quality
    scores look more like a Good generation than a Not-Good one. A lane that
    failed is skipped rather than scored zero.</p>

    <div class="tp-honest">
      <span>Read this honestly</span>
      <b>r &#8776; 0.16 is weak</b> &#8212; about 57% pairwise accuracy, where 50%
      is chance. Tricoherence ranks outputs better than a coin flip, not reliably.
    </div>
  </div>
</details>"""

def _tricoherence_block(scores=None):
    sc = scores or {}
    qt = sc.get("q_text")
    qi = sc.get("q_image")
    qa = sc.get("q_audio")
    lr = sc.get("lr_score")

    if qt is None and qi is None and qa is None:
        return ""

    if lr is None:
        verdict, vclass = "&#8212;", ""
        read_text = "scoring in progress"
    elif lr > 0:
        verdict, vclass = "Good", "is-good"
        read_text = "all three outputs are tracking the same idea"
    else:
        verdict, vclass = "Not-Good", "is-bad"
        read_text = "outputs diverged &#8212; the models didn&#8217;t agree on direction"

    lr_txt = "&#8212;" if lr is None else f"{lr:+.3f}"

    return f"""
<div class="tp-coh" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:9px 2px;min-height:42px">
  <span class="tp-coh-k mono">Tricoherence</span>
  <span class="tp-verdict {vclass}">{verdict}</span>
  <span class="tp-coh-n">{lr_txt}</span>
  <span class="tp-coh-read">{read_text}</span>
</div>
{_method_block(scores)}"""

def _transcript_block(scores):
    """Whisper's transcript, under the player.

    compute_scores() already runs Whisper to build Q_audio and then discards
    the text. Showing it fills a column that was otherwise a player above 400px
    of nothing, and it lets the reader check for themselves that the narration
    says what the paragraph says.
    """
    t = (scores or {}).get("transcript") or ""
    t = t.strip()
    if not t:
        return ""
    if len(t) > 600:
        t = t[:600].rsplit(" ", 1)[0] + "…"
    safe = (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return ('<div class="tp-transcript">'
            '<div class="tp-transcript-h mono">Transcript</div>'
            f'<p class="tp-transcript-b prose">&#8220;{safe}&#8221;</p>'
            '</div>')


def _model_chip(models, kind):
    """Small mono model name in a card header, e.g. llama-3.1-8b."""
    name = (models or {}).get(kind, "")
    if not name:
        return ""
    # the catalogue keys read "Llama 3.1 8B (Cloudflare)"; the provider is
    # already implied by the app, so show just the model
    name = name.split("(")[0].strip()
    return f'<span class="tp-model mono">{name}</span>'


def _result_cards_html(text_out, text_err, text_t,
                       img_b64, img_err, img_t,
                       audio_b64, audio_err, audio_t,
                       scores, prompt="", models=None):
    models = models or {}
    cards = []

    if text_out:
        paras = [f'<p class="tp-text prose">{p.strip()}</p>'
                 for p in text_out.split("\n") if p.strip()]
        body = "\n".join(paras) or f'<p class="tp-text prose">{text_out}</p>'
        copy_data = json.dumps(text_out)
        cards.append(
            '<article class="tp-card is-done">'
            '<div class="tp-card-h">'
            '<i class="tp-out is-text" style="border:0;padding:0;background:none"><i></i></i>'
            f'<b>Text</b><span class="tp-tick mono">&#10003; {text_t:.1f}s</span>'
            + _model_chip(models, "text") +
            '<span class="tp-acts">'
            f'<button class="tp-btn" type="button" onclick="navigator.clipboard.writeText({copy_data})">&#8853; Copy</button>'
            '</span></div>'
            f'<div class="tp-body" data-tw>{body}</div>'
            '</article>'
        )
    elif text_err:
        cards.append(
            '<article class="tp-card is-failed">'
            '<div class="tp-card-h"><b>Text</b><span class="tp-eta">error</span></div>'
            f'<div class="tp-body"><div class="tp-fail"><p>{text_err}</p></div></div>'
            '</article>'
        )

    if img_b64:
        data_uri = "data:image/png;base64," + img_b64
        cards.append(
            '<article class="tp-card is-done">'
            '<div class="tp-card-h">'
            '<i class="tp-out is-image" style="border:0;padding:0;background:none"><i></i></i>'
            f'<b>Image</b><span class="tp-tick mono">&#10003; {img_t:.1f}s</span>'
            + _model_chip(models, "image") +
            '<span class="tp-acts">'
            f'<a class="tp-btn" href="{data_uri}" download="trio-image.png">&#8595; PNG</a>'
            '</span></div>'
            f'<div class="tp-body"><div class="tp-figure"><img src="{data_uri}" alt="Generated image"></div></div>'
            '</article>'
        )
    elif img_err:
        cards.append(
            '<article class="tp-card is-failed">'
            '<div class="tp-card-h"><b>Image</b><span class="tp-eta">error</span></div>'
            f'<div class="tp-body"><div class="tp-fail"><p>{img_err}</p></div></div>'
            '</article>'
        )

    if audio_b64:
        cards.append(
            '<article class="tp-card is-done">'
            '<div class="tp-card-h">'
            '<i class="tp-out is-audio" style="border:0;padding:0;background:none"><i></i></i>'
            f'<b>Audio</b><span class="tp-tick mono">&#10003; {audio_t:.1f}s</span>'
            + _model_chip(models, "audio") +
            '<span class="tp-acts">'
            f'<a class="tp-btn" href="data:audio/wav;base64,{audio_b64}" download="trio-audio.wav">&#8595; WAV</a>'
            '</span></div>'
            '<div class="tp-body">'
            f'<audio controls style="width:100%;border-radius:4px"><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>'
            + _transcript_block(scores) +
            '</div></article>'
        )
    elif audio_err:
        cards.append(
            '<article class="tp-card is-failed">'
            '<div class="tp-card-h"><b>Audio</b><span class="tp-eta">no audio</span></div>'
            '<div class="tp-body"><div class="tp-fail">'
            '<p><b>The voice model didn&#8217;t answer.</b> It&#8217;s the flakiest of the three '
            'and this happens now and then. Your paragraph and image are unaffected.</p>'
            '<button class="tp-btn" type="button" onclick="trioRetryAudio()">Retry audio only</button>'
            '</div></div>'
            '</article>'
        )

    coh_html = ""
    if scores and not scores.get("error"):
        coh_html = _tricoherence_block(scores)
    elif scores and scores.get("error"):
        coh_html = f'<p style="margin-top:10px;font-size:12px;color:var(--ink-45)">Scoring: {scores["error"]}</p>'

    echo = ""
    if prompt:
        safe = (prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        # the form is hidden on the results page, so without this the page
        # never says what was actually asked for
        echo = f'<div class="tp-echo"><span class="tp-echo-k mono">Prompt</span>{safe}</div>'

    return echo + '<div class="tp-results">' + "".join(cards) + '</div>' + coh_html

def _generating_cards_html():
    return (
        '<div class="tp-results">'
        '<article class="tp-card">'
        '<div class="tp-lane lane-text"></div>'
        '<div class="tp-card-h"><b>Text</b><span class="tp-eta">writing&#8230;</span></div>'
        '<div class="tp-body"><div class="tp-skel"><i></i><i></i><i></i><i></i></div></div>'
        '</article>'
        '<article class="tp-card">'
        '<div class="tp-lane lane-img"></div>'
        '<div class="tp-card-h"><b>Image</b><span class="tp-eta">painting&#8230;</span></div>'
        '<div class="tp-body"><div class="tp-figure is-loading"><span>generating</span></div></div>'
        '</article>'
        '<article class="tp-card">'
        '<div class="tp-lane lane-audio"></div>'
        '<div class="tp-card-h"><b>Audio</b><span class="tp-eta">waiting on the paragraph&#8230;</span></div>'
        '<div class="tp-body"><div class="tp-wave is-live" aria-hidden="true">' + WAVE_BARS + '</div></div>'
        '</article>'
        '</div>'
    )

def _wrap_results(inner):
    """Wrap results in a container that enables container queries for the 3-col grid.

    The `tp tp-page-root` classes are load-bearing, not decoration. Gradio puts
    #trio-results-area in its own block, a SIBLING of the static HTML - so the
    results are not descendants of .tp, where --ink / --card / --rule-str and
    the rest of the palette are declared. Without them every var() in here
    resolved to nothing: cards lost their panel, border and offset shadow, and
    the coherence meter rendered blank. tp-page-root re-applies the palette
    while suppressing .tp's own border/padding, so it does not draw a second
    panel around the results.
    """
    return ('<div class="tp tp-page-root" style="container-type:inline-size">'
            + inner + '</div>')

# ── Generation helpers ────────────────────────────────────────────────────────
TEXT_INSTRUCTION = "Describe this scene in vivid detail in 3-4 sentences: {p}"
TEXT_MAX_TOKENS  = 200

def _trim(text):
    text = (text or "").strip()
    m = list(re.finditer(r"[.!?][\"')\]]*(?=\s|$)", text))
    return text[:m[-1].end()].strip() if m else text

def _tts_text(text):
    t = _trim(text)
    return t if len(t) <= 900 else _trim(t[:900]) or t[:900]

def _pcm_to_wav(pcm_bytes, sr=24000, ch=1, bits=16):
    br = sr * ch * bits // 8
    ba = ch * bits // 8
    hdr = (b"RIFF" + struct.pack("<I", 36+len(pcm_bytes)) + b"WAVE"
           + b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, br, ba, bits)
           + b"data" + struct.pack("<I", len(pcm_bytes)))
    return hdr + pcm_bytes

def _groq():
    if not GROQ_KEY: return None
    from openai import OpenAI
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)

def _gemini():
    if not GEMINI_KEY: return None
    from google import genai
    return genai.Client(api_key=GEMINI_KEY)

def gen_text(prompt, info):
    prov, mid = info["provider"], info["id"]
    t0 = time.time()
    try:
        if prov == "cloudflare":
            r = requests.post(f"{CF_BASE_URL}/{mid}", headers=CF_HEADERS,
                json={"messages":[{"role":"user","content":TEXT_INSTRUCTION.format(p=prompt)}],
                      "max_tokens":TEXT_MAX_TOKENS}, timeout=25)
            if r.status_code == 200:
                return _trim(r.json().get("result",{}).get("response","")), None, time.time()-t0
            return None, f"Error {r.status_code}: {r.text[:120]}", 0
        if prov == "groq":
            c = _groq()
            if not c: return None, "GROQ_KEY not set in Secrets", 0
            resp = c.chat.completions.create(model=mid,
                messages=[{"role":"user","content":TEXT_INSTRUCTION.format(p=prompt)}],
                max_tokens=TEXT_MAX_TOKENS)
            return _trim(resp.choices[0].message.content), None, time.time()-t0
        if prov == "gemini":
            c = _gemini()
            if not c: return None, "GEMINI_API_KEY not set in Secrets", 0
            resp = c.models.generate_content(model=mid, contents=TEXT_INSTRUCTION.format(p=prompt))
            return _trim(resp.text), None, time.time()-t0
        return None, f"Unknown provider: {prov}", 0
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", 0

def gen_image(prompt, info):
    prov, mid = info["provider"], info["id"]
    t0 = time.time()
    try:
        if prov == "cloudflare":
            r = requests.post(f"{CF_BASE_URL}/{mid}", headers=CF_HEADERS,
                json={"prompt":prompt}, timeout=50)
            if r.status_code == 200:
                try:
                    b64 = r.json().get("result",{}).get("image","")
                    if b64: return base64.b64decode(b64), None, time.time()-t0
                except Exception:
                    pass
                if r.content: return r.content, None, time.time()-t0
            return None, f"Error {r.status_code}: {r.text[:120]}", 0
        if prov == "gemini":
            c = _gemini()
            if not c: return None, "GEMINI_API_KEY not set in Secrets", 0
            resp = c.models.generate_content(model=mid, contents=prompt)
            for part in resp.candidates[0].content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    return inline.data, None, time.time()-t0
            return None, "No image data in response", 0
        return None, f"Unknown provider: {prov}", 0
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", 0

def gen_audio(text, info):
    prov, mid = info["provider"], info["id"]
    clean = _tts_text(text)
    t0 = time.time()
    try:
        if prov == "cloudflare":
            payload = {"text":clean} if "deepgram" in mid else {"prompt":clean}
            r = requests.post(f"{CF_BASE_URL}/{mid}", headers=CF_HEADERS, json=payload, timeout=25)
            if r.status_code == 200:
                try:
                    b64 = r.json().get("result",{}).get("audio","")
                    if b64: return base64.b64decode(b64), None, time.time()-t0
                except Exception:
                    pass
                if r.content: return r.content, None, time.time()-t0
            return None, f"Error {r.status_code}: {r.text[:120]}", 0
        if prov == "gemini":
            c = _gemini()
            if not c: return None, "GEMINI_API_KEY not set in Secrets", 0
            from google.genai import types
            resp = c.models.generate_content(model=mid, contents=clean,
                config=types.GenerateContentConfig(response_modalities=["AUDIO"]))
            for part in resp.candidates[0].content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    return _pcm_to_wav(inline.data), None, time.time()-t0
            return None, "No audio data in response", 0
        return None, f"Unknown provider: {prov}", 0
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", 0

# ── Scoring ───────────────────────────────────────────────────────────────────
_qm = None

def _load_qm():
    global _qm
    if _qm: return _qm
    from transformers import CLIPModel, CLIPProcessor, pipeline as hfp
    import whisper
    _qm = {
        "clip":      CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(),
        "clip_proc": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        "aesthetic": hfp("image-classification", model="cafeai/cafe_aesthetic"),
        "whisper":   whisper.load_model("base"),
    }
    return _qm

def _qpair(a, b):
    avg = (a+b)/2; var = ((a-avg)**2+(b-avg)**2)/2
    return float(min(max(avg-LAMBDA*var, 0), 1))

def _lr(f):
    """S(p) = sum over available features of log P(f|Good) - log P(f|Not-Good).

    A missing feature is SKIPPED, not scored as 0.0. The fitted sigmas are tiny
    (sd_good[0] = 0.0088), so substituting zero puts the value ~95 standard
    deviations from the mean: a failed text lane used to give S = -831, and a
    failed image gave +8.52 - i.e. losing an output made the result score as
    strongly "Good". A lane that did not run is simply no evidence either way.
    """
    p = LR_PARAMS
    s = 0.0
    for i in range(3):
        if f[i] is None:
            continue
        s += (((f[i]-p["mu_bad"][i])**2/(2*p["sd_bad"][i]**2))
              - ((f[i]-p["mu_good"][i])**2/(2*p["sd_good"][i]**2))
              + math.log(p["sd_bad"][i]/p["sd_good"][i]))
    return float(s)

def compute_scores(prompt, text_out, img_bytes, audio_bytes):
    try:
        import torch, numpy as np, soundfile as sf
        from scipy import signal as sc_sig
        from bert_score import score as bscore
        from jiwer import wer as jwer
        M = _load_qm()
        def bs_f1(c, r):
            if not (c or "").strip(): return 0.0
            _, _, f1 = bscore([c], [r], lang="en", verbose=False)
            return float(f1.mean())
        q_text = bs_f1(text_out, prompt) if text_out else None
        q_image = None
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ti = M["clip_proc"](text=[prompt], return_tensors="pt",
                                padding=True, truncation=True, max_length=77)
            ii = M["clip_proc"](images=img, return_tensors="pt")
            with torch.no_grad():
                out = M["clip"](input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                                pixel_values=ii["pixel_values"])
            te = out.text_embeds.squeeze().numpy()
            ie = out.image_embeds.squeeze().numpy()
            clip_s = float(np.clip(np.dot(te,ie)/(np.linalg.norm(te)*np.linalg.norm(ie)), 0, 1))
            aes = next((r["score"] for r in M["aesthetic"](img) if r["label"]=="aesthetic"), 0.0)
            q_image = _qpair(clip_s, float(aes))
        q_audio = None
        transcript = ""
        if audio_bytes:
            wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != 16000: wav = sc_sig.resample(wav, int(len(wav)*16000/sr)).astype("float32")
            transcript = M["whisper"].transcribe(np.ascontiguousarray(wav))["text"].strip()
            sem  = bs_f1(transcript, prompt)
            wer_inv = 1.0 - float(np.clip(jwer(_tts_text(text_out).lower(), (transcript or "").lower()), 0, 1))
            q_audio = _qpair(sem, wer_inv)
        valid = [q for q in [q_text, q_image, q_audio] if q is not None]
        lr = _lr([q_text, q_image, q_audio]) if len(valid) >= 2 else None
        return {"q_text": q_text, "q_image": q_image, "q_audio": q_audio,
                "lr_score": lr, "transcript": transcript}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# ── Main generate function ────────────────────────────────────────────────────
# Holds the text and image of recent generations so "Retry audio only" can
# call the voice model alone. Without it the button had to re-run all three
# lanes - roughly triple the neurons, and the image changed underneath a user
# who was happy with it. Capped; this is a convenience, not storage.
_LAST_RUNS = OrderedDict()
_LAST_RUNS_MAX = 20


def _remember_run(token, payload):
    _LAST_RUNS[token] = payload
    while len(_LAST_RUNS) > _LAST_RUNS_MAX:
        _LAST_RUNS.popitem(last=False)


def generate(data_json):
    """Takes JSON from hidden textbox, yields results HTML only (form stays static)."""
    try:
        data = json.loads(data_json or "{}")
    except Exception:
        data = {}

    prompt      = (data.get("prompt") or "").strip()
    text_model  = data.get("text_model")  or list(TEXT_MODELS.keys())[0]
    image_model = data.get("image_model") or list(IMAGE_MODELS.keys())[0]
    audio_model = data.get("audio_model") or list(AUDIO_MODELS.keys())[0]
    do_scoring  = bool(data.get("scoring", False))

    if text_model  not in TEXT_MODELS:  text_model  = list(TEXT_MODELS.keys())[0]
    if image_model not in IMAGE_MODELS: image_model = list(IMAGE_MODELS.keys())[0]
    if audio_model not in AUDIO_MODELS: audio_model = list(AUDIO_MODELS.keys())[0]

    # ---- audio-only retry: reuse the cached paragraph and picture ----------
    if data.get("retry") == "audio":
        cached = _LAST_RUNS.get(data.get("token") or "")
        if not cached:
            yield _wrap_results(
                '<p style="font-size:12.5px;color:var(--ink-45)">That result has '
                'expired &#8212; generate again to retry its audio.</p>')
            return

        prompt = cached["prompt"]
        yield _wrap_results(_result_cards_html(
            cached["text_out"], None, cached["text_t"],
            cached["img_b64"], None, cached["img_t"],
            None, None, 0.0,          # audio lane shows as pending
            None, prompt, cached["models"]))

        t0 = time.time()
        audio_bytes, audio_err, _ = gen_audio(cached["text_out"], AUDIO_MODELS[audio_model])
        audio_t = time.time() - t0
        audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        models = dict(cached["models"]); models["audio"] = audio_model
        done = 2 + (1 if audio_b64 else 0)
        meta = ('<span id="trio-gen-meta" style="display:none"'
                f' data-done="{done}" data-total="3" data-time="{audio_t:.1f}"'
                f' data-token="{data.get("token")}"></span>')
        yield _wrap_results(_result_cards_html(
            cached["text_out"], None, cached["text_t"],
            cached["img_b64"], None, cached["img_t"],
            audio_b64, audio_err, audio_t,
            None, prompt, models) + meta)
        return

    if not prompt:
        yield ""
        return

    yield _wrap_results(_generating_cards_html())

    from concurrent.futures import ThreadPoolExecutor
    ti = TEXT_MODELS[text_model]
    ii = IMAGE_MODELS[image_model]
    ai = AUDIO_MODELS[audio_model]

    with ThreadPoolExecutor(max_workers=3) as pool:
        ft = pool.submit(gen_text,  prompt, ti)
        fi = pool.submit(gen_image, prompt, ii)
        text_out, text_err, text_t = ft.result()
        fa = pool.submit(gen_audio, text_out or prompt, ai)
        img_bytes,   img_err,   img_t   = fi.result()
        audio_bytes, audio_err, audio_t = fa.result()

    img_b64   = base64.b64encode(img_bytes).decode()   if img_bytes   else None
    audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None

    # Compute state metadata for JS state machine
    done_count = sum(1 for x in [text_out, img_b64, audio_b64] if x is not None)
    total_t    = max(text_t or 0, img_t or 0, audio_t or 0)
    # keep the finished text and image so the audio lane alone can be retried
    _token = _tok.token_urlsafe(8)
    _remember_run(_token, {
        "prompt": prompt, "text_out": text_out, "text_t": text_t or 0.0,
        "img_b64": img_b64, "img_t": img_t or 0.0,
        "models": {"text": text_model, "image": image_model, "audio": audio_model},
    })
    gen_meta   = (
        '<span id="trio-gen-meta" style="display:none"'
        f' data-done="{done_count}" data-total="3" data-time="{total_t:.1f}"'
        f' data-token="{_token}"></span>'
    )

    footer = (
        '<div class="tp-foot mono">'
        '<span>Runs on shared free quota</span>'
        '<span>&#183;</span>'
        '<span>Nothing stored after you close the tab</span>'
        '</div>'
    )

    # Yield results immediately so the user sees text/image/audio right away
    _models = {"text": text_model, "image": image_model, "audio": audio_model}
    results_html = _result_cards_html(
        text_out, text_err, text_t,
        img_b64,  img_err,  img_t,
        audio_b64, audio_err, audio_t,
        None, prompt, _models,
    )
    yield _wrap_results(results_html + gen_meta + footer)

    # Scoring runs when text and image both succeeded (audio optional)
    if do_scoring and text_out and img_bytes:
        try:
            scores = compute_scores(prompt, text_out, img_bytes, audio_bytes)
        except Exception as e:
            scores = {"error": f"{type(e).__name__}: {e}"}
        results_scored = _result_cards_html(
            text_out, text_err, text_t,
            img_b64,  img_err,  img_t,
            audio_b64, audio_err, audio_t,
            scores, prompt, _models,
        )
        yield _wrap_results(results_scored + gen_meta + footer)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Trio") as demo:
    # Entire visible UI — the Trio design — rendered as static custom HTML
    gr.HTML(STATIC_MAIN_HTML)

    # Bridge components: visible=True so they stay in the DOM; GRADIO_CSS moves them off-screen
    data_box = gr.Textbox(elem_id="trio-data-hidden", show_label=False)
    gen_btn  = gr.Button("go", elem_id="trio-btn-hidden")

    # Results area — only this updates during / after generation
    results = gr.HTML(elem_id="trio-results-area")

    gen_btn.click(fn=generate, inputs=[data_box], outputs=[results])

# css and js go in launch() in Gradio 6
demo.launch(css=GRADIO_CSS, js=TRIO_JS)
