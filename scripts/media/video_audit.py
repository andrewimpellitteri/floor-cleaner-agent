#!/usr/bin/env python3
"""Native-video analysis via OpenRouter (Gemini and other video-input models).

Frame-by-frame stills lose exactly what matters for a wash-bay pass: whether
the operator is walking, whether the jet is actually on, and how the wand angle
changes through a stroke. A model that ingests the video natively sees the
motion, so this asks one.

The same blind protocol as vision_audit.py applies: do not put the expected
answer in the prompt, and prefer two independent reads over one confident one.

    python scripts/media/video_audit.py clip.mp4 --question "..." --n 2
"""
from __future__ import annotations
import argparse, base64, json, os, pathlib, sys, urllib.error, urllib.request

API = "https://openrouter.ai/api/v1/chat/completions"


def call(fmt: str, b64: str, question: str, model: str, key: str, timeout: int) -> str:
    url = f"data:video/mp4;base64,{b64}"
    if fmt == "video_url":
        part = {"type": "video_url", "video_url": {"url": url}}
    elif fmt == "input_video":
        part = {"type": "input_video", "input_video": {"data": url}}
    else:  # file
        part = {"type": "file", "file": {"filename": "clip.mp4", "file_data": url}}
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": question}, part]}]}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/floorclean", "X-Title": "floorclean"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(json.dumps(d["error"])[:500])
    return d["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    p = pathlib.Path(a.video)
    if not p.exists():
        sys.exit(f"missing: {p}")
    b64 = base64.b64encode(p.read_bytes()).decode()
    print(f"# {p.name}  {p.stat().st_size/1e6:.2f} MB source  ->  {len(b64)/1e6:.2f} MB base64")

    for i in range(a.n):
        if a.n > 1:
            print(f"\n{'='*70}\nINDEPENDENT READ {i+1} of {a.n}  ({a.model})\n{'='*70}")
        last = None
        for fmt in ("video_url", "input_video", "file"):
            try:
                print(call(fmt, b64, a.question, a.model, key, a.timeout))
                break
            except Exception as e:
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    detail = e.read()[:300].decode("utf8", "replace")
                last = f"[{fmt}] {e} {detail}"
        else:
            print(f"ALL FORMATS FAILED. last: {last}")


if __name__ == "__main__":
    main()
