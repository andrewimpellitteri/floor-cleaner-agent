#!/usr/bin/env python3
"""Second-opinion vision audit via the OpenAI API.

Why this exists: the in-house Haiku subagent pass produced confident, precise,
WRONG measurements three separate times on this project (a 3-inch pond depth
that was a perspective artifact; "no wand in frame" on a frame containing a
full side-on lance; "no tape" on a frame containing a tape). A single vision
reader is not trustworthy for anything metric.

So this asks a second, independent model, and the protocol matters:

  * BLIND. The prompt must never contain the answer we expect, our prior
    reading, or the other model's reading. A vision model asked to confirm will
    confirm. This script therefore takes the question only, and the caller is
    responsible for not leaking the hypothesis into it.
  * TWO READS. --n 2 asks the same question twice in separate calls. Two
    independent reads that disagree is itself the finding: it means the image
    does not support the measurement, whatever either read says.

    python scripts/media/vision_audit.py IMG.jpg --question "..." --n 2
    python scripts/media/vision_audit.py A.jpg B.jpg --question "..."   # both in one call
"""
from __future__ import annotations
import argparse, base64, json, mimetypes, os, pathlib, sys, urllib.request

API = "https://api.openai.com/v1/responses"


def data_url(p: pathlib.Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def ask(images: list[pathlib.Path], question: str, model: str, effort: str,
        key: str, timeout: int) -> str:
    content = [{"type": "input_text", "text": question}]
    for p in images:
        content.append({"type": "input_image", "image_url": data_url(p)})
    body = {"model": model, "input": [{"role": "user", "content": content}]}
    if effort:
        body["reasoning"] = {"effort": effort}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    out = []
    for item in d.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                out.append(c["text"])
    return "\n".join(out).strip() or json.dumps(d)[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--n", type=int, default=1, help="independent repeat reads")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set")
    imgs = [pathlib.Path(i) for i in a.images]
    for p in imgs:
        if not p.exists():
            sys.exit(f"missing: {p}")

    for i in range(a.n):
        if a.n > 1:
            print(f"\n{'='*70}\nINDEPENDENT READ {i+1} of {a.n}  ({a.model})\n{'='*70}")
        try:
            print(ask(imgs, a.question, a.model, a.effort, key, a.timeout))
        except Exception as e:
            body = getattr(e, "read", lambda: b"")()
            print(f"CALL FAILED: {e}\n{body[:800]}")


if __name__ == "__main__":
    main()
