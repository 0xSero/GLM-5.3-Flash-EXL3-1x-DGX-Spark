#!/usr/bin/env python3
"""Decode rate vs context length, with MTP acceptance, on the served model.

The recorded 18.7-19.2 tok/s for this recipe came from long-prompt cells; a
20-token greedy probe measures ~11-12. This script isolates context length as
the variable: same model, same 128 greedy tokens, three prompt lengths, each
with a unique opening (so no prefix cache), reporting TTFT-split decode rate and
the server's speculative-decoding counters across each cell.

Usage: python3 decode_vs_context.py <base_url> <out_dir> [model_name]
"""
import json, os, secrets, sys, time, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080").rstrip("/")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/decode-ctx"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "glm-5.3-flash"
os.makedirs(OUT, exist_ok=True)
FILLER = ("The archive records a cache entry. A reader verifies the revision before reuse. "
          "Workers preserve unrelated records.\n")
CELLS = [20, 4000, 16000]           # approximate prompt tokens
PROMPT = ("Write a detailed technical explanation of how a transformer mixture-of-experts "
          "layer routes tokens, covering the router, the top-k selection, capacity factors, "
          "and load balancing.")


def w(name, value):
    p = os.path.join(OUT, name + ".json")
    with open(p + ".tmp", "w") as f:
        json.dump(value, f, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def counters():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
            txt = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        for key in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(key + "{"):
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def cell(approx_tokens):
    nonce = secrets.token_hex(8)
    pad = FILLER * max(0, (approx_tokens - 20) // 20)
    prompt = ("Probe %s. %s" % (nonce, PROMPT)) + ("\n" + pad if pad else "")
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 128,
                       "temperature": 0, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    before = counters()
    t0, ttft, usage, chunks = time.time(), None, None, 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("choices"):
                    ch = ev["choices"][0]
                    if (ch.get("text") or ch.get("delta", {}).get("content")) and ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    after = counters()
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or chunks
    dd = (after.get("vllm:spec_decode_num_drafts_total", 0)
          - before.get("vllm:spec_decode_num_drafts_total", 0))
    dt = (after.get("vllm:spec_decode_num_draft_tokens_total", 0)
          - before.get("vllm:spec_decode_num_draft_tokens_total", 0))
    da = (after.get("vllm:spec_decode_num_accepted_tokens_total", 0)
          - before.get("vllm:spec_decode_num_accepted_tokens_total", 0))
    row = {"target_prompt_tokens": approx_tokens, "prompt_tokens": pt, "completion_tokens": ct,
           "ttft": round(ttft, 3) if ttft else None, "wall": round(wall, 3), "error": err,
           "drafts": dd, "draft_tokens": dt, "accepted": da}
    if ttft and ct and wall > ttft:
        row["decode_tok_s"] = round((ct - 1) / (wall - ttft), 2)
    if dt:
        row["acceptance_pct"] = round(100.0 * da / dt, 1)
    print(json.dumps(row), flush=True)
    return row


def main():
    out = {"base_url": BASE, "model": MODEL,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    w("decode_vs_context", out)
    for c in CELLS:
        out["cells"].append(cell(c))
        w("decode_vs_context", out)
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    w("decode_vs_context", out)
    print("DECODE_VS_CONTEXT_DONE", flush=True)


if __name__ == "__main__":
    main()