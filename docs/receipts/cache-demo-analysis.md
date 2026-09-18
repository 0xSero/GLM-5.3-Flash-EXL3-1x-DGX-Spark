# Cache demonstration — what the server actually did

Date: 2026-09-18. Server: SGLang `exl3-plain` (`glm53-mosaic-test`), 262,144 context,
`--max-prefill-tokens 256 --chunked-prefill-size 256`, `--max-running-requests 1`,
`--mem-fraction-static 0.95` (KV pool 595,200 tokens), radix cache at default (enabled).

## Intent

Show that RadixAttention serves repeated prefixes from cache, and that the streaming
sweep's unique per-rung nonce therefore defeats it. Four requests of a ~14 k-token body:

| Arm | Opening | Expected TTFT |
|---|---|---|
| hit_1 | nonce A | full cold prefill |
| hit_2 | nonce A (identical) | collapsed (cache hit) |
| miss_1 | nonce B | full cold prefill |
| miss_2 | nonce C | full cold prefill |

## Measured

| Arm | prompt tokens | TTFT s | prefill tok/s |
|---|---|---|---|
| hit_1 | 14,043 | 28.847 | 486.8 |
| hit_2 | 14,043 | **57.114** | 245.9 |
| miss_1 | 14,044 | 56.940 | 246.6 |
| miss_2 | 14,042 | 57.133 | 245.8 |

The identical repeat was **2× slower**, not faster. Verdict from the script: `inconclusive`.

## What the server's own log says

Parsed from `Prefill batch` lines (`parse_prefill_log.py`), grouping steps into requests by
a ≥3 s gap:

```
521 prefill steps in 2 requests
start     end        steps   new_tok    cached    secs
17:28:00  17:28:41      79     20224         0      41   <- tail of the sweep's last rung
17:28:58  17:32:48     442    113088         0     230   <- the four demo requests
```

Every demo step reports `#cached-token: 0`, and the four requests account for **113,088 new
tokens against 56,172 tokens of prompt text** — i.e. requests 2–4 each had their prompt
prefilled twice, at the normal per-step rate (~256 tokens per ~0.52 s ≈ 490 tok/s), which is
why their wall time doubled instead of collapsing.

## Why

The KV pool holds 595,200 tokens, and the streaming sweep that ran immediately before the
demo had inserted the whole ladder into the radix cache:

990 + 3,792 + 14,950 + 59,631 + 119,213 + 181,870 + 236,510 = **616,956 tokens > 595,200**

The pool was over-subscribed by its own sweep. A repeated 14 k prompt arriving into that
state was re-prefilled rather than served, and under this chunked-prefill configuration it
paid the prompt twice. This is a property of a *saturated, throttled* server, not a general
statement about RadixAttention — the vision probes that followed it demonstrate reuse
working normally on the same server:

```
17:33:17  1 step  64 new   384 cached    <- 512x512 image prompt, repeat arm
17:34:21  1 step  64 new  5504 cached    <- 2048x2048 image prompt, repeat arm
17:35:33  1 step  64 new  7936 cached    <- 4096x4096 image prompt, repeat arm
17:35:57  1 step  64 new  7936 cached
17:39:13  1 step  64 new  7936 cached    <- thinking arm, same image
```

One step, 64 new tokens, 7,936 tokens served from cache: prefix reuse is real and large when
the pool has room for the entry.

## What this changes

1. **Nothing about the streaming sweep.** Its own steps logged `#cached-token: 0`, its TTFT
   is linear in prompt length at r² = 0.9999, and each rung paid a full cold prefill. The
   measurement stands.
2. **The superseded battery ladder remains explained.** Its rungs were prefix-extensions of
   one another (`FILLER × N`), so rung *N* matched rung *N−1*'s cached entry as a prefix and
   prefilled only the delta — which is why its 260 k rung came out faster than its 131 k rung.
3. **A practical note for anyone benchmarking this recipe.** A ladder sweep on this
   configuration contaminates itself twice over: shared prefixes make later rungs look fast
   while saturating the pool, and a saturated pool then makes repeated prompts look slow. Run
   sweeps with a unique opening per request, and either drain the cache between sweeps or
   accept that the pool carries ~0.6 M tokens of prior work.