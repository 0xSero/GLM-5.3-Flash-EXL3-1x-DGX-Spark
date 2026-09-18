# QUARANTINE — E9 "grouped" run measured the baseline, not the grouped path

**Status: the E9 numbers in this directory are NOT a measurement of the K2/K3
grouped-fat-expert path. They are a re-measurement of the pre-E5 baseline
recipe. The E9 conclusion ("grouped path regresses") is void — no grouped code
was ever loaded.**

Written 2026-09-18T17:12Z, after the validation session found the
misconfiguration during a battery pass on the mosaic server.

## What E9 intended

Measure the v3 overlay's K2/K3 grouped-fat-expert kernels against the shipped
v2 image: bind-mount the overlay copy of `exl3.py` over the container's copy,
set `EXL3_FAT_GROUPED=1` and `EXL3_FUSED_TEMP_ROWS=4096`, and compare prefill.

## What E9 actually ran

`launch-k2-mtp-b12x-e9-grouped.sh:47-62` — the `docker run` block mounts
`MODEL`, `PROFILES`, and `PLUGIN` only. **There is no bind-mount for
`exl3.py`**, although the launcher's own header (lines 9-11) still documents one
and the env vars are still exported (lines 58-59). The container therefore ran
the image's own module.

Verified against the image itself on 2026-09-18T17:11Z:

| Check | Result |
|---|---|
| `docker image inspect glm53-b12x-exl3:jovian-3aada677-r3` id | `sha256:afb74c79…afffd5` — **matches `IMG_ID_EXPECT`**, so the guard passed |
| `grep -c EXL3_FAT_GROUPED` in image `exl3.py` | **0** |
| `grep -c EXL3_FUSED_TEMP_ROWS` in image `exl3.py` | **0** |
| `TEMP_ROWS_FUSED` in image `exl3.py` | `= 128`, hardcoded at line 64 |
| grouped-kernel markers (`dq_dispatch`, `fm_packed_words`) | **0** |

Both env vars are read nowhere in that module, so they were inert. The run
executed with the pre-E5 row cap (128), i.e. the fat-expert fallback was still
in place — the exact configuration E5/E6/imaging v2 exists to remove.

## Why the guard did not catch it

`IMG_ID_EXPECT` asserted the **base image** id (`launch-k2-mtp-b12x-e9-grouped.sh:16,28-29`).
The run deliberately replaced code by mount rather than by image, so the image
id was *supposed* to stay the base id — the guard could not distinguish
"overlay mounted as intended" from "overlay mount dropped". A guard that passes
under both intent and its violation is not a guard.

## What this invalidates

`RUN.txt` in this directory records prefill 434.5 tok/s (4k) and 463.3 tok/s
(131k), decode 18.5-20.0, acceptance 0.970-0.996. Those numbers are correct
measurements **of the baseline recipe** — consistent with the E6 baseline
(520.1 prefill at 131k on the v2 image; this run differs in runtime config,
which is expected) — and say nothing about the grouped kernels.

Therefore: **the v3 K2/K3 grouped-fat-expert kernels remain performance-unmeasured.**
v3 is verified functional only (it compiles and serves), never benchmarked in a
valid A/B.

## Corrected test (E9b) — the guard must fail closed

1. Assert the *mounted overlay*, not the image: `grep -q` a sentinel string in
   the host overlay file before launch (as `run-m1-mul1-mtp.sh:38` now does),
   **and** grep the overlay's env-var names inside the running container
   (`docker exec … grep -c EXL3_FAT_GROUPED <module>`) before timing anything.
2. Choose a row cap that actually exercises the grouped path. At the E6 setting
   (TR=4096) the over-cap tail is ~199 mean routed rows per expert, so nearly
   every expert is already absorbed by the fused kernel and a corrected E9b at
   TR=4096 is close to a no-op by construction. The honest test is **TR=256**,
   which forces essentially all experts down the grouped path and measures the
   thing the kernels exist to do.

Nothing here is deleted. The receipt is preserved as the record of an invalid
experiment, which is itself a result.