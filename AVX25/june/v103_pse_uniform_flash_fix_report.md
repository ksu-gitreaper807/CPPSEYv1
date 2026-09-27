# PSE Uniform-Flash False-Negative Fix — Report (v103)

**Source file:** `AVX25/june/loadv102_DD.cpp` (was `loadv102_DD.cpp`, now carries the v103 fixes in comments)
**Date:** 2026-09-18
**Status:** Fixed and verified against a 33-case synthetic matrix (all cases pass).

---

## 1. Problem

The detector missed **spatially uniform (full-screen) flashing** at every frequency 1–30 Hz, at every
tested fps (24/25/30/50/60). Baseline matrix (`test/results/baseline/summary.txt`): **0 alarms** for all
13 full-screen / saturated-red / partial-area / low-contrast positive cases. The only case that alarmed was
"anti-phase halves 4 Hz" (351 alarms, t=0.40 s), because there the left and right halves transition in
opposite directions within the same window.

### Root cause (confirmed by [D] trace)

The alarm gate requires `has_d2b && has_b2d` — **both** a dark→bright **and** a bright→dark combo in the
*same* analysis window. For a spatially uniform flash, every pixel's (min, max) frame pair points the same
direction: the direction of the **first transition inside the window**. So a window either contains a d2b
transition (all pixels vote d2b) or a b2d transition (all pixels vote b2d) — they never co-occur. Trace for
6 Hz full-screen @60 fps showed `Y_conc=100%` with `pair=FAIL:d2b` / `FAIL:b2d` alternating every window and
`oy=or=0` — the rate counter never saw a single onset. The opposing-pair gate was designed to reject
one-directional events (scene cuts, ramps) and, as a side effect, rejected 100 % of uniform flashes.

## 2. Fix (v103) — temporal direction alternation instead of in-window pairs

The flash *rate* is a temporal property. Instead of demanding both directions inside one window, the fix
tracks the **population direction** of each window (which direction dominates the combo hits) and counts a
flash **onset** when the population direction flips **FALL→RISE** — i.e., when a dark→bright transition
becomes dominant after a bright→dark one. One flip per flash cycle ⇒ onsets/s ≈ flash rate, which feeds the
existing 1-second onset counter (alarm at >2 onsets/s, unchanged).

The original `(d2b && b2d)` path is kept as an alternative (it still covers the anti-phase case, whose
behavior is bit-identical to baseline — verified: 351 alarms, t=0.40 s, unchanged).

Two additional defects surfaced while validating the fix and were corrected (see §3, items 4–5):

* **Window-boundary double-admission** — `events_in_window` used `<=`, admitting an onset exactly 1.000 s
  old. A 2 Hz flash has onsets at exactly 0.5 s spacing, so one onset always sat exactly on the boundary →
  3 counted events/s → false alarms (13 per 7-s run).
* **2.5 Hz ambiguity** — with onsets now being produced, onset *count* alone cannot separate 2.5 Hz
  (events at Δ=0/0.4/0.8 s → 3 in any 1-s window) from 3 Hz (Δ=0/0.333/0.667 → 3). A frequency gate on the
  inter-onset interval of the last two onsets is required.

## 3. Parameter changes

| # | Parameter / construct | Before | After | Reason |
|---|----------------------|--------|-------|--------|
| 1 | `PSEConfig::dir_majority_frac`, `PSEConfig16::dir_majority_frac` (new) | — | `0.55f` | Population direction is defined only when one direction holds ≥55 % of all directional combo hits. Incoherent motion (panning, noise) has d2b≈b2d → no direction → no flips → stays silent. |
| 2 | `PSEState::prev_dir_y/r`, `PSEState16::prev_dir_y/r` (new) | — | `int = 0`, persisted | Remembers the last non-zero population direction across mask-free windows (a window inside a single phase has no transitions) so a FALL→RISE flip is detected across those gaps. |
| 3 | `is_flashing_y/r` (both detectors) | `spatial && (has_d2b && has_b2d)` | `spatial && ((has_d2b && has_b2d) \|\| (flip && (has_d2b \|\| has_b2d)))` | One-directional uniform flashes now produce onsets via FALL→RISE flips; the original in-window pair gate is retained (anti-phase path unchanged). `flip` = population direction went FALL→RISE this window. |
| 4 | `FlashRateCounter::events_in_window` | `pts_now − ring[i] <= 1.0` | `pts_now − ring[i] < 1.0 − 1e-3` | Inclusive boundary double-admitted an onset exactly 1.0 s old (2 Hz false alarms); the 1 ms margin also absorbs double-precision rounding (0.999999999…). 1 ms ≪ 333 ms inter-onset at 3 Hz, so no true positive is lost. |
| 5 | Alarm frequency gate `PSE_MIN_ALARM_HZ` (new, both detectors) | — | `2.95` | Alarm additionally requires last-two-onsets inter-onset frequency ≥ 2.95 Hz. Onset count cannot tell 2.5 Hz (3 events/s) from 3 Hz (3 events/s). 2.0/2.5 Hz → hz_est 2.0/2.5 → rejected; 3.0000 Hz ≥ 2.95 → accepted (tolerance for float at exactly 3.0). |
| 6 | `[D]` / `[S]` diagnostic lines | `… pair=PASS|FAIL:d2b|FAIL:b2d|FAIL:both` | `… pair=… diry=RISE|FALL|- flipy=Y|N` | Observability: the two new decision stages are now visible in the trace (used to diagnose this defect). |

No existing threshold was moved: `PSE_FLASH_RATE_THRESHOLD=2` (>2 onsets in 1 s), the 1 s window,
`static_luma_thresh`, area thresholds, drift scaling, the 1.05499681027f per-pixel criterion, and the
P-function are all unchanged.

## 4. Measured results (harness: synthetic BGR0 square wave, 1280×720 unless noted)

Harness: `AVX25/test/` (fake FFmpeg/X11 headers + minimal implementation, scenario generator,
`run_eval.py`). Clips are 6 s (60 fps) unless noted. "First alarm" = time of first `[!!!]` line.

### Positive cases — before → after

| Case | fps | Before (baseline) | After (v103) | First alarm |
|------|-----|-------------------|--------------|-------------|
| Full-screen 3.0 Hz | 60 | 0 alarms (FN) | 1084 alarms | **t=1.00 s** (SLOW path; fast joins, 249 of them) |
| Full-screen 4.0 Hz | 60 | 0 alarms (FN) | 942 alarms | **t=0.90 s** |
| Full-screen 6.0 Hz | 60 | 0 alarms (FN) | 1381 alarms | **t=0.50 s** |
| Full-screen 10.0 Hz | 60 | 0 alarms (FN) | 1739 alarms | t=0.50 s |
| Full-screen 15.0 Hz | 60 | 0 alarms (FN) | 875 alarms | t=0.40 s |
| Full-screen 20.0 Hz | 60 | 0 alarms (FN) | 1760 alarms | t=0.30 s |
| Full-screen 30.0 Hz | 60 | 0 alarms (FN) | 1760 alarms | t=0.20 s |
| Saturated red 3.0 Hz | 60 | 0 alarms (FN) | 1084 alarms | t=1.00 s |
| Saturated red 6.0 Hz | 60 | 0 alarms (FN) | 1381 alarms | t=0.50 s |
| 40 %-area 4.0 Hz | 60 | 0 alarms (FN) | 942 alarms | t=0.90 s |
| 25 %-area 4.0 Hz (at area threshold) | 60 | 0 alarms (FN) | 942 alarms | t=0.90 s |
| 100×100 tiny flash 6.0 Hz | 60 | 0 alarms (FN) | 1381 alarms | t=0.50 s |
| Low-contrast 100/140, 6.0 Hz | 60 | 0 alarms (FN) | 1381 alarms | t=0.50 s |
| 3.0 Hz @ 24 fps | 24 | 0 alarms (FN) | 1204 alarms | t=1.00 s |
| 3.0 Hz @ 25 fps | 25 | 0 alarms (FN) | 886 alarms | t=1.00 s |
| 3.0 Hz @ 30 fps | 30 | 0 alarms (FN) | 1123 alarms | t=1.00 s |
| 3.0 Hz @ 50 fps | 50 | 0 alarms (FN) | 542 alarms | t=1.00 s |
| 6.0 Hz @ 24 fps | 24 | 0 alarms (FN) | 695 alarms | t=1.00 s |
| 6.0 Hz @ 30 fps | 30 | 0 alarms (FN) | 1306 alarms | t=0.90 s |
| 6.0 Hz @ 50 fps | 50 | 0 alarms (FN) | 1664 alarms | t=0.50 s |
| 6.0 Hz, 1080p native | 60 | 0 alarms (FN) | 901 alarms | t=0.50 s |
| Anti-phase halves 4 Hz (control) | 60 | 351 alarms, t=0.40 s | **351 alarms, t=0.40 s — unchanged** | t=0.40 s |

### Negative cases — all silent before and after (0 alarms)

Full-screen 1.0 Hz, 2.0 Hz, 2.5 Hz; static grey; static checkerboard; static saturated red; single scene
cut; gradual brightness ramp; panning high-contrast stripes; white noise; 300×300 blink 2 Hz.

**Result: 17/17 positive cases alarm, 16/16 negative cases silent, control case unchanged.**

## 5. Borderline cases (honest notes)

* **Effective threshold is ≈2.95 Hz, not a hard 3.0.** The gate constant is 2.95 for float tolerance at
  exactly 3.000 Hz. A 2.9 Hz flash is therefore *not* alarmed. The 2.5→3.0 Hz band is where the two cases
  are indistinguishable by onset count; the frequency gate is the only discriminator and it is what places
  the effective threshold. If a stricter hard-3.0 gate is wanted, `PSE_MIN_ALARM_HZ` can be raised to 2.99
  (3.0000 Hz still passes; 2.999 Hz then also passes — float-limited).
* **3 Hz latency = 1.00 s.** Inherent to the rate gate: ≥3 onsets must fall inside a 1-s window, plus the
  slow detector's 16-frame (267 ms @60) warmup and flip phase alignment. Reducing it would require
  shortening `WINDOW_SECONDS`, which changes the "within 1 second" semantics the rate counter is built on.
* **3 Hz @ non-60 fps** still alarms (t=1.00 s) but with fewer alarm frames (e.g. 886 @25 fps vs 1084 @60)
  because the 20-frame/24-frame period doesn't divide the 16-frame window evenly; onset spacing jitters
  between ~0.32 and ~0.36 s, occasionally pushing the last onset just outside the 1 s window.
* **P display is not the alarm gate.** For full-screen uniform flashes the `[i]` line still prints
  P≈60 % (the P-function's area-trust weighting caps it); the hard ITU-style alarm fires from the
  spatial + rate + frequency gates independently of P. No P-function parameter was changed.
* **Synthetic stimulus only.** The harness drives ideal square waves (0↔255) through a minimal fake
  decode/scale/X11 path. Real footage (encoding artifacts, camera noise, partial occlusion, imperfect
  50/50 duty) is untested; the 55 % direction-majority constant is a prior, not a measured optimum.

## 6. Files changed

| File | Change |
|------|--------|
| `AVX25/june/loadv102_DD.cpp` | The fix: +169/−9 lines (items 1–6 above, both detectors + shared `FlashRateCounter`). |
| `AVX25/test/` (new) | Validation harness: `fakeff/` (12 minimal FFmpeg+X11 headers, `extern "C"`-guarded), `fakeff_impl.cpp` (fake device/decoder/box-average scaler/X11), `scenarios.cpp` (16-case BGR0 generator), `Makefile` (target `build/detector_harness`), `run_eval.py` (33-case matrix), `run_eval_baseline.py`, `results/` (v103 logs), `results/baseline/` (pre-fix 33-case logs + `summary.txt`). |

## 7. Recommendation

The change is **safe to adopt**: it removes the mechanism that made the pair gate blind to uniform
flashes, adds no new per-pixel cost (direction is derived from combo hit counts already computed), and the
33-case matrix — including the unchanged anti-phase control and all 16 negatives — passes.

Caveats to carry forward:

1. **Validate on real PSE-labeled video** before deployment — the 55 % majority fraction and the 2.95 Hz
   gate constant were chosen analytically from the synthetic ladder, not tuned against real content.
2. **Decide the 2.9–3.0 Hz band policy explicitly** (current behavior: silent below ≈2.95 Hz). If the
   product spec says "≥3 Hz must alarm, <3 Hz must not", the current constants satisfy it for clean
   square waves; real footage with jittered frame timing may blur that line.
3. If sub-1-second 3 Hz latency ever becomes a requirement, the lever is `WINDOW_SECONDS` (1.0 → 0.8),
   with the frequency gate still protecting 2.5 Hz; that is a deliberate semantic change and should be a
   separate, measured decision.
