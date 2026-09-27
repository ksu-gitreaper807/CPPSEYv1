#!/usr/bin/env python3
"""PSE detector sensitivity evaluation runner.

Runs the detector harness (build/detector_harness) over a matrix of
synthetic positive/negative cases and summarises:

  - alarm count and first-alarm time (detection latency)
  - graded P-function statistics (max P, #windows P>=45 / P>=70)
  - wall-clock throughput (perf proxy)

Usage:
  python3 run_eval.py [--case NAME] [--all] [--quiet]
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE / "build" / "detector_harness"
OUT = HERE / "results"

# (case, fps, hz, l1, l2, frames, size, label, expectation)
# expectation: "pos" = should alarm, "neg" = should not, "borderline"
CASES = [
    # ── Positives: full-screen luminance flashes (ITU Guideline 1 core) ─────
    ("flash_hz", 60, 3.0,  0, 255, 900,  "1280x720", "full-screen 3.0 Hz",    "pos"),
    ("flash_hz", 60, 4.0,  0, 255, 900,  "1280x720", "full-screen 4.0 Hz",    "pos"),
    ("flash_hz", 60, 6.0,  0, 255, 900,  "1280x720", "full-screen 6.0 Hz",    "pos"),
    ("flash_hz", 60, 10.0, 0, 255, 900,  "1280x720", "full-screen 10.0 Hz",   "pos"),
    ("flash_hz", 60, 15.0, 0, 255, 900,  "1280x720", "full-screen 15.0 Hz",   "pos"),
    ("flash_hz", 60, 20.0, 0, 255, 900,  "1280x720", "full-screen 20.0 Hz",   "pos"),
    ("flash_hz", 60, 30.0, 0, 255, 900,  "1280x720", "full-screen 30.0 Hz",   "pos"),
    ("flash_red", 60, 3.0, 0, 255, 900,  "1280x720", "saturated red 3.0 Hz",  "pos"),
    ("flash_red", 60, 6.0, 0, 255, 900,  "1280x720", "saturated red 6.0 Hz",  "pos"),
    ("flash_half_aph", 60, 4.0, 0, 255, 900, "1280x720", "anti-phase halves 4 Hz (pair-gate control)", "pos"),
    ("flash_part_40", 60, 4.0, 0, 255, 900, "1280x720", "40%-area flash 4 Hz", "pos"),
    ("flash_lowc", 60, 6.0, 100, 140, 900, "1280x720", "low-contrast 100/140 6 Hz", "borderline"),

    # ── FPS scaling (same 3 Hz and 6 Hz flashes) ───────────────────────────
    ("flash_hz", 24, 3.0, 0, 255, 720, "1280x720", "3.0 Hz @ 24 fps", "pos"),
    ("flash_hz", 25, 3.0, 0, 255, 750, "1280x720", "3.0 Hz @ 25 fps", "pos"),
    ("flash_hz", 30, 3.0, 0, 255, 900, "1280x720", "3.0 Hz @ 30 fps", "pos"),
    ("flash_hz", 50, 3.0, 0, 255, 1000, "1280x720", "3.0 Hz @ 50 fps", "pos"),
    ("flash_hz", 24, 6.0, 0, 255, 720, "1280x720", "6.0 Hz @ 24 fps", "pos"),
    ("flash_hz", 30, 6.0, 0, 255, 900, "1280x720", "6.0 Hz @ 30 fps", "pos"),
    ("flash_hz", 50, 6.0, 0, 255, 1000, "1280x720", "6.0 Hz @ 50 fps", "pos"),

    # ── Downsample path (1080p native → 720p analysis) ─────────────────────
    ("flash_hz", 60, 6.0, 0, 255, 600, "1920x1080", "6.0 Hz @ 60fps, 1080p native", "pos"),

    # ── Borderline / negative spatial ──────────────────────────────────────
    ("flash_part_25", 60, 4.0, 0, 255, 900, "1280x720", "25%-area flash 4 Hz (at threshold)", "borderline"),
    ("flash_small", 60, 6.0, 0, 255, 900, "1280x720", "100x100 flash 6 Hz (tiny)", "neg"),

    # ── Negatives: rate below ITU 3 Hz floor ───────────────────────────────
    ("flash_hz", 60, 1.0, 0, 255, 900, "1280x720", "full-screen 1.0 Hz", "neg"),
    ("flash_hz", 60, 2.0, 0, 255, 900, "1280x720", "full-screen 2.0 Hz", "neg"),
    ("flash_hz", 60, 2.5, 0, 255, 900, "1280x720", "full-screen 2.5 Hz", "neg"),

    # ── Negatives: non-flash content ───────────────────────────────────────
    ("static_gray", 60, 0, 0, 0, 900, "1280x720", "static grey", "neg"),
    ("static_checker", 60, 0, 0, 0, 900, "1280x720", "static checkerboard", "neg"),
    ("red_static", 60, 0, 0, 0, 900, "1280x720", "static saturated red", "neg"),
    ("scene_cut", 30, 0, 0, 0, 600, "1280x720", "single scene cut", "neg"),
    ("grad_ramp", 30, 0, 0, 0, 900, "1280x720", "gradual brightness ramp", "neg"),
    ("camera_pan", 30, 0, 0, 0, 900, "1280x720", "panning high-contrast stripes", "neg"),
    ("noise_white", 60, 0, 0, 0, 600, "1280x720", "white noise", "neg"),
    ("blink_small_2hz", 60, 2.0, 0, 0, 900, "1280x720", "300x300 blink 2 Hz", "neg"),
]

ALARM_RE = re.compile(r"\[!!!\](?:\s*\[SLOW\])?\s+PSE [^\n]*FLASH ALARM\s+t=([\d.]+)")
P_RE = re.compile(r"P=([\d.]+)%")
RUNTIME_RE = re.compile(r"Runtime: (\d+) ms")


def run_case(case, fps, hz, l1, l2, frames, size, label):
    env = dict(os.environ)
    env["PSE_SYNTH_CASE"] = case
    env["PSE_SYNTH_FPS"] = f"{fps:g}"
    env["PSE_SYNTH_FRAMES"] = str(frames)
    env["PSE_SYNTH_SIZE"] = size
    if hz:
        env["PSE_SYNTH_HZ"] = f"{hz:g}"
    env["PSE_SYNTH_L1"] = str(l1)
    env["PSE_SYNTH_L2"] = str(l2)
    env.pop("DISPLAY", None)

    t0 = time.time()
    p = subprocess.run([str(BIN), ":0.0", f"{fps:g}"],
                       capture_output=True, text=True, env=env, timeout=600)
    wall = time.time() - t0
    out = p.stdout

    alarms = ALARM_RE.findall(out)
    pvals = [float(x) for x in P_RE.findall(out)]
    m = RUNTIME_RE.search(out)
    runtime_ms = int(m.group(1)) if m else 0

    with open(OUT / f"{case}_{fps}fps_{hz:g}hz_{size.replace('x','p')}.log", "w") as f:
        f.write(f"# {label}\n# case={case} fps={fps} hz={hz} size={size} frames={frames}\n")
        f.write(out)
        if p.stderr:
            f.write("\n--- stderr ---\n")
            f.write(p.stderr)

    return {
        "label": label,
        "case": case,
        "fps": fps,
        "hz": hz or 0,
        "frames": frames,
        "size": size,
        "alarms": len(alarms),
        "first_alarm_t": float(alarms[0]) if alarms else None,
        "max_P": max(pvals) if pvals else 0.0,
        "n_P45": sum(1 for x in pvals if x >= 45.0),
        "n_P70": sum(1 for x in pvals if x >= 70.0),
        "n_P_lines": len(pvals),
        "runtime_ms": runtime_ms,
        "wall_ms": int(wall * 1000),
        "rc": p.returncode,
        "log": str(OUT / f"{case}_{fps}fps_{hz:g}hz_{size.replace('x','p')}.log"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", default=None,
                    help="run a single case by label substring")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not BIN.exists():
        sys.exit(f"binary not built: {BIN} (run make first)")
    OUT.mkdir(exist_ok=True)

    selected = [c for c in CASES
                if (not args.case) or any(k in c[7] for k in args.case)]
    if not selected:
        sys.exit("no matching cases")

    print(f"{'LABEL':<42} {'FPS':>4} {'HZ':>5} {'ALARMS':>7} {'1st(t)':>8} "
          f"{'maxP':>6} {'#P45':>5} {'#P70':>5} {'ms':>6}")
    print("-" * 100)
    rows = []
    for (case, fps, hz, l1, l2, frames, size, label, _) in selected:
        r = run_case(case, fps, hz, l1, l2, frames, size, label)
        first = f"{r['first_alarm_t']:.2f}" if r["first_alarm_t"] is not None else "-"
        print(f"{label:<42} {fps:>4} {r['hz']:>5g} {r['alarms']:>7} {first:>8} "
              f"{r['max_P']:>5.1f}% {r['n_P45']:>5} {r['n_P70']:>5} {r['runtime_ms']:>6}")
        rows.append((label, r))
    if not args.quiet:
        print("-" * 100)
        print("logs in", OUT)


if __name__ == "__main__":
    main()
