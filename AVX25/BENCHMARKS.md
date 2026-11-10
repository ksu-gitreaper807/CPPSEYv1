# Synthetic benchmarks

Measured on 10 November 2026 with an Intel Core i7-12800H (20 logical CPUs),
Linux x86_64, and GCC 16.2.1. The harness uses the flags in `test/Makefile`:
`-O3 -mavx2 -mfma -mbmi2 -mpopcnt -std=c++20 -pthread`.

## Results

| Synthetic workload | Input pixels | Frames/run | Median frames/s | ms/frame | Frames/s range |
|---|---|---:|---:|---:|---:|
| full-screen 3.0 Hz | 1280x720 | 900 | 57.8 | 17.292 | 57.2–59.5 |
| full-screen 6.0 Hz | 1280x720 | 900 | 49.6 | 20.152 | 49.2–51.8 |
| 6.0 Hz @ 60fps, 1080p native | 1920x1080 | 600 | 21.9 | 45.761 | 21.7–22.3 |
| static grey | 1280x720 | 900 | 79.8 | 12.535 | 73.6–87.5 |

All three measured runs of each flash workload produced alarms; static grey
produced none. The 720p flash workloads measured below 60 processing frames/s
on this machine. The 1080p input is downsampled for analysis by the harness.

[Raw samples, environment, and harness binary SHA-256](benchmarks/2026-11-10.json).
The recorded base commit identifies the pre-publication checkout; the committed
source and test Makefile accompany this report.

## Reproduce

```sh
make -C test
python3 test/benchmark.py --output test/results/benchmark.json
# Optional, longer run of every existing evaluation scenario:
python3 test/benchmark.py --all --output test/results/full-matrix.json
```

Each workload runs sequentially in a fresh subprocess, with one unmeasured
warm-up and three measured runs. Protection and detailed diagnostics are
disabled. Wall time includes process startup, synthetic frame generation,
BGR0 conversion, analysis, alarm output and cleanup. Throughput is frame count
divided by median wall time; milliseconds/frame is the same wall time divided
by frame count, not a per-frame latency percentile. No CPU affinity or fixed
CPU frequency was configured.

The harness substitutes FFmpeg/X11 and uses a box-average downsampling
implementation. These measurements do not establish production libswscale
performance, PipeWire/portal throughput, dropped-frame behavior, or live
end-to-end latency. Source timestamps advance at the configured input rate
even when the harness processes frames faster or slower than real time.

## Verification

`make check` passed all 35 Python tests, the standalone probability and
intervention-policy checks (including 10,000 score samples and hold timing at
24/30/60/144 fps), and a real FFmpeg/GStreamer synthetic capture of 20 frames.
The integrated run reported 60 fps source cadence and zero long capture gaps.
Source cadence is distinct from measured processing throughput.

Live compositor capture and physical display intervention were not exercised.
The synthetic checks do not establish clinical safety or an error rate on
real-world videos. Existing compiler warnings remain.
