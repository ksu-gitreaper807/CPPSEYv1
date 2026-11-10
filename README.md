# CPPSEYv1

An experimental C++20 screen-flash detector using AVX2, with fast and slow
temporal analysis paths, saturated-red analysis, and Linux screen capture.

## Build and run

```sh
cd AVX25
make
./detector wayland 60
```

The Wayland helper opens the desktop's screen-sharing chooser through
xdg-desktop-portal and receives frames through PipeWire/GStreamer. X11 capture
is also supported. See [Linux setup and capture options](AVX25/LINUX-CAPTURE.md)
for dependencies and configuration.

Wayland currently reports detections and alarms; XRandR gamma changes cannot
protect a physical Wayland display. Risk scores are heuristic, and the project
has not been clinically validated.

## Verification and benchmarks

```sh
cd AVX25
make check
make benchmark
```

`make check` runs the standalone probability/intervention checks, Python
detector and portal regressions, and the integrated FFmpeg capture path with
generated video. It does not open a screen-sharing prompt.

`make benchmark` measures four representative synthetic workloads with one warm-up
and three measured subprocesses per case. Pass `--all` to `test/benchmark.py`
to measure the full 33-case evaluation matrix. Raw results go to
`AVX25/test/results/benchmark.json`. See the
[measured results and methodology](AVX25/BENCHMARKS.md).

Measured on an Intel Core i7-12800H with GCC 16.2.1 (10 November 2026):

| Synthetic workload | Median throughput |
|---|---:|
| full-screen 3.0 Hz | 57.8 frames/s |
| full-screen 6.0 Hz | 49.6 frames/s |
| 6.0 Hz @ 60fps, 1080p native | 21.9 frames/s |
| static grey | 79.8 frames/s |

These are synthetic harness measurements, including generation and analysis,
with one warm-up and three measured runs per workload. They do not measure live
screen-capture latency. The 720p flash workloads measured below 60 frames/s.
