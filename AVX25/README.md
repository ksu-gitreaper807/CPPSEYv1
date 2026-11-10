# AVX25 detector

Build with `make`, run checks with `make check`, and measure the synthetic
workloads with `make benchmark`. See [Linux capture](LINUX-CAPTURE.md)
for runtime dependencies and [benchmarks](BENCHMARKS.md) for measured results.

For a custom benchmark output or repeat count:

```sh
make -C test
python3 test/benchmark.py --repeats 5 --output test/results/benchmark.json
```

The harness generates synthetic BGR0 pixels, uses the detector's conversion
and analysis code, and substitutes FFmpeg/X11 interfaces. Its downsampling
implementation is a test substitute, so its 1080p measurements are not a
benchmark of production libswscale or live screen capture.
