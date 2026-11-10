# Linux capture

Build the detector from this directory with:

```sh
make
```

Requirements: x86_64 CPU with AVX2/FMA/BMI2/POPCNT, C++20, pkg-config, FFmpeg development libraries (avdevice, avformat, avcodec, swscale, avutil), X11/XRandR development libraries for the X11 fallback, Python 3 with PyGObject, GStreamer with GstApp/GstVideo introspection and base/good plugins, and an xdg-desktop-portal ScreenCast backend with PipeWire.

On Wayland, start with `./detector wayland 60`. Choose the physical monitor in the compositor's sharing dialog. The portal uses negotiated video dimensions and source timestamps. A disconnect ends capture with an error, and a capture gap clears detector timing history.

The command `./detector "$DISPLAY" 60` selects X11 in an X11 session. Under Wayland, `$DISPLAY` usually refers to XWayland rather than the composed desktop; the detector routes it to the monitor portal. For explicit X11/XWayland capture use `PSE_CAPTURE=x11 ./detector "$DISPLAY" 60`.

Run the full FFmpeg, pipe, timestamp, and analysis path with generated video, without opening the portal:

```sh
PSE_NO_PROTECTION=1 PSE_TEST_CAPTURE=1 PSE_MAX_FRAMES=120 ./detector
```

Per-frame debug output is off by default to keep capture responsive. Set `PSE_DIAGNOSTICS=1` to print it.

Run synthetic detector and portal protocol/GStreamer tests with `make check`.
