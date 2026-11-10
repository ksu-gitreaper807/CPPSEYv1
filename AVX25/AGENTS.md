# Repository Guidelines

## Project Structure

The Linux detector is implemented in `june/loadv102_DD.cpp`; supporting probability, intervention, and Linux capture code lives in `june/*.hpp`. The Wayland portal/PipeWire helper is `pse_portal_capture.py`. Linux build and capture notes are in `LINUX-CAPTURE.md`. Tests and their fake FFmpeg/X11 interfaces are under `test/`; generated test binaries belong in `test/build/`. Keep platform-specific capture work in the capture layer and avoid coupling analysis logic to a particular desktop environment.

## Build, Test, and Development

- `make` builds `./detector` against the installed FFmpeg and X11 development libraries.
- `make check` builds the synthetic C++ harness, runs Python detection and portal tests, builds the detector, then exercises its FFmpeg capture/analysis path using generated video. It does not require a screen-share prompt.
- `make -C test` builds the detector harness and standalone sensitivity-policy checks.
- `PSE_NO_PROTECTION=1 PSE_TEST_CAPTURE=1 PSE_MAX_FRAMES=120 ./detector` runs a finite synthetic capture; use `PSE_DIAGNOSTICS=1` when per-frame logs are needed.

Production Wayland capture requires a working ScreenCast portal, PipeWire, GStreamer/PyGObject, and user permission. X11 capture requires an accessible X server. See `LINUX-CAPTURE.md` for dependencies and selection behavior.

## Coding Style

Follow the surrounding C++ style: four-space indentation, descriptive `snake_case` helper names, and small focused headers for reusable policies or capture components. Preserve explicit validation around frame geometry, pixel format, timestamps, and subprocess errors. Python helpers should use standard-library `unittest`-friendly interfaces and report actionable errors to stderr. Keep diagnostic output opt-in where it can affect capture throughput.

## Testing Guidelines

Add regression cases beside the relevant tests: detector behavior in `test/test_detection.py`, portal protocol and GStreamer behavior in `test/test_portal_capture.py`, and C++ policy checks in `test/sensitivity_tests.cpp`. Name Python tests `test_*`. Run `make check` after changes to capture, timing, thresholds, or intervention policy; use the finite synthetic capture to verify the integrated executable.

## Commits and Pull Requests

Recent history uses short, imperative summaries (for example, “Wayland-native capture…”). Keep commits similarly focused and explain user-visible behavior. Pull requests should describe the problem and solution, list test commands and results, note any portal/session limitations, and include logs or screenshots only when they clarify a user-visible change. Do not commit generated binaries or test output.

## Configuration and Safety

Do not bypass the desktop portal or silently substitute a fake display when real capture fails. Screen capture requires explicit user consent from the compositor; test mode must remain clearly marked as synthetic.
