"""Behavioral regressions using synthetic pixels; never touches a display."""
import os
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "test/build/detector_harness"


class DetectionTests(unittest.TestCase):
    def run_case(self, case, hz=6, fps=60, frames=240, size="640x360"):
        env = dict(os.environ, PSE_SYNTH_CASE=case, PSE_SYNTH_HZ=str(hz),
                   PSE_SYNTH_FPS=str(fps), PSE_SYNTH_FRAMES=str(frames),
                   PSE_SYNTH_SIZE=size, PSE_NO_PROTECTION="1",
                   WAYLAND_DISPLAY="deliberately-unavailable",
                   DISPLAY=":9999", PSE_CAPTURE="wayland", PSE_MAX_FRAMES="0")
        result = subprocess.run([str(HARNESS), "desktop", str(fps)], env=env,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("Function Timing Report", result.stdout)
        self.assertNotIn("Portal", result.stderr)
        return result.stdout

    def test_full_screen_luminance_rates(self):
        for hz in (3, 4, 6, 10, 15, 20, 30):
            with self.subTest(hz=hz):
                self.assertIn("FLASH ALARM", self.run_case("flash_hz", hz))

    def test_red_and_partial_flashes(self):
        for case in ("flash_red", "flash_part_40", "flash_half_aph"):
            with self.subTest(case=case):
                self.assertIn("FLASH ALARM", self.run_case(case, 4))

    def test_non_flashing_content(self):
        for case in ("static_gray", "static_checker", "red_static",
                     "scene_cut", "grad_ramp", "noise_white", "flash_small"):
            with self.subTest(case=case):
                output = self.run_case(case, frames=420)
                self.assertNotIn("FLASH ALARM", output)
                self.assertNotIn("Protection active", output)

    def test_slow_changes(self):
        for hz in (1, 2, 2.5):
            with self.subTest(hz=hz):
                output = self.run_case("flash_hz", hz, frames=300)
                self.assertNotIn("FLASH ALARM", output)
                self.assertNotIn("Protection active", output)

    def test_frame_rate_scaling(self):
        for fps in (24, 30, 120):
            with self.subTest(fps=fps):
                self.assertIn("FLASH ALARM", self.run_case(
                    "flash_hz", 3, fps=fps, frames=fps * 4))


if __name__ == "__main__":
    unittest.main()
