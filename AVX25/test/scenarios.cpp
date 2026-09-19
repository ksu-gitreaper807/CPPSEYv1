// scenarios.cpp — synthetic PSE test content for the detector harness.
//
// Generates BGR0 frames (as a real x11grab capture would) so the UNMODIFIED
// detector pipeline — BGR0→YUV420p, downsample, dual PSE detectors — can be
// exercised deterministically without a real display or video file.
//
// Selected by environment variables:
//   PSE_SYNTH_CASE   — case name (see list below)
//   PSE_SYNTH_FPS    — nominal capture frame rate (double)
//   PSE_SYNTH_FRAMES — total frames (set by the harness runner)
//
// Case list
//   static_gray       — uniform mid grey (pure negative)
//   static_checker    — static 64px black/white checkerboard (high-contrast negative)
//   red_static        — static saturated red (sat-red negative: colour present, no flash)
//   scene_cut         — 5s grey 40 then 5s grey 220 (single hard cut, negative)
//   grad_ramp         — triangular brightness sweep 60..200, 8s period (negative)
//   camera_pan        — high-contrast vertical stripes panning at 300 px/s (negative)
//   flash_hz          — full-screen square-wave luminance flash (positive; Hz/L1/L2 env)
//   flash_red         — full-screen saturated red ↔ black square wave (positive)
//   flash_half_aph    — two half-screens flashing in OPPOSITE phase (positive control
//                        for the pixel-level opposing-pair gate)
//   flash_part_25     — 25%-area block flashing on static grey (borderline spatial)
//   flash_part_40     — 40%-area block flashing on static grey (positive)
//   flash_small       — 100x100 block flashing on grey (negative: below area threshold)
//   flash_lowc        — low-contrast full-screen flash 100↔140 (borderline amplitude)
//   flash_checker     — full-screen 64px checkerboard inversion (pattern positive)
//   noise_white       — full-frame white noise (negative: incoherent)
//   blink_small_2hz   — 300x300 block blinking at 2 Hz (negative: small + <3 Hz)
//
// Flash cases read: PSE_SYNTH_HZ (default 4), PSE_SYNTH_L1 (default 0),
// PSE_SYNTH_L2 (default 255).
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>

extern "C" {

static double env_d(const char *k, double dflt)
{
    const char *s = getenv(k);
    return (s && *s) ? strtod(s, nullptr) : dflt;
}
static int env_i(const char *k, int dflt)
{
    return (int)env_d(k, (double)dflt);
}
static std::string env_s(const char *k, const char *dflt)
{
    const char *s = getenv(k);
    return (s && *s) ? std::string(s) : std::string(dflt);
}

// Deterministic per-pixel white noise (xorshift64*)
static uint32_t g_noise_seed = 0x9E3779B97F4A7C15ULL;
static uint32_t noise_u32()
{
    uint64_t x = g_noise_seed;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    g_noise_seed = x;
    return (uint32_t)(x * 0x2545F4914F6CDD1DULL >> 32);
}

static void fill_solid(uint8_t *bgr0, int w, int h, int8_t b, int8_t g, int8_t r)
{
    for (int i = 0; i < w * h; ++i) {
        bgr0[4*i+0] = (uint8_t)b;
        bgr0[4*i+1] = (uint8_t)g;
        bgr0[4*i+2] = (uint8_t)r;
        bgr0[4*i+3] = 0;
    }
}

static bool flash_on(int idx, double fps, double hz)
{
    double t = (double)idx / fps;
    double c = fmod(t * hz, 1.0);
    return c < 0.5;
}

// 50% of each half-period on — used for checkerboard inversions (the whole
// pattern flips twice per period: phase A, phase B, phase A, ...).
static int checker_phase(int idx, double fps, double hz)
{
    double t = (double)idx / fps;
    int n = (int)lround(t * 2.0 * hz);
    return n & 1;
}

extern "C" void pse_synth_frame(int idx, uint8_t *bgr0, int w, int h)
{
    const std::string case_name = env_s("PSE_SYNTH_CASE", "static_gray");
    const double fps = env_d("PSE_SYNTH_FPS", 60.0);
    const double hz  = env_d("PSE_SYNTH_HZ", 4.0);
    const int l1     = env_i("PSE_SYNTH_L1", 0);
    const int l2     = env_i("PSE_SYNTH_L2", 255);
    const bool on    = flash_on(idx, fps, hz);

    if (case_name == "static_gray") {
        fill_solid(bgr0, w, h, 128, 128, 128);

    } else if (case_name == "static_checker") {
        int sq = 64;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int v = (((x / sq) + (y / sq)) & 1) ? 235 : 20;
                bgr0[4*(y*w+x)+0] = (uint8_t)v;
                bgr0[4*(y*w+x)+1] = (uint8_t)v;
                bgr0[4*(y*w+x)+2] = (uint8_t)v;
                bgr0[4*(y*w+x)+3] = 0;
            }

    } else if (case_name == "red_static") {
        fill_solid(bgr0, w, h, 0, 0, 255);

    } else if (case_name == "scene_cut") {
        double t = (double)idx / fps;
        int v = (t < 5.0) ? 40 : 220;
        fill_solid(bgr0, w, h, v, v, v);

    } else if (case_name == "grad_ramp") {
        double t = fmod((double)idx / fps, 8.0);
        double v = (t < 4.0) ? 60.0 + (200.0 - 60.0) * (t / 4.0)
                             : 200.0 - (200.0 - 60.0) * ((t - 4.0) / 4.0);
        int vi = (int)lround(v);
        fill_solid(bgr0, w, h, vi, vi, vi);

    } else if (case_name == "camera_pan") {
        int stripe = 120;
        int period = stripe * 2;
        double t = (double)idx / fps;
        int off = (int)lround(t * 300.0) % period;
        for (int y = 0; y < h; ++y) {
            int row_start = (y * w) * 4;
            for (int x = 0; x < w; ++x) {
                int v = ((((x + off) / stripe) & 1)) ? 40 : 220;
                bgr0[row_start + 4*x + 0] = (uint8_t)v;
                bgr0[row_start + 4*x + 1] = (uint8_t)v;
                bgr0[row_start + 4*x + 2] = (uint8_t)v;
                bgr0[row_start + 4*x + 3] = 0;
            }
        }

    } else if (case_name == "flash_hz") {
        int v = on ? l2 : l1;
        fill_solid(bgr0, w, h, v, v, v);

    } else if (case_name == "flash_red") {
        if (on) fill_solid(bgr0, w, h, 0, 0, 255);
        else    fill_solid(bgr0, w, h, 0, 0, 0);

    } else if (case_name == "flash_half_aph") {
        int lv = on ? l2 : l1;
        int rv = on ? l1 : l2;
        for (int y = 0; y < h; ++y) {
            int row = y * w;
            for (int x = 0; x < w; ++x) {
                int v = (x < w / 2) ? lv : rv;
                bgr0[4*(row+x)+0] = (uint8_t)v;
                bgr0[4*(row+x)+1] = (uint8_t)v;
                bgr0[4*(row+x)+2] = (uint8_t)v;
                bgr0[4*(row+x)+3] = 0;
            }
        }

    } else if (case_name == "flash_part_25" || case_name == "flash_part_40") {
        double frac = (case_name == "flash_part_25") ? 0.5 : 0.632; // 25% / ~40%
        int bw = (int)lround(w * frac); bw &= ~1;
        int bh = (int)lround(h * frac); bh &= ~1;
        int x0 = (w - bw) / 2, y0 = (h - bh) / 2;
        int v = on ? l2 : l1;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int vv = (x >= x0 && x < x0 + bw && y >= y0 && y < y0 + bh) ? v : 128;
                bgr0[4*(y*w+x)+0] = (uint8_t)vv;
                bgr0[4*(y*w+x)+1] = (uint8_t)vv;
                bgr0[4*(y*w+x)+2] = (uint8_t)vv;
                bgr0[4*(y*w+x)+3] = 0;
            }
        }

    } else if (case_name == "flash_small") {
        int sz = 100;
        int x0 = (w - sz) / 2, y0 = (h - sz) / 2;
        int v = on ? 255 : 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int vv = (x >= x0 && x < x0 + sz && y >= y0 && y < y0 + sz) ? v : 128;
                bgr0[4*(y*w+x)+0] = (uint8_t)vv;
                bgr0[4*(y*w+x)+1] = (uint8_t)vv;
                bgr0[4*(y*w+x)+2] = (uint8_t)vv;
                bgr0[4*(y*w+x)+3] = 0;
            }
        }

    } else if (case_name == "flash_lowc") {
        int v = on ? l2 : l1; // runner sets 100/140
        fill_solid(bgr0, w, h, v, v, v);

    } else if (case_name == "flash_checker") {
        int sq = 64;
        int ph = checker_phase(idx, fps, hz);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int v = ((((x / sq) + (y / sq)) & 1) == ph) ? l2 : l1;
                bgr0[4*(y*w+x)+0] = (uint8_t)v;
                bgr0[4*(y*w+x)+1] = (uint8_t)v;
                bgr0[4*(y*w+x)+2] = (uint8_t)v;
                bgr0[4*(y*w+x)+3] = 0;
            }
        }

    } else if (case_name == "noise_white") {
        for (int i = 0; i < w * h; ++i) {
            uint32_t n = noise_u32();
            bgr0[4*i+0] = (uint8_t)(n & 0xFF);
            bgr0[4*i+1] = (uint8_t)((n >> 8) & 0xFF);
            bgr0[4*i+2] = (uint8_t)((n >> 16) & 0xFF);
            bgr0[4*i+3] = 0;
        }

    } else if (case_name == "blink_small_2hz") {
        int sz = 300;
        int x0 = (w - sz) / 2, y0 = (h - sz) / 2;
        int v = on ? 235 : 0; // PSE_SYNTH_HZ=2 set by runner
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int vv = (x >= x0 && x < x0 + sz && y >= y0 && y < y0 + sz) ? v : 60;
                bgr0[4*(y*w+x)+0] = (uint8_t)vv;
                bgr0[4*(y*w+x)+1] = (uint8_t)vv;
                bgr0[4*(y*w+x)+2] = (uint8_t)vv;
                bgr0[4*(y*w+x)+3] = 0;
            }
        }

    } else {
        std::fprintf(stderr, "[fakeff] unknown PSE_SYNTH_CASE '%s' — filling grey\n",
                     case_name.c_str());
        fill_solid(bgr0, w, h, 128, 128, 128);
    }
}

} // extern "C"
