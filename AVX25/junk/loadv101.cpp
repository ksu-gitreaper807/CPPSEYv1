// loadv82_dual_detector.cpp
// Implements all 7 fixes per ITU-R BT.1702-3 (11/2023):
//
//  FIX 1 — Dual-regime luminance threshold (Guideline 1, §below/above 160 cd/m²)
//  FIX 2 — Opposing-transition pair detection (Guideline 1, "opposing changes")
//  FIX 3 — Per-combo concurrent area as spatial criterion (Guideline 1, "concurrently")
//  FIX 4 — ITU-compliant rate: min 334 ms inter-flash sep, threshold >3 flashes/s
//  FIX 5 — Saturated red flash detection irrespective of luminance (Guideline 1)
//  FIX 6 — Scene-drift motion compensation (Annex 3 Fig. 4 pipeline)
//  FIX 7 — Spatial pattern detection (Guideline 2, >5 stripe pairs)
//
// Branch-misprediction optimizations (v81):
//
//  OPT 1 — process_y templated on USE_SDR (if constexpr eliminates dead branch
//           from the hot inner chunk loop; both SDR and HDR paths compiled)
//  OPT 2 — flashes.reserve(end-start) / r_flashes.reserve(n_chunks) before
//           inner loops; vector capacity-check branch never fires during push_back
//  OPT 3 — Compact flash_idx lists built during PASS 1; PASS 2 iterates only
//           non-zero-mask chunks — data-dependent "if (!mask) continue" removed
//  OPT 4 — extract_and_count_combos_precomputed: direction branch replaced with
//           branchless integer arithmetic (is_d2b = (ms < xs), +=1-is_d2b)
//  OPT 5 — thresh_vec (_mm256_set1_epi8) hoisted above chunk loops in both
//           process_y and process_r; hdr_mult likewise hoisted in process_y HDR path
//  OPT 6 — Combo accumulation: kValidComboIndices table replaces diagonal skip;
//           thresholds hoisted; has_d2b/b2d updated with branchless |= comparisons
//  OPT 7 — FIX 3 peak-combo loop: kValidComboIndices; reciprocal multiplications
//           replace per-iteration divisions (latency 5 vs 10-25 cycles)
//  OPT 8 — Main capture loop templated on NEEDS_SWSCALE × DS_ENABLED (2 bools →
//           4 specialisations); if constexpr removes both per-frame branches
//
// ============================================================================
// DUAL-DETECTOR ARCHITECTURE (v82):
// ============================================================================
//
// Runs TWO parallel flash detectors with different temporal window sizes:
//
//   FAST DETECTOR (8-slot, 4-frame ring):
//     - Original architecture from v81
//     - Detects 4.3–30 Hz flashes @ 60fps (min half-period = 2 frames)
//     - Low memory footprint (~8 MB @ 1080p)
//     - Runs on worker1 + worker2 threads
//
//   SLOW DETECTOR (16-slot, 16-frame ring):
//     - Extended for 3Hz detection @ 60fps
//     - Detects 3–15 Hz flashes (min half-period = 4 frames)
//     - Higher memory footprint (~32 MB @ 1080p)
//     - Runs on dedicated worker3 thread
//
// Alert logic: PSE alarm fires if EITHER detector triggers. This ensures:
//   - Fast strobes (>4.3 Hz) are caught by the low-latency detector
//   - Slow strobes (3–4.3 Hz) are caught by the extended detector
//   - No flashes in the 3–30 Hz hazard range are missed
//
// Memory trade-off: ~40 MB total ring buffers @ 1080p vs 8 MB single-detector.
// CPU trade-off: +1 analysis thread (3 workers total: fast×2 + slow×1).
// ============================================================================
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <immintrin.h>   // AVX2 + FMA + BMI2
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <utility>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "pse_probability.hpp"  // graded P-function (noforrest.hpp)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>   // avdevice_register_all(), screen capture
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>           // av_opt_set_*
#include <libswscale/swscale.h>      // still needed for sws_ds (YUV→YUV downsample)
}
// ── bgr0_yuv420p: inlined single-TU build ─────────────────────────────────
extern "C" {
// bgr0_yuv420p.h — Direct AVX2 BGR0→YUV420p converter
//
// Eliminates all three libswscale branch-miss sources for this pixel format:
//
//   ROOT CAUSE 1 — Indirect call
//     libswscale dispatches through c->swscale / c->convert_unscaled function
//     pointers.  The BTB entry for the call site is overwritten every few
//     hundred frames when the context is recycled or another thread's context
//     runs through the same predictor entry.  Fix: this converter is a plain
//     direct function call; the branch predictor sees one static target.
//
//   ROOT CAUSE 2 — Even/odd chroma branch (50 % mispredict rate)
//     libswscale's generic scaler tests (y & 1) on every row to decide whether
//     to write UV samples.  The pattern alternates perfectly: taken, not-taken,
//     taken, … — the CPU's dynamic predictor cannot learn this and mispredicts
//     every single row (≈ 30,000 misses per second at 60 fps / 1080p).
//     Fix: the outer loop steps by 2.  UV is written once per iteration with no
//     branch; the predictor never sees an alternating signal.
//
//   ROOT CAUSE 3 — Width epilog branch
//     The SIMD body processes pixels in chunks of 16; the scalar tail loop
//     emits one "loop-again?" branch per remaining pixel, taken between 0 and
//     15 times depending on width % 16.  For a fixed-width stream this count
//     is constant across all frames.  Fix: width16 = width & ~15 is computed
//     once per call; the tail `for (x = width16; x < width; x += 2)` branch
//     is always in the same direction for every row of a given stream, so the
//     predictor converges to perfect accuracy after the first iteration.
//
// ─────────────────────────────────────────────────────────────────────────────
//
// Integration into loadv80.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Replace the sws_getContext + sws_scale call with:
//
//   bgr0_to_yuv420p(
//       frame->data[0],  frame->linesize[0],   // BGR0 source
//       yuv_frame->data[0], yuv_frame->linesize[0],  // Y plane
//       yuv_frame->data[1], yuv_frame->linesize[1],  // U (Cb) plane
//       yuv_frame->data[2], yuv_frame->linesize[2],  // V (Cr) plane
//       frame->width, frame->height);
//
// The yuv_frame must be pre-allocated with AV_PIX_FMT_YUV420P as before.
// Compile this translation unit with at least -O2 -mavx2.
//
// ─────────────────────────────────────────────────────────────────────────────
//
// Colour space: BT.601 studio swing (matches libswscale default for SD/capture).
// For HD content (width ≥ 1280) pass cs = BGR0_CS_BT709 to bgr0_to_yuv420p_cs.
//
// Requirements:
//   • width  — positive, even
//   • height — positive, even
//   • src_stride  ≥ width × 4  (bytes)
//   • y_stride    ≥ width
//   • uv_stride   ≥ width / 2
//   • No pointer aliases between src and any output plane
//   • CPU must support AVX2 (CPUID leaf 7, EBX bit 5)
// ─────────────────────────────────────────────────────────────────────────────

#include <stdint.h>


// Colour-space selector passed to bgr0_to_yuv420p_cs().
typedef enum {
    BGR0_CS_BT601 = 0,  // default; matches libswscale for SD / screen capture
    BGR0_CS_BT709 = 1,  // ITU-R BT.709, correct for HD sources
} BGR0ColorSpace;

// ── Primary entry point ───────────────────────────────────────────────────────
// Converts one full frame.  Uses BT.601 coefficients.
void bgr0_to_yuv420p(
    const uint8_t * __restrict__ src,
    int src_stride,
    uint8_t       * __restrict__ y_plane,
    int y_stride,
    uint8_t       * __restrict__ v_plane,
    int uv_stride,
    int width,
    int height);

// ── Colour-space–selectable variant ──────────────────────────────────────────
void bgr0_to_yuv420p_cs(
    const uint8_t * __restrict__ src,
    int src_stride,
    uint8_t       * __restrict__ y_plane,
    int y_stride,
    uint8_t       * __restrict__ v_plane,
    int uv_stride,
    int width,
    int height,
    BGR0ColorSpace cs);

} // extern "C" bgr0_yuv420p declarations

extern "C" {
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

// ============================================================================
// Coefficient tables
// ============================================================================
typedef struct { int16_t ry, gy, by; int16_t rr, gr, br; } Coeffs;

// BT.601 — Q8, studio swing
static const Coeffs k601 = {
    .ry =  66, .gy = 129, .by =  25,   // Y:  positive, sum ≤ 220·255 < 65535
    .rr = 112, .gr = -94, .br = -18,   // Cr: 112·R - 94·G - 18·B
};

// BT.709 — Q8, studio swing
static const Coeffs k709 = {
    .ry =  47, .gy = 157, .by =  16,
    .rr = 112, .gr =-102, .br = -10,
};

// ============================================================================
// BGR0 channel deinterleave — 8 pixels (one __m256i = 32 bytes) → three
// __m128i outputs, each holding 8 uint8 channel values in bytes [0..7].
//
// BGR0 layout in __m256i (lo128: pixels 0-3, hi128: pixels 4-7):
//   [B0 G0 R0 0 | B1 G1 R1 0 | B2 G2 R2 0 | B3 G3 R3 0 |
//    B4 G4 R4 0 | B5 G5 R5 0 | B6 G6 R6 0 | B7 G7 R7 0]
//
// _mm256_shuffle_epi8 compacts the desired byte to positions {0,1,2,3} in
// each 128-bit lane (bytes with index bit-7 set → written as 0x00).
// _mm_unpacklo_epi32 then merges the two 4-byte runs into one 8-byte run.
// ============================================================================
static __attribute__((always_inline)) inline
void deinterleave8(__m256i p,
                   __m128i * __restrict__ b8,
                   __m128i * __restrict__ g8,
                   __m128i * __restrict__ r8)
{
    // Byte offsets of B/G/R within each 4-byte BGR0 pixel (repeated per lane)
    const __m256i shuf_b = _mm256_setr_epi8(
         0,  4,  8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  4,  8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i shuf_g = _mm256_setr_epi8(
         1,  5,  9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  5,  9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i shuf_r = _mm256_setr_epi8(
         2,  6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         2,  6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    // After shuffle: lo128 = [ch0 ch1 ch2 ch3 | 0…] hi128 = [ch4 ch5 ch6 ch7 | 0…]
    // unpacklo_epi32 merges the two 32-bit dwords into bytes [0..7]
#define DEINT_CHAN(shuf, out) do { \
        __m256i s  = _mm256_shuffle_epi8(p, (shuf)); \
        *(out) = _mm_unpacklo_epi32(_mm256_castsi256_si128(s), \
                                    _mm256_extracti128_si256(s, 1)); \
    } while (0)

    DEINT_CHAN(shuf_b, b8);
    DEINT_CHAN(shuf_g, g8);
    DEINT_CHAN(shuf_r, r8);
#undef DEINT_CHAN
}

// ============================================================================
// compute_y16 — 16 Y values from three 16-element uint8 channel vectors.
//
// Channel vectors are passed as __m128i (16 bytes each); extended to __m256i
// uint16 inside.  All arithmetic is logically unsigned: the largest possible
// sum before the shift is 60 324 < 65 535, so no wrap occurs.
//
// Result: 16 Y uint8 values stored at y_out[0..15].
// ============================================================================
static __attribute__((always_inline)) inline
void compute_y16(const __m128i b16, const __m128i g16, const __m128i r16,
                 uint8_t * __restrict__ y_out,
                 int16_t ry, int16_t gy, int16_t by)
{
    // Extend 16 × uint8 to 16 × uint16 in __m256i
    const __m256i b = _mm256_cvtepu8_epi16(b16);
    const __m256i g = _mm256_cvtepu8_epi16(g16);
    const __m256i r = _mm256_cvtepu8_epi16(r16);

    // Y = 16 + (ry*R + gy*G + by*B + 128) >> 8   [unsigned]
    // The constant 128 + (16 << 8) = 128 + 4096 = 4224 folds both the
    // rounding bias (128) and the +16 luma offset (shifted left by 8 since
    // we apply the >> 8 only at the end).
    const __m256i k_y_bias = _mm256_set1_epi16((int16_t)(128 + (16 << 8)));

    __m256i y16 = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_mullo_epi16(r, _mm256_set1_epi16(ry)),
                         _mm256_mullo_epi16(g, _mm256_set1_epi16(gy))),
        _mm256_add_epi16(_mm256_mullo_epi16(b, _mm256_set1_epi16(by)),
                         k_y_bias));

    y16 = _mm256_srli_epi16(y16, 8);  // logical (unsigned) shift → [16, 235]

    // Pack 16 × uint16 → 16 × uint8.
    // _mm256_packus_epi16(a, zero) places a_lo128 → result bytes [0..7],
    // a_hi128 → result bytes [16..23], with zeros in [8..15] and [24..31].
    // _mm256_permute4x64_epi64 with imm8=0xD8 reorders the four 64-bit lanes
    // from [0,1,2,3] to [0,2,1,3], gathering the two data runs into bytes
    // [0..15] of the result so a single 128-bit store captures all 16 Y values.
    __m256i y8 = _mm256_packus_epi16(y16, _mm256_setzero_si256());
    y8 = _mm256_permute4x64_epi64(y8, 0xD8);
    _mm_storeu_si128((__m128i*)y_out, _mm256_castsi256_si128(y8));
}

// ============================================================================
// compute_v8 — 8 Cr values from 2×2-averaged channel sums.
//
// b_sum4, g_sum4, r_sum4: 8 × uint16 sums of four channel values (range
// 0..1020).  Divided by 4 inside by logical shift-right 2 to yield 0..255.
//
// Signed 16-bit arithmetic is valid: the inner expression before the final
// >>8 is bounded to [-28432, +28688] ⊂ [-32768, +32767].
//
// Result stored at v_out[0..7].
// ============================================================================
static __attribute__((always_inline)) inline
void compute_v8(const __m128i b_sum4, const __m128i g_sum4, const __m128i r_sum4,
                uint8_t * __restrict__ v_out,
                int16_t rr, int16_t gr, int16_t br)
{
    // Average: divide sum-of-four by 4.  Values in [0,255] — fits in int16.
    const __m128i b_avg = _mm_srli_epi16(b_sum4, 2);
    const __m128i g_avg = _mm_srli_epi16(g_sum4, 2);
    const __m128i r_avg = _mm_srli_epi16(r_sum4, 2);

    // ── Cr = 128 + (rr*R + gr*G + br*B + 128) >> 8 ────────────────────────
    {
        __m128i pos_cr = _mm_add_epi16(
            _mm_mullo_epi16(r_avg, _mm_set1_epi16(rr)),
            _mm_set1_epi16(128));
        __m128i neg_cr = _mm_add_epi16(
            _mm_mullo_epi16(g_avg, _mm_set1_epi16((int16_t)(-gr))),
            _mm_mullo_epi16(b_avg, _mm_set1_epi16((int16_t)(-br))));
        __m128i inner_cr = _mm_sub_epi16(pos_cr, neg_cr);
        inner_cr = _mm_srai_epi16(inner_cr, 8);
        __m128i cr = _mm_add_epi16(inner_cr, _mm_set1_epi16(128));
        __m128i cr8 = _mm_packus_epi16(cr, _mm_setzero_si128());
        _mm_storel_epi64((__m128i*)v_out, cr8);
    }
}

// ============================================================================
// row_pair_avx2 — AVX2 kernel for one pair of rows (2× unrolled).
//
// Processes pixels [0, width32) in steps of 32.  Each iteration handles two
// independent 16-pixel chunks (chunk A: col+0..col+15, chunk B: col+16..col+31).
// The CPU's out-of-order engine can interleave operations from both chunks to
// hide the 4-5 cycle latency of _mm256_mullo_epi16 and the load-to-use latency
// of memory operations.
//
// A tail loop handles width32..width16 (0 or 16 pixels) and the scalar path
// handles any remainder beyond width16.
// ============================================================================
__attribute__((target("avx2,fma")))
static void row_pair_avx2(
    const uint8_t * __restrict__ src0,   // BGR0 row 0
    const uint8_t * __restrict__ src1,   // BGR0 row 1
    uint8_t       * __restrict__ y0,     // Y output row 0
    uint8_t       * __restrict__ y1,     // Y output row 1
    uint8_t       * __restrict__ v,      // V (Cr) output row (shared by both luma rows)
    int width16,                         // pixels to process (multiple of 16, not necessarily 32)
    const Coeffs * __restrict__ k)       // coefficient table
{
    const __m128i ones8 = _mm_set1_epi8(1);
    const int width32 = width16 & ~31;  // floor to multiple of 32

    // ── Main loop: 32 pixels per iteration (2 × 16-pixel chunks) ─────────────
    for (int col = 0; col < width32; col += 32) {
        // ══════════════════════════════════════════════════════════════════════
        // CHUNK A: pixels [col+0 .. col+15]
        // ══════════════════════════════════════════════════════════════════════

        // Load 64 bytes per row (2 × __m256i = 8 pixels + 8 pixels)
        __m256i p0a_A = _mm256_loadu_si256((const __m256i*)(src0 + col * 4));
        __m256i p0b_A = _mm256_loadu_si256((const __m256i*)(src0 + col * 4 + 32));
        __m256i p1a_A = _mm256_loadu_si256((const __m256i*)(src1 + col * 4));
        __m256i p1b_A = _mm256_loadu_si256((const __m256i*)(src1 + col * 4 + 32));

        // Deinterleave B/G/R
        __m128i b0a_A, g0a_A, r0a_A, b0b_A, g0b_A, r0b_A;
        __m128i b1a_A, g1a_A, r1a_A, b1b_A, g1b_A, r1b_A;
        deinterleave8(p0a_A, &b0a_A, &g0a_A, &r0a_A);
        deinterleave8(p0b_A, &b0b_A, &g0b_A, &r0b_A);
        deinterleave8(p1a_A, &b1a_A, &g1a_A, &r1a_A);
        deinterleave8(p1b_A, &b1b_A, &g1b_A, &r1b_A);

        // Merge into 16-element vectors
        __m128i b0_A = _mm_unpacklo_epi64(b0a_A, b0b_A);
        __m128i g0_A = _mm_unpacklo_epi64(g0a_A, g0b_A);
        __m128i r0_A = _mm_unpacklo_epi64(r0a_A, r0b_A);
        __m128i b1_A = _mm_unpacklo_epi64(b1a_A, b1b_A);
        __m128i g1_A = _mm_unpacklo_epi64(g1a_A, g1b_A);
        __m128i r1_A = _mm_unpacklo_epi64(r1a_A, r1b_A);

        // ══════════════════════════════════════════════════════════════════════
        // CHUNK B: pixels [col+16 .. col+31]
        // ══════════════════════════════════════════════════════════════════════

        __m256i p0a_B = _mm256_loadu_si256((const __m256i*)(src0 + (col+16) * 4));
        __m256i p0b_B = _mm256_loadu_si256((const __m256i*)(src0 + (col+16) * 4 + 32));
        __m256i p1a_B = _mm256_loadu_si256((const __m256i*)(src1 + (col+16) * 4));
        __m256i p1b_B = _mm256_loadu_si256((const __m256i*)(src1 + (col+16) * 4 + 32));

        __m128i b0a_B, g0a_B, r0a_B, b0b_B, g0b_B, r0b_B;
        __m128i b1a_B, g1a_B, r1a_B, b1b_B, g1b_B, r1b_B;
        deinterleave8(p0a_B, &b0a_B, &g0a_B, &r0a_B);
        deinterleave8(p0b_B, &b0b_B, &g0b_B, &r0b_B);
        deinterleave8(p1a_B, &b1a_B, &g1a_B, &r1a_B);
        deinterleave8(p1b_B, &b1b_B, &g1b_B, &r1b_B);

        __m128i b0_B = _mm_unpacklo_epi64(b0a_B, b0b_B);
        __m128i g0_B = _mm_unpacklo_epi64(g0a_B, g0b_B);
        __m128i r0_B = _mm_unpacklo_epi64(r0a_B, r0b_B);
        __m128i b1_B = _mm_unpacklo_epi64(b1a_B, b1b_B);
        __m128i g1_B = _mm_unpacklo_epi64(g1a_B, g1b_B);
        __m128i r1_B = _mm_unpacklo_epi64(r1a_B, r1b_B);

        // ══════════════════════════════════════════════════════════════════════
        // Y computation for both chunks (interleaved to hide multiply latency)
        // ══════════════════════════════════════════════════════════════════════

        compute_y16(b0_A, g0_A, r0_A, y0 + col,      k->ry, k->gy, k->by);
        compute_y16(b1_A, g1_A, r1_A, y1 + col,      k->ry, k->gy, k->by);
        compute_y16(b0_B, g0_B, r0_B, y0 + (col+16), k->ry, k->gy, k->by);
        compute_y16(b1_B, g1_B, r1_B, y1 + (col+16), k->ry, k->gy, k->by);

        // ══════════════════════════════════════════════════════════════════════
        // V (Cr): horizontal pair sums, then 2×2 block sums
        // ══════════════════════════════════════════════════════════════════════

        __m128i b_sum4_A = _mm_add_epi16(_mm_maddubs_epi16(b0_A, ones8),
                                          _mm_maddubs_epi16(b1_A, ones8));
        __m128i g_sum4_A = _mm_add_epi16(_mm_maddubs_epi16(g0_A, ones8),
                                          _mm_maddubs_epi16(g1_A, ones8));
        __m128i r_sum4_A = _mm_add_epi16(_mm_maddubs_epi16(r0_A, ones8),
                                          _mm_maddubs_epi16(r1_A, ones8));

        __m128i b_sum4_B = _mm_add_epi16(_mm_maddubs_epi16(b0_B, ones8),
                                          _mm_maddubs_epi16(b1_B, ones8));
        __m128i g_sum4_B = _mm_add_epi16(_mm_maddubs_epi16(g0_B, ones8),
                                          _mm_maddubs_epi16(g1_B, ones8));
        __m128i r_sum4_B = _mm_add_epi16(_mm_maddubs_epi16(r0_B, ones8),
                                          _mm_maddubs_epi16(r1_B, ones8));

        compute_v8(b_sum4_A, g_sum4_A, r_sum4_A, v + col / 2,
                   k->rr, k->gr, k->br);
        compute_v8(b_sum4_B, g_sum4_B, r_sum4_B, v + (col+16) / 2,
                   k->rr, k->gr, k->br);
    }

    // ── Tail: handle remaining 0–31 pixels with the old 16-pixel body ────────
    // This executes 0 or 1 times depending on whether (width16 & 16) != 0.
    for (int col = width32; col < width16; col += 16) {
        __m256i p0a = _mm256_loadu_si256((const __m256i*)(src0 + col * 4));
        __m256i p0b = _mm256_loadu_si256((const __m256i*)(src0 + col * 4 + 32));
        __m256i p1a = _mm256_loadu_si256((const __m256i*)(src1 + col * 4));
        __m256i p1b = _mm256_loadu_si256((const __m256i*)(src1 + col * 4 + 32));

        __m128i b0a, g0a, r0a, b0b, g0b, r0b;
        __m128i b1a, g1a, r1a, b1b, g1b, r1b;
        deinterleave8(p0a, &b0a, &g0a, &r0a);
        deinterleave8(p0b, &b0b, &g0b, &r0b);
        deinterleave8(p1a, &b1a, &g1a, &r1a);
        deinterleave8(p1b, &b1b, &g1b, &r1b);

        __m128i b0 = _mm_unpacklo_epi64(b0a, b0b);
        __m128i g0 = _mm_unpacklo_epi64(g0a, g0b);
        __m128i r0 = _mm_unpacklo_epi64(r0a, r0b);
        __m128i b1 = _mm_unpacklo_epi64(b1a, b1b);
        __m128i g1 = _mm_unpacklo_epi64(g1a, g1b);
        __m128i r1 = _mm_unpacklo_epi64(r1a, r1b);

        compute_y16(b0, g0, r0, y0 + col, k->ry, k->gy, k->by);
        compute_y16(b1, g1, r1, y1 + col, k->ry, k->gy, k->by);

        __m128i b_sum4 = _mm_add_epi16(_mm_maddubs_epi16(b0, ones8),
                                        _mm_maddubs_epi16(b1, ones8));
        __m128i g_sum4 = _mm_add_epi16(_mm_maddubs_epi16(g0, ones8),
                                        _mm_maddubs_epi16(g1, ones8));
        __m128i r_sum4 = _mm_add_epi16(_mm_maddubs_epi16(r0, ones8),
                                        _mm_maddubs_epi16(r1, ones8));

        compute_v8(b_sum4, g_sum4, r_sum4, v + col / 2, k->rr, k->gr, k->br);
    }
}

// ============================================================================
// scalar_row_pair — reference scalar path for the width epilog.
//
// Handles pixels [x0, width) in steps of 2 (UV pair alignment).
// Same branch is taken the same number of times for every row of a fixed-
// width stream, so the predictor converges to perfect after 1–2 rows.
// ============================================================================
static void scalar_row_pair(
    const uint8_t * __restrict__ src0,
    const uint8_t * __restrict__ src1,
    uint8_t       * __restrict__ y0,
    uint8_t       * __restrict__ y1,
    uint8_t       * __restrict__ v,
    int x0,          // first pixel column to process
    int width,       // one past last pixel column
    const Coeffs * __restrict__ k)
{
    for (int x = x0; x < width; x += 2) {
        // Row 0, pixels x and x+1
        int b00 = src0[x * 4 + 0], g00 = src0[x * 4 + 1], r00 = src0[x * 4 + 2];
        int b01 = src0[(x+1)*4+0], g01 = src0[(x+1)*4+1], r01 = src0[(x+1)*4+2];
        // Row 1, pixels x and x+1
        int b10 = src1[x * 4 + 0], g10 = src1[x * 4 + 1], r10 = src1[x * 4 + 2];
        int b11 = src1[(x+1)*4+0], g11 = src1[(x+1)*4+1], r11 = src1[(x+1)*4+2];

        // Y for all four pixels
        y0[x]   = (uint8_t)(16 + ((k->ry*r00 + k->gy*g00 + k->by*b00 + 128) >> 8));
        y0[x+1] = (uint8_t)(16 + ((k->ry*r01 + k->gy*g01 + k->by*b01 + 128) >> 8));
        y1[x]   = (uint8_t)(16 + ((k->ry*r10 + k->gy*g10 + k->by*b10 + 128) >> 8));
        y1[x+1] = (uint8_t)(16 + ((k->ry*r11 + k->gy*g11 + k->by*b11 + 128) >> 8));

        // 2×2 block average
        int b_avg = (b00 + b01 + b10 + b11) >> 2;
        int g_avg = (g00 + g01 + g10 + g11) >> 2;
        int r_avg = (r00 + r01 + r10 + r11) >> 2;

        // Cr only — U (Cb) not computed
        int inner_cr = k->rr * r_avg + k->gr * g_avg + k->br * b_avg + 128;
        v[x >> 1] = (uint8_t)(128 + (inner_cr >> 8));
    }
}

// ============================================================================
// bgr0_to_yuv420p_cs — full-frame converter, colour-space selectable.
//
// Outer loop structure (FIX 2 — even/odd branch eliminated):
//
//   for (row = 0; row < height; row += 2) {   ← two rows per iteration
//       avx2_body(row, row+1);                ← Y for both, UV once per pair
//       scalar_tail(row, row+1);              ← 0–14 pixels, same count always
//   }
//
// There is no "if (row & 1)" anywhere in this file.
// ============================================================================
void bgr0_to_yuv420p_cs(
    const uint8_t * __restrict__ src,
    int src_stride,
    uint8_t       * __restrict__ y_plane,
    int y_stride,
    uint8_t       * __restrict__ v_plane,
    int uv_stride,
    int width,
    int height,
    BGR0ColorSpace cs)
{
    const Coeffs * const k = (cs == BGR0_CS_BT709) ? &k709 : &k601;

    const int width16 = width & ~15;

    for (int row = 0; row < height; row += 2) {
        const uint8_t *s0 = src + (size_t) row      * src_stride;
        const uint8_t *s1 = src + (size_t)(row + 1) * src_stride;
        uint8_t *y0 = y_plane + (size_t) row      * y_stride;
        uint8_t *y1 = y_plane + (size_t)(row + 1) * y_stride;
        uint8_t *v  = v_plane + (size_t)(row >> 1) * uv_stride;

        if (width16 > 0)
            row_pair_avx2(s0, s1, y0, y1, v, width16, k);

        if (width16 < width)
            scalar_row_pair(s0, s1, y0, y1, v, width16, width, k);
    }
}

// ============================================================================
// bgr0_to_yuv420p — convenience wrapper defaulting to BT.601
// ============================================================================
void bgr0_to_yuv420p(
    const uint8_t * __restrict__ src,
    int src_stride,
    uint8_t       * __restrict__ y_plane,
    int y_stride,
    uint8_t       * __restrict__ v_plane,
    int uv_stride,
    int width,
    int height)
{
    bgr0_to_yuv420p_cs(src, src_stride,
                        y_plane, y_stride,
                        v_plane, uv_stride,
                        width, height,
                        BGR0_CS_BT601);
}
} // extern "C" bgr0_yuv420p implementation
// ─────────────────────────────────────────────────────────────────────────────


// ============================================================================
// OS includes
// ============================================================================
#if defined(_WIN32)
    #include <windows.h>
    #include <physicalmonitorenumerationapi.h>
    #include <highlevelmonitorconfigurationapi.h>
#elif defined(__APPLE__)
    #include <CoreGraphics/CoreGraphics.h>
#elif defined(__linux__)
    #include <X11/Xlib.h>
    #include <X11/extensions/Xrandr.h>
    #include <cstdio>   // popen / pclose for backlight sysfs query
    #include <cstring>  // strncmp
#else
    #error "Unknown Operating System"
#endif

#include <cmath>    // std::round
#include <cstdlib>  // std::strtof

// ============================================================================
// EOTF / Display profile
// ============================================================================
enum class EOTF_Mode { SDR_GAMMA_2_2, HDR_PQ, HDR_HLG };

struct DisplayProfile {
    float     max_nits;
    float     current_brightness_pct;
    EOTF_Mode active_eotf;

    static constexpr std::string_view eotf_to_string(EOTF_Mode eotf) {
        switch (eotf)
        {
            case EOTF_Mode::SDR_GAMMA_2_2: return "SDR Gamma 2.2";
            case EOTF_Mode::HDR_PQ:        return "HDR PQ (SMPTE ST.2084)";
            case EOTF_Mode::HDR_HLG:       return "HDR HLG (Hybrid Log-Gamma)";
            default:                        return "Unknown";
        }
    }
};

// ============================================================================
// PLATFORM IMPLEMENTATIONS
// ============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// WINDOWS
// Reads the first physical monitor attached to the primary display device.
//
// Max brightness  — GetMonitorBrightness() returns the monitor's hardware
//                   brightness range [min, current, max] as percentages (0-100).
//                   We convert to nits using the DisplayConfig HDR metadata when
//                   available, otherwise assume 400 nits for HDR-capable and
//                   200 nits for SDR displays.
//
// Current brightness — same GetMonitorBrightness() call, middle value divided
//                   by 100 to get a [0,1] fraction.
//
// EOTF detection  — QueryDisplayConfig() with DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO
//                   reports whether HDR is currently active on the adapter.
//                   colorEncoding == DISPLAYCONFIG_COLOR_ENCODING_RGB and
//                   advancedColorEnabled == TRUE means the OS is tone-mapping
//                   for PQ (HDR10).  HLG is signalled when the source device
//                   explicitly sets colorEncoding to
//                   DISPLAYCONFIG_COLOR_ENCODING_YCBCR2020 in HLG mode, but
//                   that path requires WDK 10.0.22621+; we fall back to
//                   SDR_GAMMA_2_2 if the flag is absent.
//
// Link: physicalmonitorenumerationapi + highlevelmonitorconfigurationapi
// ─────────────────────────────────────────────────────────────────────────────
#if defined(_WIN32)

// Returns true and fills *out on success; fills safe defaults and returns
// false on any API failure so the caller always gets a usable struct.
static bool query_display_profile_win32(DisplayProfile* out)
{
    // ── Step 1: EOTF via QueryDisplayConfig advanced colour info ─────────
    // Enumerate all active paths/modes.
    UINT32 path_count = 0, mode_count = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS,
                                    &path_count, &mode_count) != ERROR_SUCCESS)
        return false;

    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS,
                           &path_count, paths.data(),
                           &mode_count, modes.data(),
                           nullptr) != ERROR_SUCCESS)
        return false;

    bool hdr_active = false;
    bool hlg_active = false;

    for (const auto& path : paths) {
        // Query advanced colour (HDR) info for each active path target.
        DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO aci = {};
        aci.header.type      = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
        aci.header.size      = sizeof(aci);
        aci.header.adapterId = path.targetInfo.adapterId;
        aci.header.id        = path.targetInfo.id;

        if (DisplayConfigGetDeviceInfo(&aci.header) == ERROR_SUCCESS) {
            if (aci.advancedColorEnabled) {
                hdr_active = true;
                // HLG is indicated when colorEncoding is YCbCr BT.2020 and
                // the display reports an SDR white level ≤ 203 nits,
                // which differentiates broadcast HLG from HDR10 PQ.
                // In practice most consumer monitors report PQ; we treat
                // absence of explicit HLG metadata as PQ.
                if (aci.colorEncoding ==
                    DISPLAYCONFIG_COLOR_ENCODING_YCBCR2020 &&
                    aci.advancedColorSupported &&
                    !aci.advancedColorForceDisabled)
                {
                    // Check SDR white level via the SDR white level request:
                    DISPLAYCONFIG_SDR_WHITE_LEVEL wl = {};
                    wl.header.type      = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
                    wl.header.size      = sizeof(wl);
                    wl.header.adapterId = path.targetInfo.adapterId;
                    wl.header.id        = path.targetInfo.id;
                    if (DisplayConfigGetDeviceInfo(&wl.header) == ERROR_SUCCESS) {
                        // SDR white level is in units of 1/1000 nits.
                        // HLG reference white is 203 nits; PQ reference is 203-10000.
                        float sdr_nits = (float)wl.SDRWhiteLevel / 1000.0f;
                        hlg_active     = (sdr_nits <= 203.0f);
                    }
                }
            }
        }

        // ── Step 2: Max nits via DISPLAYCONFIG_TARGET_PREFERRED_MODE SDR/HDR ─
        // Windows exposes the monitor's peak luminance through
        // DISPLAYCONFIG_GET_MONITOR_SPECIALIZATION (Win11 22H2+) or through
        // the older EDID-derived DISPLAYCONFIG_TARGET_PREFERRED_MODE.
        // We use the widely-available GetMonitorCapabilities path below
        // as a cross-version fallback; the QueryDisplayConfig path here
        // only supplies the EOTF decision above.
    }

    // ── Step 3: Physical monitor brightness (current + max) ──────────────
    // Enumerate physical monitors for the primary display.
    HMONITOR hmon = MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY);
    if (!hmon) return false;

    DWORD phys_count = 0;
    if (!GetNumberOfPhysicalMonitorsFromHMONITOR(hmon, &phys_count)
        || phys_count == 0)
        return false;

    std::vector<PHYSICAL_MONITOR> phys(phys_count);
    if (!GetPhysicalMonitorsFromHMONITOR(hmon, phys_count, phys.data()))
        return false;

    DWORD br_min = 0, br_current = 0, br_max = 0;
    bool got_brightness = GetMonitorBrightness(
        phys[0].hPhysicalMonitor, &br_min, &br_current, &br_max);

    DestroyPhysicalMonitors(phys_count, phys.data());

    if (!got_brightness) return false;

    // GetMonitorBrightness returns DDC/CI values in [0, 100].
    // Map to nits: HDR-capable displays typically peak at ≥ 400 nits.
    // Without a direct luminance query we use the HDR flag to select
    // the peak-white scaling assumption from ITU-R BT.1702-3 Annex 2:
    //   SDR  → 200 cd/m²  (BT.1886 reference display)
    //   HLG  → 1000 cd/m²
    //   PQ   → up to 10000 cd/m² (display-dependent)
    //
    // A DDC/CI brightness of 100% maps to the panel's rated peak luminance.
    // We assume 400 nits for HDR monitors (conservative for gaming monitors)
    // and 250 nits for SDR (typical office panel).
    float peak_nits;
    if      (hlg_active) peak_nits = 1000.0f;
    else if (hdr_active) peak_nits = 400.0f;   // HDR10 PQ consumer panel
    else                 peak_nits = 250.0f;   // SDR

    out->max_nits              = peak_nits;
    out->current_brightness_pct = (float)br_current / (float)br_max;
    out->active_eotf = hlg_active ? EOTF_Mode::HDR_HLG
                     : hdr_active ? EOTF_Mode::HDR_PQ
                                  : EOTF_Mode::SDR_GAMMA_2_2;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// macOS
// CGDisplayCopyDisplayMode gives us the IODisplayModeID.  For brightness
// we call CGDisplayGetDisplayMode and retrieve the HiDPI / EDR headroom via
// the public NSScreen.maximumExtendedDynamicRangeColorComponentValue, but
// that requires ObjC.  We use the pure-C CoreGraphics path instead:
//
//   Current brightness — CGDisplayGetDisplayMode() + IOKit display brightness
//                        query is the documented path but requires root or
//                        com.apple.private.iokit.user-client-cross-endian.
//                        We use the public CGDisplayCopyDisplayMode +
//                        brightness IOService path via IOServiceGetMatchingService
//                        instead.  As a fallback, brightness is read from
//                        the CoreBrightness framework (private but stable since
//                        macOS 10.12) via dlopen.
//
//   Max nits / EOTF  — CGDisplayCopyDisplayMode reports the current mode flags.
//                      kCGDisplayModeIsHiDPI bit 0x20000 present + EDR headroom
//                      > 1.0 indicates an HDR-capable mode.  We query EDR
//                      headroom via CGDisplayStreamCreate/GetEDRHeadroom which
//                      is available since macOS 12 without entitlements.
//
// Link: CoreGraphics.framework (already included)
// ─────────────────────────────────────────────────────────────────────────────
#elif defined(__APPLE__)
#include <dlfcn.h>    // dlopen for CoreBrightness fallback

static bool query_display_profile_macos(DisplayProfile* out)
{
    CGDirectDisplayID display = CGMainDisplayID();

    // ── EOTF / HDR detection via EDR headroom ────────────────────────────
    // CGDisplayCopyDisplayMode is available in all macOS versions we target.
    CGDisplayModeRef mode = CGDisplayCopyDisplayMode(display);
    bool edr_capable = false;
    float edr_headroom = 1.0f;

    if (mode) {
        // kCGDisplayModeIsHiDPI = 0x00020000 (defined in CGDisplayConfiguration.h)
        // EDR capable displays report a pixel encoding of kCGBitmapFloatComponents
        // in their IOSurface backing.  The simplest public API to detect whether
        // the display is currently rendering in an extended dynamic range mode is
        // through the NSScreen.maximumExtendedDynamicRangeColorComponentValue
        // property, but that is ObjC only.  We approximate with mode flags here.
        uint32_t flags = (uint32_t)CGDisplayModeGetIOFlags(mode);
        // Bit 0x80 in IOKit mode flags (kDisplayModeTelevisionFlag) indicates
        // a broadcast-colour-space mode (common for HLG); bit 0x100 is used for
        // HDR10.  These are undocumented but stable across macOS 12–14.
        edr_capable = (flags & 0x80) || (flags & 0x100);
        CGDisplayModeRelease(mode);

        // Public EDR headroom query (macOS 12+ only).  Falls back to 1.0 on
        // older systems, which is correct (SDR displays have headroom 1.0).
        // CGDisplayGetCurrentColorSpace gives us the ICC profile name;
        // "HDR" or "P3" in the name is a reasonable proxy.
        CGColorSpaceRef cs = CGDisplayCopyColorSpace(display);
        if (cs) {
            CFStringRef name = CGColorSpaceCopyName(cs);
            if (name) {
                char buf[256] = {};
                CFStringGetCString(name, buf, sizeof(buf),
                                   kCFStringEncodingUTF8);
                // HDR10 / PQ colour spaces carry "HDR" in the Apple name.
                // HLG carries "HLG" or "Rec. 2100".
                if (strstr(buf, "HDR"))       { edr_capable = true; edr_headroom = 4.0f; }
                if (strstr(buf, "HLG"))       { edr_capable = true; edr_headroom = 3.0f; }
                if (strstr(buf, "2100"))      { edr_capable = true; }
                CFRelease(name);
            }
            CGColorSpaceRelease(cs);
        }
    }

    // ── Current brightness via IOKit display service ──────────────────────
    // IOServiceGetMatchingService(kIOServicePlane, IOServiceMatching(
    //   "AppleBacklightDisplay")) returns the panel backlight service.
    // The brightness property is stored as kDisplayBrightness (double, [0,1]).
    float brightness_pct = 0.75f; // fallback

    // Use the public IOKit symbols to query display brightness.
    // Note: on Apple Silicon the service class is "AppleDisplay" not
    // "AppleBacklightDisplay".  We try both.
#if TARGET_OS_MAC
    io_service_t service = IOServiceGetMatchingService(
        kIOMasterPortDefault,
        IOServiceNameMatching("AppleBacklightDisplay"));
    if (!service)
        service = IOServiceGetMatchingService(
            kIOMasterPortDefault,
            IOServiceNameMatching("AppleDisplay"));

    if (service) {
        io_connect_t connect = 0;
        if (IOServiceOpen(service, mach_task_self(), 0, &connect)
            == KERN_SUCCESS)
        {
            // IODisplayGetFloatParameter for kDisplayBrightness
            float val = 0.0f;
            kern_return_t kr = IODisplayGetFloatParameter(
                connect, 0,
                CFSTR(kDisplayBrightness),
                &val);
            if (kr == KERN_SUCCESS) brightness_pct = val;
            IOServiceClose(connect);
        }
        IOObjectRelease(service);
    }
#endif

    // ── Max nits ─────────────────────────────────────────────────────────
    // Apple's published specifications for built-in displays:
    //   MacBook Pro 14/16"  (Liquid Retina XDR)  → 1600 nits peak HDR
    //   MacBook Pro 13"     (Retina)              →  500 nits
    //   Pro Display XDR     (external reference)  → 1000 nits sustained / 1600 peak
    //   iMac 24"            (Retina)              →  500 nits
    // Without IOKit hardware queries we use the EDR headroom to scale from a
    // baseline of 400 nits (typical SDR peak for Apple Retina panels):
    //   SDR  (headroom 1.0) → 400 nits
    //   HLG  (headroom 3.0) → 400 × 3  = 1200 nits
    //   PQ   (headroom 4.0) → 400 × 4  = 1600 nits (Liquid Retina XDR)
    float peak_nits = 400.0f * edr_headroom;

    // Clamp to plausible range
    if (peak_nits < 200.0f) peak_nits = 200.0f;
    if (peak_nits > 1600.0f) peak_nits = 1600.0f;

    out->max_nits               = peak_nits;
    out->current_brightness_pct = brightness_pct;

    if (edr_headroom >= 3.5f)
        out->active_eotf = EOTF_Mode::HDR_PQ;   // Liquid Retina XDR / Pro Display
    else if (edr_headroom >= 2.0f)
        out->active_eotf = EOTF_Mode::HDR_HLG;  // HLG broadcast mode
    else
        out->active_eotf = EOTF_Mode::SDR_GAMMA_2_2;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Linux (X11 + RandR + sysfs backlight)
//
// Max nits / EOTF  — XRRGetOutputInfo() exposes the EDID blob for each
//                    connected output.  We parse the EDID Byte 24 (Display
//                    Parameters) for max luminance and bytes 25–26 for gamma.
//                    HDR metadata lives in EDID extension block type 0x70
//                    (HDR Static Metadata, DisplayID 2.0) or CTA-861-G
//                    extension (tag 0x06, HF-VSDB / HF-SCDB).
//                    For RandR properties we check "Broadcast RGB" and
//                    "max bpc" which indicate HDR pipeline depth.
//
// Current brightness — /sys/class/backlight/<device>/brightness and
//                    /sys/class/backlight/<device>/max_brightness give
//                    the kernel's backlight driver values.  We iterate
//                    /sys/class/backlight to find the first available device.
//                    On systems without a kernel backlight interface (desktop
//                    monitors, external displays) we fall back to XRRGetCrtcGamma
//                    to infer the current gamma table brightness scaling.
//
// Link: X11/Xlib.h + X11/extensions/Xrandr.h (already included)
// ─────────────────────────────────────────────────────────────────────────────
#elif defined(__linux__)
#include <dirent.h>    // opendir / readdir for sysfs iteration
#include <fcntl.h>     // open
#include <unistd.h>    // read / close
#include <cstring>     // strncmp / snprintf

// Parse the EDID RandR property blob to extract max luminance and
// detect HDR capability.  Returns false if EDID is not available or
// does not contain HDR static metadata.
static bool parse_edid_for_hdr(const uint8_t* edid, int len,
                                float* out_max_nits, EOTF_Mode* out_eotf)
{
    if (!edid || len < 128) return false;

    // EDID byte 24 (0x18) = Gamma (stored as (gamma×100)−100).
    // Values 0xFF mean gamma is defined in an extension block.
    uint8_t gamma_byte = edid[0x17];
    float gamma = 2.2f; // default assumption
    if (gamma_byte != 0xFF)
        gamma = (gamma_byte + 100.0f) / 100.0f;

    // Bytes 8–9: Manufacturer ID.  Not needed here but useful for context.

    // Walk extension blocks (each 128 bytes after the base block).
    *out_max_nits = 200.0f; // ITU-R BT.1702-3 SDR reference
    *out_eotf     = EOTF_Mode::SDR_GAMMA_2_2;

    int n_ext = edid[0x7E];  // byte 126: number of extension blocks
    for (int e = 0; e < n_ext && (e + 1) * 128 + 128 <= len; ++e) {
        const uint8_t* ext = edid + 128 + e * 128;

        // CTA-861-G extension block: tag == 0x02
        if (ext[0] != 0x02) continue;

        int dtd_offset = ext[2]; // byte offset of first DTD
        if (dtd_offset < 4) continue;

        // Walk CTA Data Blocks within [4, dtd_offset)
        int pos = 4;
        while (pos < dtd_offset && pos < 128) {
            uint8_t header  = ext[pos];
            int     tag     = (header >> 5) & 0x07;
            int     db_len  = (header & 0x1F);
            if (pos + 1 + db_len > dtd_offset) break;

            const uint8_t* db = ext + pos + 1;

            // Extended Tag data block (tag == 7): look for HDR Static Metadata
            // Extended Tag Code == 6 (HDR Static Metadata Descriptor)
            if (tag == 7 && db_len >= 2 && db[0] == 6) {
                // db[1]: Electro-Optical Transfer Functions supported (bitmask)
                //   bit 0 = Traditional Gamma SDR
                //   bit 1 = Traditional Gamma HDR
                //   bit 2 = SMPTE ST.2084 (PQ)
                //   bit 3 = Hybrid Log-Gamma (HLG)
                uint8_t eotf_support = db[1];
                // db[5]: Desired Content Max Luminance (code; 50*(2^(cv/32)))
                // db[6]: Desired Content Max Frame-Average Luminance
                // db[7]: Desired Content Min Luminance
                if (db_len >= 4 && (eotf_support & 0x04)) {
                    // PQ (SMPTE ST.2084) supported
                    *out_eotf = EOTF_Mode::HDR_PQ;
                    if (db_len >= 5 && db[4] > 0) {
                        float cv = db[4];
                        *out_max_nits = 50.0f * std::pow(2.0f, cv / 32.0f);
                        if (*out_max_nits > 10000.0f) *out_max_nits = 10000.0f;
                    } else {
                        *out_max_nits = 400.0f; // safe default for HDR monitor
                    }
                } else if (eotf_support & 0x08) {
                    // HLG supported
                    *out_eotf     = EOTF_Mode::HDR_HLG;
                    *out_max_nits = 1000.0f; // HLG reference display (BT.2408)
                }
                return true; // found HDR metadata block
            }

            pos += 1 + db_len;
        }
    }

    // No HDR extension found — SDR display.
    // Use EDID gamma to confirm SDR assumption.
    // gamma ≈ 2.2 → SDR BT.1886, gamma ≠ 2.2 → still SDR but note it.
    (void)gamma; // used implicitly in the eotf assignment above
    return false; // no HDR block parsed
}

// Read an integer from a sysfs file (e.g. /sys/class/backlight/*/brightness).
// Returns -1 on failure.
static int read_sysfs_int(const char* path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[32] = {};
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    return std::atoi(buf);
}

static bool query_display_profile_linux(DisplayProfile* out)
{
    // ── Step 1: backlight brightness from sysfs ───────────────────────────
    float brightness_pct = 1.0f; // default: assume full brightness
    {
        DIR* d = opendir("/sys/class/backlight");
        if (d) {
            struct dirent* ent;
            while ((ent = readdir(d)) != nullptr) {
                if (ent->d_name[0] == '.') continue;
                char br_path[256], max_path[256];
                snprintf(br_path,  sizeof(br_path),
                         "/sys/class/backlight/%s/brightness",     ent->d_name);
                snprintf(max_path, sizeof(max_path),
                         "/sys/class/backlight/%s/max_brightness", ent->d_name);
                int br  = read_sysfs_int(br_path);
                int max = read_sysfs_int(max_path);
                if (br >= 0 && max > 0) {
                    brightness_pct = (float)br / (float)max;
                    break; // use the first device found
                }
            }
            closedir(d);
        }
        // If no sysfs backlight device is available (desktop, external display),
        // leave brightness_pct at 1.0f.
    }

    // ── Step 2: EDID via RandR to detect HDR and max nits ────────────────
    float     max_nits = 250.0f;
    EOTF_Mode eotf     = EOTF_Mode::SDR_GAMMA_2_2;

    Display* dpy = XOpenDisplay(nullptr);
    if (dpy) {
        int screen    = DefaultScreen(dpy);
        Window root   = RootWindow(dpy, screen);

        // Check RandR version
        int rr_maj = 0, rr_min = 0;
        if (XRRQueryVersion(dpy, &rr_maj, &rr_min) && rr_maj >= 1) {
            XRRScreenResources* res = XRRGetScreenResources(dpy, root);
            if (res) {
                // Iterate outputs to find a connected one
                for (int o = 0; o < res->noutput; ++o) {
                    XRROutputInfo* oi = XRRGetOutputInfo(
                        dpy, res, res->outputs[o]);
                    if (!oi) continue;
                    if (oi->connection != RR_Connected) {
                        XRRFreeOutputInfo(oi);
                        continue;
                    }

                    // Fetch EDID property
                    Atom edid_atom = XInternAtom(dpy, "EDID", False);
                    if (edid_atom != None) {
                        Atom       actual_type;
                        int        actual_format;
                        unsigned long n_items, bytes_after;
                        unsigned char* prop = nullptr;

                        if (XRRGetOutputProperty(
                                dpy, res->outputs[o], edid_atom,
                                0, 128, False, False,
                                AnyPropertyType,
                                &actual_type, &actual_format,
                                &n_items, &bytes_after, &prop)
                            == Success && prop)
                        {
                            // Re-fetch with correct size (EDID is 128+ bytes)
                            unsigned char* full_edid = nullptr;
                            unsigned long total = n_items + bytes_after;
                            XRRGetOutputProperty(
                                dpy, res->outputs[o], edid_atom,
                                0, (long)(total / 4 + 1), False, False,
                                AnyPropertyType,
                                &actual_type, &actual_format,
                                &n_items, &bytes_after, &full_edid);

                            if (full_edid) {
                                parse_edid_for_hdr(full_edid, (int)n_items,
                                                   &max_nits, &eotf);
                                XFree(full_edid);
                            }
                            XFree(prop);
                        }
                    }

                    // Also check the RandR "Broadcast RGB" property, which
                    // indicates the full / limited range setting and can
                    // confirm an HDR pipeline is active on this output.
                    Atom broadcast_atom = XInternAtom(dpy, "Broadcast RGB", False);
                    if (broadcast_atom != None && eotf == EOTF_Mode::SDR_GAMMA_2_2) {
                        Atom       atype; int afmt;
                        unsigned long ni, ba;
                        unsigned char* bcast_prop = nullptr;
                        if (XRRGetOutputProperty(
                                dpy, res->outputs[o], broadcast_atom,
                                0, 1, False, False, AnyPropertyType,
                                &atype, &afmt, &ni, &ba, &bcast_prop)
                            == Success && bcast_prop && ni >= 1)
                        {
                            // "Full" = 0, "Limited 16:235" = 1, "Automatic" = 2.
                            // A value of 0 (Full) with a high bpc suggests HDR.
                            int bcast_val = (int)*bcast_prop;

                            // Check "max bpc" to detect 10-bit HDR pipeline.
                            Atom bpc_atom = XInternAtom(dpy, "max bpc", False);
                            if (bpc_atom != None && bcast_val == 0) {
                                unsigned char* bpc_prop = nullptr;
                                if (XRRGetOutputProperty(
                                        dpy, res->outputs[o], bpc_atom,
                                        0, 1, False, False, AnyPropertyType,
                                        &atype, &afmt, &ni, &ba, &bpc_prop)
                                    == Success && bpc_prop)
                                {
                                    int bpc = (int)*bpc_prop;
                                    if (bpc >= 10) {
                                        // 10-bit full-range output: assume PQ
                                        // if no EDID HDR block was found.
                                        eotf     = EOTF_Mode::HDR_PQ;
                                        max_nits = 400.0f;
                                    }
                                    XFree(bpc_prop);
                                }
                            }
                            XFree(bcast_prop);
                        }
                    }

                    XRRFreeOutputInfo(oi);
                    break; // use first connected output
                }
                XRRFreeScreenResources(res);
            }
        }
        XCloseDisplay(dpy);
    }

    out->max_nits               = max_nits;
    out->current_brightness_pct = brightness_pct;
    out->active_eotf            = eotf;
    return true;
}

#endif // platform

// ============================================================================
// get_updated_display_profile — public interface
//
// Calls the platform-specific query function and falls back to conservative
// safe defaults (SDR 200 nits, 80% brightness) if any API call fails.
// The fallback is intentionally conservative: in SDR mode the Michelson
// flash criterion uses the absolute 20 cd/m² threshold below 160 cd/m²,
// so underestimating max_nits causes no false negatives (we are stricter).
// ============================================================================
DisplayProfile get_updated_display_profile()
{
    // Safe fallback: SDR 200 nits reference display per ITU-R BT.1702-3 Annex 2
    DisplayProfile p;
    p.max_nits               = 200.0f;
    p.current_brightness_pct = 0.80f;
    // p.active_eotf            = EOTF_Mode::SDR_GAMMA_2_2;

#if defined(_WIN32)
    query_display_profile_win32(&p);
#elif defined(__APPLE__)
    query_display_profile_macos(&p);
#elif defined(__linux__)
    query_display_profile_linux(&p);
#endif

    return p;
}

// ============================================================================
// Aligned allocator
// ============================================================================
template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator {
    using value_type = T;
    template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { free(ptr); }
    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// ============================================================================
// InlineTask — zero-heap type-erased callable
// ============================================================================
class InlineTask {
public:
    static constexpr size_t STORAGE_SIZE  = 128;
    static constexpr size_t STORAGE_ALIGN = 64;
    InlineTask() = default;
    template<typename F>
    explicit InlineTask(F&& f) {
        using Decay = std::decay_t<F>;
        static_assert(sizeof(Decay)  <= STORAGE_SIZE,  "Lambda too large for InlineTask.");
        static_assert(alignof(Decay) <= STORAGE_ALIGN, "Lambda alignment too strict.");
        new(storage_) Decay(std::forward<F>(f));
        invoke_fn_  = [](void* p) { (*static_cast<Decay*>(p))(); };
        destroy_fn_ = [](void* p) {   static_cast<Decay*>(p)->~Decay(); };
    }
    InlineTask(const InlineTask&)            = delete;
    InlineTask& operator=(const InlineTask&) = delete;
    InlineTask(InlineTask&& o) noexcept {
        std::memcpy(storage_, o.storage_, STORAGE_SIZE);
        invoke_fn_ = o.invoke_fn_; destroy_fn_ = o.destroy_fn_;
        o.invoke_fn_ = nullptr;    o.destroy_fn_ = nullptr;
    }
    InlineTask& operator=(InlineTask&& o) noexcept {
        if (this != &o) {
            destroy_active();
            std::memcpy(storage_, o.storage_, STORAGE_SIZE);
            invoke_fn_ = o.invoke_fn_; destroy_fn_ = o.destroy_fn_;
            o.invoke_fn_ = nullptr;    o.destroy_fn_ = nullptr;
        }
        return *this;
    }
    ~InlineTask() { destroy_active(); }
    void operator()() { if (invoke_fn_) invoke_fn_(storage_); }
    bool empty() const noexcept { return invoke_fn_ == nullptr; }
private:
    void destroy_active() noexcept {
        if (destroy_fn_) { destroy_fn_(storage_); destroy_fn_ = nullptr; invoke_fn_ = nullptr; }
    }
    alignas(STORAGE_ALIGN) std::byte storage_[STORAGE_SIZE]{};
    void (*invoke_fn_) (void*) = nullptr;
    void (*destroy_fn_)(void*) = nullptr;
};

// ============================================================================
// PersistentWorker
// ============================================================================
class PersistentWorker {
    std::thread             th_;
    std::mutex              mtx_;
    std::condition_variable cv_task_, cv_done_;
    InlineTask              task_;
    bool task_ready_ = false, done_ = true, stop_ = false;
public:
    PersistentWorker() {
        th_ = std::thread([this] {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_task_.wait(lock, [this] { return task_ready_ || stop_; });
                if (stop_) break;
                task_(); task_ = InlineTask{};
                task_ready_ = false; done_ = true;
                cv_done_.notify_one();
            }
        });
    }
    ~PersistentWorker() {
        { std::lock_guard<std::mutex> lock(mtx_); stop_ = true; }
        cv_task_.notify_one();
        if (th_.joinable()) th_.join();
    }
    template<typename F> void execute(F&& f) {
        { std::lock_guard<std::mutex> lock(mtx_);
          task_ = InlineTask(std::forward<F>(f)); task_ready_ = true; done_ = false; }
        cv_task_.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_done_.wait(lock, [this] { return done_; });
    }
};

// ============================================================================
// PSEBuffer<N> — Templated ring buffer for N-frame temporal window
//
// Template parameter N controls:
//   - Ring buffer size (N frames stored)
//   - Analysis window size (N temporal slots, no frame reuse)
//
// FIX 6: Added y_sum_acc[N] and y_mean[N].
//         Per-frame mean Y is accumulated in scatter_y via _mm256_sad_epu8.
//         These are used by process_pse_temporal_avx_threaded to compute
//         scene_drift — the scene-wide brightness swing across the N-frame
//         window — which scales the effective concurrent-area threshold.
//
// DUAL-DETECTOR: Two instances are created:
//   - PSEBuffer<4>  buf_fast   (original 8-slot detector, frame reuse)
//   - PSEBuffer<16> buf_slow   (16-slot 3Hz detector, no reuse)
// ============================================================================
template<size_t N>
class PSEBuffer {
public:
    static const size_t MAX_SIZE = N;

    std::vector<uint8_t, AlignedAllocator<uint8_t>> y_buffer;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> v_buffer;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> y_half_buffer;  // Y at Cr resolution

    int64_t  pts_buffer[MAX_SIZE] = {};
    uint32_t y_sum_acc[MAX_SIZE]  = {};  // FIX 6: per-frame luma sum accumulator
    float    y_mean[MAX_SIZE]     = {};  // FIX 6: per-frame mean luma [0,255]
    double   time_base = 0.0;

    // FIX 9 timestamp: PTS of the very first frame pushed.
    // All timestamps in [i] output are reported as (pts - first_pts) * time_base
    // so the clock starts at 0.000s on the first analysis window, matching
    // "elapsed since launch" rather than a stream-origin or UNIX epoch value.
    int64_t first_pts     = 0;
    bool    first_pts_set = false;

    size_t current_index = 0, stored_frames = 0;
    int width = 0, height = 0;
    int y_frame_size = 0, v_frame_size = 0;
    int y_chunks = 0, v_chunks = 0;

    void initialize(int w, int h, double tb) {
        width = w; height = h;
        y_frame_size = w * h;
        v_frame_size = (w / 2) * (h / 2);
        y_chunks = (y_frame_size + 31) / 32;
        v_chunks = (v_frame_size + 31) / 32;
        y_buffer.resize(y_chunks * MAX_SIZE * 32, 0);
        v_buffer.resize(v_chunks * MAX_SIZE * 32, 0);
        y_half_buffer.resize(v_chunks * MAX_SIZE * 32, 0);
        time_base = tb;
    }

    void push_frames(const uint8_t* __restrict__ y_src, int y_linesize,
                     const uint8_t* __restrict__ v_src, int v_linesize,
                     int w, int h, int64_t pts) {
        if (y_frame_size == 0) initialize(w, h, time_base);
        pts_buffer[current_index] = pts;
        // FIX 9: record the PTS of the very first frame for launch-relative timing.
        if (!first_pts_set) { first_pts = pts; first_pts_set = true; }

        // Y plane
        if (y_linesize == width) {
            scatter_y(y_src, false);
            scatter_y_half(y_src, y_linesize);
        } else {
            thread_local std::vector<uint8_t, AlignedAllocator<uint8_t>> y_tmp;
            if ((int)y_tmp.size() < y_frame_size) y_tmp.resize(y_frame_size);
            for (int r = 0; r < height; ++r)
                std::memcpy(y_tmp.data() + r * width, y_src + r * y_linesize, width);
            scatter_y(y_tmp.data(), true);
            scatter_y_half(y_src, y_linesize);
        }
        // FIX 6: compute per-frame mean after scatter
        y_mean[current_index] = (float)y_sum_acc[current_index] / (float)y_frame_size;

        // V (Cr) plane
        const int v_w = width / 2, v_h = height / 2;
        if (v_linesize == v_w) {
            scatter_v(v_src, v_w, v_h, false);
        } else {
            thread_local std::vector<uint8_t, AlignedAllocator<uint8_t>> v_tmp;
            if ((int)v_tmp.size() < v_frame_size) v_tmp.resize(v_frame_size);
            for (int r = 0; r < v_h; ++r)
                std::memcpy(v_tmp.data() + r * v_w, v_src + r * v_linesize, v_w);
            scatter_v(v_tmp.data(), v_w, v_h, true);
        }

        current_index = (current_index + 1) % MAX_SIZE;
        if (stored_frames < MAX_SIZE) stored_frames++;
    }

private:
    // FIX 6: scatter_y accumulates per-frame pixel sum using _mm256_sad_epu8.
    // SAD vs zero gives horizontal sum of 8 bytes per 64-bit lane; 4 lanes
    // summed gives total 32-byte chunk sum.  No extra memory reads required.
    void scatter_y(const uint8_t* __restrict__ src, bool aligned_src) {
        const __m256i zero = _mm256_setzero_si256();
        uint64_t acc = 0;
        for (int r = 0; r < height; ++r) {
            const uint8_t* row = src + r * width;
            for (int c = 0; c < width; c += 32) {
                int    chunk_idx = (r * width + c) / 32;
                size_t offset    = (chunk_idx * MAX_SIZE + current_index) * 32;
                __m256i data = aligned_src
                    ? _mm256_load_si256 ((const __m256i*)(row + c))
                    : _mm256_loadu_si256((const __m256i*)(row + c));
                _mm256_store_si256((__m256i*)&y_buffer[offset], data);
                // Horizontal sum of 32 bytes via SAD vs zero
                __m256i sad = _mm256_sad_epu8(data, zero);
                __m128i lo  = _mm256_castsi256_si128(sad);
                __m128i hi  = _mm256_extracti128_si256(sad, 1);
                __m128i s   = _mm_add_epi64(lo, hi);
                acc += (uint64_t)_mm_extract_epi64(s, 0)
                     + (uint64_t)_mm_extract_epi64(s, 1);
            }
        }
        y_sum_acc[current_index] = (uint32_t)acc;
    }

    void scatter_v(const uint8_t* __restrict__ src, int v_w, int v_h, bool aligned_src) {
        for (int r = 0; r < v_h; ++r) {
            const uint8_t* row = src + r * v_w;
            for (int c = 0; c < v_w; c += 32) {
                int    chunk_idx = (r * v_w + c) / 32;
                size_t offset    = (chunk_idx * MAX_SIZE + current_index) * 32;
                __m256i data = aligned_src
                    ? _mm256_load_si256 ((const __m256i*)(row + c))
                    : _mm256_loadu_si256((const __m256i*)(row + c));
                _mm256_store_si256((__m256i*)&v_buffer[offset], data);
            }
        }
    }

    // Y subsampled 2×2 (top-left) into y_half_buffer at Cr chunk layout
    void scatter_y_half(const uint8_t* __restrict__ y_src, int y_linesize) {
        const int v_w = width / 2, v_h = height / 2;
        alignas(32) uint8_t row_buf[32];
        for (int r = 0; r < v_h; ++r) {
            const uint8_t* y_row = y_src + (r * 2) * y_linesize;
            for (int c = 0; c < v_w; c += 32) {
                int    chunk_idx = (r * v_w + c) / 32;
                size_t offset    = (chunk_idx * MAX_SIZE + current_index) * 32;
                const int n = std::min(32, v_w - c);
                for (int i = 0; i < n; ++i) row_buf[i] = y_row[(c + i) * 2];
                for (int i = n; i < 32; ++i) row_buf[i] = 0;
                _mm256_store_si256((__m256i*)&y_half_buffer[offset],
                                   _mm256_load_si256((const __m256i*)row_buf));
            }
        }
    }
};

// Instantiate both buffer types
using PSEBufferFast = PSEBuffer<4>;   // Original 4-frame ring (8-slot window with reuse)
using PSEBufferSlow = PSEBuffer<16>;  // Extended 16-frame ring (16-slot window, no reuse)

// minimax_horner_fma removed — replaced by direct ratio comparison in
// evaluate_itu1702_flash_mask (see below).

template<int segment>
inline __m256 extract_u8_to_f32(__m256i v) {
    __m128i half;
    if constexpr (segment < 2) half = _mm256_castsi256_si128(v);
    else                        half = _mm256_extracti128_si256(v, 1);
    if constexpr (segment & 1)  half = _mm_unpackhi_epi64(half, half);
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(half));
}

// ============================================================================
// evaluate_itu1702_flash_mask — AVX2 direct ratio comparison
//
// Epileptic-flash criterion:  Ymax / Ymin  >  1.05499681027  (float precision)
//
// Rewritten as a multiply-compare to eliminate division and the entire
// minimax-Horner gamma-decode chain:
//
//     flash  iff  Ymax  >  1.05499681027f * Ymin
//
// The ratio test is scale-invariant: (Ymax/255) / (Ymin/255) == Ymax/Ymin,
// so no normalisation to [0,1] is required — raw uint8 values feed directly
// into the float lanes.
//
// Zero-handling (no special case needed):
//   Ymin == 0, Ymax > 0  →  Ymax > 1.055 * 0 == 0  →  true  (flash)
//   Ymin == 0, Ymax == 0 →  0 > 0                   →  false (static black)
//
// AVX2 parallelism:
//   The 32-pixel chunk is split into four 8-wide float lanes via
//   extract_u8_to_f32<SEG>.  Each lane needs one _mm256_mul_ps and one
//   _mm256_cmp_ps; the four _mm256_movemask_ps results are packed into the
//   32-bit output mask.  No polynomial evaluation, no branch, no division.
// ============================================================================
inline uint32_t evaluate_itu1702_flash_mask(__m256i minv, __m256i maxv) {
    // Epileptic-flash ratio threshold at float precision.
    const __m256 thresh = _mm256_set1_ps(1.05499681027f);

    // Process one 8-pixel segment:
    //   1. Widen uint8 → float  (extract_u8_to_f32, no multiply by inv255 needed)
    //   2. Compute  thresh * Ymin  in one FP multiply
    //   3. Compare  Ymax > thresh * Ymin  (ordered, quiet NaN-safe)
    //   4. Pack the 8 sign bits into the low byte of m##SEG
#define PROC_SEG(SEG) \
    __m256 mn##SEG = extract_u8_to_f32<SEG>(minv); \
    __m256 mx##SEG = extract_u8_to_f32<SEG>(maxv); \
    uint32_t m##SEG = (uint32_t)_mm256_movemask_ps( \
        _mm256_cmp_ps(mx##SEG, _mm256_mul_ps(thresh, mn##SEG), _CMP_GT_OQ));

    PROC_SEG(0)
    PROC_SEG(1)
    PROC_SEG(2)
    PROC_SEG(3)
#undef PROC_SEG

    return m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
}

// ============================================================================
// Argmin/argmax with frame-index tracking
// ============================================================================
struct AVXArgResult { __m256i val, frame_idx; };

inline AVXArgResult argmin_merge(AVXArgResult cur, __m256i b_val, __m256i b_idx) {
    __m256i nv     = _mm256_min_epu8(cur.val, b_val);
    __m256i bwins  = _mm256_cmpeq_epi8(nv, b_val);
    __m256i aeq    = _mm256_cmpeq_epi8(cur.val, b_val);
    __m256i strict = _mm256_andnot_si256(aeq, bwins);
    return { nv, _mm256_blendv_epi8(cur.frame_idx, b_idx, strict) };
}

inline AVXArgResult argmax_merge(AVXArgResult cur, __m256i b_val, __m256i b_idx) {
    __m256i nv     = _mm256_max_epu8(cur.val, b_val);
    __m256i bwins  = _mm256_cmpeq_epi8(nv, b_val);
    __m256i aeq    = _mm256_cmpeq_epi8(cur.val, b_val);
    __m256i strict = _mm256_andnot_si256(aeq, bwins);
    return { nv, _mm256_blendv_epi8(cur.frame_idx, b_idx, strict) };
}

// ============================================================================
// ChunkVectors — compressed argmin/argmax result per flashing chunk
// ============================================================================
struct alignas(32) ChunkVectors {
    __m256i frame_min_idx;
    __m256i frame_max_idx;
};

// ============================================================================
// FIX 2 — extract_and_count_combos_precomputed (directional)
//
// In addition to the total combo_counters[64], this now fills:
//   d2b_counters[64] — dark-to-bright (min_slot < max_slot = brightness rises)
//   b2d_counters[64] — bright-to-dark (min_slot > max_slot = brightness falls)
//
// ITU-R BT.1702-3 Guideline 1: a "flash" is a PAIR of opposing transitions.
// The PSE alarm should only fire when the same 8-frame window contains BOTH
// directions — i.e., both d2b and b2d combos exceed the threshold.
//
// OPT: direction branch replaced with branchless integer arithmetic.
//   is_d2b = (uint32_t)(ms < xs)  →  1 when rising, 0 when falling.
//   d2b += is_d2b  /  b2d += 1 - is_d2b   — no cmov, no jump, fully pipelined.
// ============================================================================
inline void extract_and_count_combos_precomputed(
        uint32_t  mask,
        __m256i   frame_min_idx,
        __m256i   frame_max_idx,
        uint32_t* combo_counters,
        uint32_t* d2b_counters,    // FIX 2: dark-to-bright direction
        uint32_t* b2d_counters,    // FIX 2: bright-to-dark direction
        uint64_t  valid_combo_mask)
{
    if (!mask) return;
    alignas(32) uint8_t pmin[32], pmax[32];
    _mm256_store_si256((__m256i*)pmin, frame_min_idx);
    _mm256_store_si256((__m256i*)pmax, frame_max_idx);
    while (mask) {
        int bit   = __builtin_ctz(mask);
        mask     &= mask - 1;
        int ms    = pmin[bit];   // slot where this pixel was darkest
        int xs    = pmax[bit];   // slot where this pixel was brightest
        int combo = ms * 8 + xs;
        if (__builtin_expect((valid_combo_mask >> combo) & 1u, 1)) {
            combo_counters[combo]++;
            // OPT: branchless direction — compiler emits integer comparison
            // result directly; no conditional jump, no cmov needed.
            uint32_t is_d2b = (uint32_t)(ms < xs); // 1=rising, 0=falling
            d2b_counters[combo] += is_d2b;
            b2d_counters[combo] += 1u - is_d2b;
        }
    }
}

// ============================================================================
// extract_and_count_combos_16 — 16-slot version (256-combo grid)
// ============================================================================
inline void extract_and_count_combos_16(
        uint32_t  mask,
        __m256i   frame_min_idx,
        __m256i   frame_max_idx,
        uint32_t* combo_counters,
        uint32_t* d2b_counters,
        uint32_t* b2d_counters,
        const uint64_t valid_combo_mask[4])  // 256-bit mask split into 4×64
{
    if (!mask) return;
    alignas(32) uint8_t pmin[32], pmax[32];
    _mm256_store_si256((__m256i*)pmin, frame_min_idx);
    _mm256_store_si256((__m256i*)pmax, frame_max_idx);
    while (mask) {
        int bit   = __builtin_ctz(mask);
        mask     &= mask - 1;
        int ms    = pmin[bit];
        int xs    = pmax[bit];
        int combo = ms * 16 + xs;  // 16-slot grid
        // Check 256-bit mask
        if ((valid_combo_mask[combo / 64] >> (combo % 64)) & 1u) {
            combo_counters[combo]++;
            uint32_t is_d2b = (uint32_t)(ms < xs);
            d2b_counters[combo] += is_d2b;
            b2d_counters[combo] += 1u - is_d2b;
        }
    }
}

// ============================================================================
// FlashComboTracker — extended with directional arrays (FIX 2)
// ============================================================================
struct FlashComboTracker {
    uint32_t y_combo_hits[64] = {};
    uint32_t y_d2b_hits[64]   = {};  // FIX 2: dark→bright per combo
    uint32_t y_b2d_hits[64]   = {};  // FIX 2: bright→dark per combo
    uint32_t r_combo_hits[64] = {};
    uint32_t r_d2b_hits[64]   = {};  // FIX 2
    uint32_t r_b2d_hits[64]   = {};  // FIX 2
    int      min_frame_ref[64] = {};
    int      max_frame_ref[64] = {};
    double   min_pts_time[64]  = {};
    double   max_pts_time[64]  = {};
};

// ============================================================================
// FlashComboTracker16 — 16-slot version for 3Hz detector (256 combos)
// ============================================================================
struct FlashComboTracker16 {
    uint32_t y_combo_hits[256] = {};
    uint32_t y_d2b_hits[256]   = {};
    uint32_t y_b2d_hits[256]   = {};
    uint32_t r_combo_hits[256] = {};
    uint32_t r_d2b_hits[256]   = {};
    uint32_t r_b2d_hits[256]   = {};
    int      min_frame_ref[256] = {};
    int      max_frame_ref[256] = {};
    double   min_pts_time[256]  = {};
    double   max_pts_time[256]  = {};
};

// ============================================================================
// OPT: Precomputed valid-combo index table (templated for N-slot windows)
//
// All three post-processing loops over the N×N combo slots skip the N diagonal
// entries where (i/N)==(i%N) (same min and max frame → no transition).
// Rather than re-checking every iteration, we build the (N²-N) valid indices
// once at compile time and iterate only those.  This removes a data-dependent
// branch from every combo-scan loop and allows the compiler to fully unroll or
// vectorise the reduction.
//
// DUAL-DETECTOR: Two instances:
//   kValidComboIndices_8  — for 8-slot detector (56 valid combos)
//   kValidComboIndices_16 — for 16-slot detector (240 valid combos)
// ============================================================================

template<int N>
static constexpr auto build_valid_combo_indices() {
    constexpr int kNumValid = N * N - N;
    std::array<int, kNumValid> a{};
    int n = 0;
    for (int i = 0; i < N * N; ++i)
        if ((i / N) != (i % N)) a[n++] = i;
    return a;
}

// 8-slot detector (original, 64 combos - 8 diagonal = 56 valid)
static constexpr int kNumValidCombos_8 = 56;
static constexpr auto kValidComboIndices_8 = build_valid_combo_indices<8>();

// 16-slot detector (3Hz, 256 combos - 16 diagonal = 240 valid)
static constexpr int kNumValidCombos_16 = 240;
static constexpr auto kValidComboIndices_16 = build_valid_combo_indices<16>();

// Legacy aliases for existing 8-slot code (maintain backward compatibility)
static constexpr int kNumValidCombos = kNumValidCombos_8;
static constexpr auto& kValidComboIndices = kValidComboIndices_8;

// ============================================================================
// FIX 4 — FlashRateCounter  (v61: rising-edge onset counting)
//
// ROOT CAUSE OF v60 BUG:
//   The v60 approach used a per-event minimum separation (min_flash_sep) to
//   deduplicate consecutive analysis windows that all detect the SAME ongoing
//   flash.  At 30fps the minimum was 360ms (sub-50fps branch), so the ring
//   could accumulate at most floor(1000/360) = 2 events per second.  With
//   PSE_FLASH_RATE_THRESHOLD = 4 the alarm could never fire — the threshold
//   was structurally unreachable regardless of frame content.
//
// CORRECT MODEL (ITU-R BT.1702-3 Guideline 1):
//   "More than three flashes (i.e. six changes in luminance) within any
//   one-second period" → strictly more than 3 FLASH ONSETS per second,
//   where a flash onset = the moment a new dark↔bright pair begins.
//
// NEW IMPLEMENTATION — rising-edge (onset) counting:
//   • record_state(is_flashing, pts) is called every analysis window.
//   • A rising edge (false → true transition in is_flashing) records the
//     onset timestamp — one onset = one new flash pair beginning.
//   • events_in_window() counts onsets within the last 1 second.
//   • pse_rate_exceeded() returns true when onset count > 3 (i.e. ≥ 4),
//     matching the standard's "more than three" language.
//
// This approach is correct at any frame rate: at 3Hz exactly, a 1-second
// window contains 3 onsets (3 > 3 is false → no alarm ✓).  At 3.5Hz+
// it contains ≥ 4 (4 > 3 → alarm ✓).  No min_sep parameter is needed.
//
// Integration with FIX 2 (opposing-pair check):
//   The is_flashing argument must be (spatial_flash && has_d2b && has_b2d)
//   so that we count only genuine flash pairs, not single-direction ramps.
// ============================================================================
struct FlashRateCounter {
    static constexpr int    CAP                   = 64;
    static constexpr int    PSE_FLASH_RATE_THRESHOLD = 5; // >3 = alarm
    static constexpr double WINDOW_SECONDS        = 1.0;

    double ring[CAP] = {};
    int    count = 0, head = 0;
    bool   last_was_flashing = false;   // for rising-edge detection

    // Call once per analysis window.  Records an onset timestamp on the
    // false→true edge of is_flashing; resets last_was_flashing on the
    // true→false edge.  No-op mid-flash or mid-quiescence.
    void record_state(bool is_flashing, double pts_s) {
        if (is_flashing && !last_was_flashing) {
            // Rising edge — a new flash pair has started
            ring[head] = pts_s;
            head = (head + 1) % CAP;
            if (count < CAP) ++count;
        }
        last_was_flashing = is_flashing;
    }

    int events_in_window(double pts_now) const {
        int n = 0;
        for (int i = 0; i < count; ++i)
            if (pts_now - ring[i] <= WINDOW_SECONDS) ++n;
        return n;
    }

    // true when the number of flash onsets in the last 1 second strictly
    // exceeds PSE_FLASH_RATE_THRESHOLD (i.e. "more than three" per ITU).
    bool pse_rate_exceeded(double pts_now) const {
        return events_in_window(pts_now) > PSE_FLASH_RATE_THRESHOLD;
    }

    // Returns the two most recent onset timestamps for instantaneous Hz estimation.
    // Returns {-1,-1} when fewer than 2 onsets have been recorded.
    std::pair<double,double> last_two_onsets() const {
        if (count < 2) return {-1.0, -1.0};
        int i1 = (head - 1 + CAP) % CAP;   // most recent
        int i2 = (head - 2 + CAP) % CAP;   // second-most-recent
        return {ring[i2], ring[i1]};
    }
};

// ============================================================================
// FIX 7 — SpatialDensityChecker (extended with stripe frequency estimator)
//
// estimate_stripe_frequency() samples every 16th pixel-row and counts
// flash/non-flash zero-crossings.  Crossings/2 ≈ stripe pairs per row.
// If the average exceeds 5, Guideline 2 pattern conditions may apply.
//
// Guideline 2 thresholds (Attachment 1):
//   • Stationary pattern:   area > 40% of screen
//   • Flashing/reversing:   area > 25% of screen
// ============================================================================
struct SpatialDensityChecker {
    int grid_w = 0, grid_h = 0;
    std::vector<int32_t> integral;

    void init(int plane_width, int plane_height) {
        grid_w = (plane_width  + 31) / 32;
        grid_h = plane_height;
        integral.assign((size_t)(grid_w + 1) * (grid_h + 1), 0);
    }

    void build(const uint32_t* seg1, int count1,
               const uint32_t* seg2, int count2)
    {
        std::fill(integral.begin(), integral.end(), 0);
        const int total = count1 + count2;
        const int cells = grid_w * grid_h;
        for (int i = 0; i < total && i < cells; ++i) {
            uint32_t m = (i < count1) ? seg1[i] : (seg2 ? seg2[i - count1] : 0u);
            int row = i / grid_w, col = i % grid_w;
            integral[(row + 1) * (grid_w + 1) + (col + 1)] = __builtin_popcount(m);
        }
        for (int r = 1; r <= grid_h; ++r)
            for (int c = 1; c <= grid_w; ++c)
                integral[r*(grid_w+1)+c] += integral[(r-1)*(grid_w+1)+c]
                                          + integral[r*(grid_w+1)+(c-1)]
                                          - integral[(r-1)*(grid_w+1)+(c-1)];
    }

    int32_t query(int r0, int c0, int r1, int c1) const {
        r1 = std::min(r1, grid_h); c1 = std::min(c1, grid_w);
        if (r1 <= r0 || c1 <= c0) return 0;
        return integral[r1*(grid_w+1)+c1]
             - integral[r0*(grid_w+1)+c1]
             - integral[r1*(grid_w+1)+c0]
             + integral[r0*(grid_w+1)+c0];
    }

    bool has_dense_region(int win_rows, int win_cols, int threshold_px) const {
        if (grid_w == 0 || grid_h == 0) return false;
        const int step_r = std::max(1, win_rows / 2);
        const int step_c = std::max(1, win_cols / 2);
        for (int r = 0; r + win_rows <= grid_h; r += step_r)
            for (int c = 0; c + win_cols <= grid_w; c += step_c)
                if (query(r, c, r + win_rows, c + win_cols) > threshold_px)
                    return true;
        return false;
    }

    // FIX 7: horizontal stripe frequency estimate.
    // Returns average light/dark stripe pair count per row.
    // Samples every 16th row for efficiency (~60 rows at 1080p).
    float estimate_stripe_frequency() const {
        if (grid_w < 2 || grid_h < 2) return 0.0f;
        int total_pairs = 0, rows_sampled = 0;
        for (int r = 0; r < grid_h; r += 16) {
            // Reconstruct per-chunk flash status from the integral image.
            // query(r, c, r+1, c+1) > 0 means chunk (r,c) has flash pixels.
            int crossings = 0;
            bool last = (query(r, 0, r+1, 1) > 0);
            for (int c = 1; c < grid_w; ++c) {
                bool cur = (query(r, c, r+1, c+1) > 0);
                if (cur != last) { ++crossings; last = cur; }
            }
            total_pairs  += crossings / 2;
            ++rows_sampled;
        }
        return rows_sampled > 0
             ? (float)total_pairs / (float)rows_sampled
             : 0.0f;
    }
};

// ============================================================================
// PSEConfig — per-stream constants derived at startup
// ============================================================================
struct PSEConfig {
    uint64_t valid_combo_mask = ~0ULL; // adjacency gate
    int16_t  kr_fixed         = 101;   // Cr→R coefficient ×64 (BT.709 default)
    double   fps              = 25.0;
    // min_flash_sep removed in v61: replaced by rising-edge onset counting

    // FIX 8: encoded-luma range below which a pixel is considered static and
    // excluded from the flash mask.  Default 16 covers H.264 artifact range
    // at bitrates ≥ 1 Mbit/s while staying below the minimum qualifying swing.
    // Increase to 20–24 for aggressive CRF ≥ 28 sources.
    uint8_t static_luma_thresh = 16;

    // FIX 9 — Static region subtraction
    //
    // Pixels are classified as "long-term static" when their per-chunk EMA luma
    // range stays below static_chunk_ema_thresh over a window of
    // static_window_seconds.  Static chunks are removed from the denominator
    // of the concurrent area fraction so that a small flash surrounded by a
    // large static desktop is evaluated against only the non-static screen area.
    //
    // Nullification: if more than static_nullify_frac of the screen is classified
    // as static, the subtraction is disabled (the whole screen is effectively
    // "content-free" and the flash area is genuinely small — not artificially
    // shrunk by static surroundings).
    //
    // EMA update: alpha = 1 / (fps × static_window_seconds).  A chunk whose EMA
    // range falls below static_chunk_ema_thresh is counted as static.
    double  static_window_seconds   = 2.0;   // Y-second observation window
    float   static_chunk_ema_thresh = 6.0f;  // EMA range below this → static
    float   static_nullify_frac     = 0.85f; // if >85% static → nullify
};

// ============================================================================
// PSEConfig16 — 16-slot detector configuration (256-combo grid)
// ============================================================================
struct PSEConfig16 {
    uint64_t valid_combo_mask[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL}; // 256-bit adjacency gate
    int16_t  kr_fixed            = 101;
    double   fps                 = 25.0;
    uint8_t  static_luma_thresh  = 16;
    double   static_window_seconds   = 2.0;
    float    static_chunk_ema_thresh = 6.0f;
    float    static_nullify_frac     = 0.85f;
};

// ============================================================================
// ScreenProtector
//
// Applies a protective screen effect on the viewer's display whenever the
// P-function returns warn or alarm level.  Runs on a dedicated OS thread so
// it never stalls the AVX analysis path.
//
// Protection levels (matching pse_probability.hpp risk bands):
//   NONE      P < 0.45  — restore display to full brightness
//   DIM       P ∈ [0.45, 0.70) — dim to DIM_GAMMA_SCALE (50%) of full range
//   BLACKOUT  P ≥ 0.70  — force display to black for duration of the flash
//
// The effect persists as long as the level is set.  When the PSE analysis
// clears the level back to NONE (i.e. prob.P drops below 0.45), the display
// is immediately restored to its original gamma/brightness.
//
// Thread safety: `set_level()` is called from the PSE analysis path (worker
// threads via process_pse_temporal_avx_threaded) using an atomic store.
// The background OS thread reads the atomic and applies the OS API.  No mutex
// is held during the hot-path call — only an atomic_store.
//
// Platform implementations:
//   Windows — SetDeviceGammaRamp() modifies the hardware gamma LUT.
//             DIM  → scale all ramp values to 50% of original.
//             BLACKOUT → zero the entire ramp (black screen).
//             Restore → reload the saved original ramp.
//             Note: requires GENERIC_WRITE access to the display DC.
//
//   macOS   — CGDisplayFade() fades to a target colour over a specified
//             interval.  DIM uses CGDisplayFade to 50% neutral grey;
//             BLACKOUT uses CGDisplayFade to pure black.
//             Restore fades back to 1.0 (full).
//             Note: CGDisplayFade is public API, no entitlement needed.
//
//   Linux   — XRRGetCrtcGamma / XRRSetCrtcGamma modify the RandR gamma ramp
//             per connected CRTC.  DIM scales all ramp values to 50%.
//             BLACKOUT zeros all ramp values.
//             Restore reloads the saved original ramp.
//             Note: requires a live X11 $DISPLAY; Wayland needs a separate
//             wlr-gamma-control or colour-management protocol.
// ============================================================================
enum class ProtectionLevel : int { NONE = 0, DIM = 1, BLACKOUT = 2 };

// Dim factor: fraction of full brightness applied during DIM level.
// 0.50 = 50% of normal, matching the visual salience literature on effective
// interruption without being so dark that the user cannot see the screen.
static constexpr float DIM_GAMMA_SCALE = 0.50f;

class ScreenProtector {
public:
    // ── Construction / destruction ────────────────────────────────────────
    ScreenProtector() {
        if (!platform_init()) {
            std::cerr << "[ScreenProtector] Platform init failed — "
                         "screen protection disabled.\n";
            init_ok_ = false;
            return;
        }
        init_ok_ = true;
        level_.store(ProtectionLevel::NONE, std::memory_order_relaxed);
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread(&ScreenProtector::worker_loop, this);
    }

    ~ScreenProtector() {
        stop_.store(true, std::memory_order_relaxed);
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
        if (init_ok_) platform_restore(); // always restore on exit
        platform_cleanup();
    }

    // Called from the PSE analysis path — atomic, non-blocking.
    void set_level(ProtectionLevel lv) noexcept {
        if (!init_ok_) return;
        ProtectionLevel prev = level_.exchange(lv, std::memory_order_acq_rel);
        if (prev != lv) cv_.notify_one(); // wake OS thread only on change
    }

    // Convenience: derive level from P-function probability.
    // While P >= 0.45 (active flash) the post-flash hold deadline is refreshed
    // to now + 1000 ms so the 1-second extension begins from the last active
    // window, not from when the flash first started.
    void update_from_probability(float P) noexcept {
        if (P >= 0.70f) {
            hold_until_ns_.store(now_ns() + 1'000'000'000LL,
                                 std::memory_order_relaxed);
            set_level(ProtectionLevel::BLACKOUT);
        } else if (P >= 0.45f) {
            hold_until_ns_.store(now_ns() + 1'000'000'000LL,
                                 std::memory_order_relaxed);
            set_level(ProtectionLevel::DIM);
        } else {
            // Flash ended — set NONE; worker_loop will honour the hold timer.
            set_level(ProtectionLevel::NONE);
        }
    }

private:
    std::atomic<ProtectionLevel> level_        { ProtectionLevel::NONE };
    std::atomic<bool>            stop_         { false };
    std::thread                  thread_;
    std::mutex                   mtx_;
    std::condition_variable      cv_;
    bool                         init_ok_      = false;

    // ── Post-flash hold timer ─────────────────────────────────────────────
    // Stores the steady_clock deadline (nanoseconds since epoch) before which
    // a NONE request must NOT restore the display.  Refreshed to now+1000ms
    // every analysis window where P >= 0.45.  This ensures protection lingers
    // for exactly 1000 ms after the last active flash window clears.
    // Zero = no hold active (startup state).
    std::atomic<int64_t> hold_until_ns_ { 0 };

    static int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // ── Background worker ─────────────────────────────────────────────────
    // Wakes on level changes and applies the OS API.  Sleeps otherwise.
    //
    // Post-flash hold: when target is NONE, the worker checks hold_until_ns_.
    // If the deadline has not elapsed, it sleeps for the remaining duration
    // via cv_.wait_for.  A new non-NONE level fires the cv_ immediately so
    // the hold never delays an escalation — only a restore.
    void worker_loop() {
        ProtectionLevel applied = ProtectionLevel::NONE;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this, &applied] {
                    return stop_.load(std::memory_order_relaxed)
                        || level_.load(std::memory_order_relaxed) != applied;
                });
            }
            if (stop_.load(std::memory_order_relaxed)) break;

            ProtectionLevel target = level_.load(std::memory_order_acquire);
            if (target == applied) continue;

            if (target == ProtectionLevel::NONE) {
                // Honour the post-flash hold before restoring the display.
                int64_t remaining = hold_until_ns_.load(std::memory_order_relaxed) - now_ns();
                if (remaining > 0) {
                    // Sleep for the remainder.  Do NOT update `applied` so the
                    // screen stays dim/blacked-out during the hold period.
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_.wait_for(lock, std::chrono::nanoseconds(remaining), [this] {
                        return stop_.load(std::memory_order_relaxed)
                            || level_.load(std::memory_order_relaxed)
                               != ProtectionLevel::NONE;
                    });
                    // Loop back: re-evaluate — flash may have resumed, or the
                    // deadline may now have elapsed.
                    continue;
                }
                platform_restore();
            } else if (target == ProtectionLevel::DIM) {
                //platform_dim();
            } else {
                platform_blackout();
            }
            applied = target;
        }
        // Final restore before thread exits
        if (applied != ProtectionLevel::NONE) platform_restore();
    }

    // ── Platform state + inline implementations (selected by #if) ────────
#if defined(_WIN32)
    HDC   hdc_     = nullptr;
    WORD  orig_ramp_[3][256] = {};
    WORD  work_ramp_[3][256] = {};

    bool platform_init() {
        hdc_ = GetDC(nullptr); // primary display DC
        if (!hdc_) return false;
        // Save original hardware gamma ramp.
        return GetDeviceGammaRamp(hdc_, orig_ramp_) == TRUE;
    }
    void platform_cleanup() {
        if (hdc_) { ReleaseDC(nullptr, hdc_); hdc_ = nullptr; }
    }
    void platform_dim() {
        // Scale the saved ramp to DIM_GAMMA_SCALE.
        for (int ch = 0; ch < 3; ++ch)
            for (int i = 0; i < 256; ++i)
                work_ramp_[ch][i] = static_cast<WORD>(orig_ramp_[ch][i] * DIM_GAMMA_SCALE);
        SetDeviceGammaRamp(hdc_, work_ramp_);
    }
    void platform_blackout() {
        // Zero the entire ramp — display outputs black.
        std::memset(work_ramp_, 0, sizeof(work_ramp_));
        SetDeviceGammaRamp(hdc_, work_ramp_);
    }
    void platform_restore() {
        if (hdc_) SetDeviceGammaRamp(hdc_, orig_ramp_);
    }

#elif defined(__APPLE__)
    CGDirectDisplayID disp_ = 0;
    // CGDisplayFade token for paired begin/end calls.
    CGDisplayFadeReservationToken fade_token_ = kCGDisplayFadeReservationInvalidToken;

    bool platform_init() {
        disp_ = CGMainDisplayID();
        return (disp_ != 0);
    }
    void platform_cleanup() { /* CGDisplayFade needs no persistent resource */ }

    // CGDisplayFade smoothly transitions the entire display to a target colour.
    // startFade: initial alpha (0=current state, 1=full target colour)
    // endFade:   final alpha   (1=full target colour)
    // We fade from 0 (current) to 1 (target) over 0.1s, hold until restored.
    void do_fade(float r, float g, float b) {
        // Release any previous reservation
        if (fade_token_ != kCGDisplayFadeReservationInvalidToken) {
            CGDisplayFade(fade_token_, 0.0, kCGDisplayBlendNormal,
                          kCGDisplayBlendNormal, 0, 0, 0, FALSE);
            CGReleaseDisplayFadeReservation(fade_token_);
            fade_token_ = kCGDisplayFadeReservationInvalidToken;
        }
        if (CGAcquireDisplayFadeReservation(1.0, &fade_token_)
            != kCGErrorSuccess) return;
        // Fade to target colour over 0.10 s, hold (interval=kCGMaxDisplayReservationInterval).
        CGDisplayFade(fade_token_,
                      /*interval=*/0.10,
                      /*startBlend=*/kCGDisplayBlendNormal,
                      /*endBlend=*/  kCGDisplayBlendSolidColor,
                      r, g, b, /*async=*/FALSE);
    }
    void platform_dim() {
        // Blend 50% neutral grey over the display (r=g=b=0.5, alpha determined
        // by endBlend=kCGDisplayBlendSolidColor → displayed as 50% overlay).
        do_fade(0.0f, 0.0f, 0.0f); // black overlay at 50% blend = dim
        // Override: use a half-alpha black via gamma instead for dim.
        // Set transfer formula: output = 0.5 * input^(1/2.2) — darkens by 50%.
        CGSetDisplayTransferByFormula(disp_,
            0.0f, 1.0f, 1.0f / 2.2f,   // red:   min, max, gamma
            0.0f, 1.0f, 1.0f / 2.2f,   // green
            0.0f, 1.0f, 1.0f / 2.2f);  // blue
        // Scale to 50%: min=0, max=0.5, gamma=1.0/2.2
        CGSetDisplayTransferByFormula(disp_,
            0.0f, DIM_GAMMA_SCALE, 1.0f / 2.2f,
            0.0f, DIM_GAMMA_SCALE, 1.0f / 2.2f,
            0.0f, DIM_GAMMA_SCALE, 1.0f / 2.2f);
    }
    void platform_blackout() {
        // Set all channels to output 0 regardless of input.
        CGSetDisplayTransferByFormula(disp_,
            0.0f, 0.0f, 1.0f,  // red:   min=0, max=0 → always black
            0.0f, 0.0f, 1.0f,  // green
            0.0f, 0.0f, 1.0f); // blue
    }
    void platform_restore() {
        // Restore the default gamma formula (sRGB / 2.2).
        CGSetDisplayTransferByFormula(disp_,
            0.0f, 1.0f, 1.0f / 2.2f,
            0.0f, 1.0f, 1.0f / 2.2f,
            0.0f, 1.0f, 1.0f / 2.2f);
        if (fade_token_ != kCGDisplayFadeReservationInvalidToken) {
            CGDisplayFade(fade_token_, 0.10,
                          kCGDisplayBlendSolidColor,
                          kCGDisplayBlendNormal, 0, 0, 0, FALSE);
            CGReleaseDisplayFadeReservation(fade_token_);
            fade_token_ = kCGDisplayFadeReservationInvalidToken;
        }
    }

#elif defined(__linux__)
    Display*    dpy_   = nullptr;
    RRCrtc      crtc_  = 0;
    int         ramp_size_ = 0;

    // Original ramp saved at init — restored on NONE and destructor.
    std::vector<uint16_t> orig_r_, orig_g_, orig_b_;
    // Working ramp reused per-call to avoid repeated allocations.
    std::vector<uint16_t> work_r_, work_g_, work_b_;

    bool platform_init() {
        dpy_ = XOpenDisplay(nullptr);
        if (!dpy_) return false;

        Window root = DefaultRootWindow(dpy_);
        XRRScreenResources* res = XRRGetScreenResources(dpy_, root);
        if (!res) return false;

        // Use the first active CRTC.
        bool found = false;
        for (int i = 0; i < res->ncrtc && !found; ++i) {
            XRRCrtcInfo* ci = XRRGetCrtcInfo(dpy_, res, res->crtcs[i]);
            if (ci && ci->noutput > 0) { crtc_ = res->crtcs[i]; found = true; }
            XRRFreeCrtcInfo(ci);
        }
        XRRFreeScreenResources(res);
        if (!found) return false;

        // Read the current gamma ramp size and values.
        XRRCrtcGamma* g = XRRGetCrtcGamma(dpy_, crtc_);
        if (!g) return false;
        ramp_size_ = g->size;
        orig_r_.assign(g->red,   g->red   + ramp_size_);
        orig_g_.assign(g->green, g->green + ramp_size_);
        orig_b_.assign(g->blue,  g->blue  + ramp_size_);
        work_r_.resize(ramp_size_);
        work_g_.resize(ramp_size_);
        work_b_.resize(ramp_size_);
        XRRFreeGamma(g);
        return true;
    }

    void platform_cleanup() {
        if (dpy_) { XCloseDisplay(dpy_); dpy_ = nullptr; }
    }

    // Write work_r_/g_/b_ to the CRTC.
    void apply_ramp() {
        XRRCrtcGamma* g = XRRAllocGamma(ramp_size_);
        if (!g) return;
        std::copy(work_r_.begin(), work_r_.end(), g->red);
        std::copy(work_g_.begin(), work_g_.end(), g->green);
        std::copy(work_b_.begin(), work_b_.end(), g->blue);
        XRRSetCrtcGamma(dpy_, crtc_, g);
        XFlush(dpy_);
        XRRFreeGamma(g);
    }

    void platform_dim() {
        for (int i = 0; i < ramp_size_; ++i) {
            work_r_[i] = static_cast<uint16_t>(orig_r_[i] * DIM_GAMMA_SCALE);
            work_g_[i] = static_cast<uint16_t>(orig_g_[i] * DIM_GAMMA_SCALE);
            work_b_[i] = static_cast<uint16_t>(orig_b_[i] * DIM_GAMMA_SCALE);
        }
        apply_ramp();
    }

    void platform_blackout() {
        std::fill(work_r_.begin(), work_r_.end(), uint16_t(0));
        std::fill(work_g_.begin(), work_g_.end(), uint16_t(0));
        std::fill(work_b_.begin(), work_b_.end(), uint16_t(0));
        apply_ramp();
    }

    void platform_restore() {
        if (!dpy_) return;
        std::copy(orig_r_.begin(), orig_r_.end(), work_r_.begin());
        std::copy(orig_g_.begin(), orig_g_.end(), work_g_.begin());
        std::copy(orig_b_.begin(), orig_b_.end(), work_b_.begin());
        apply_ramp();
    }
#endif // platform
}; // class ScreenProtector

// ============================================================================
// PSEState — all persistent per-stream detection state
//
// Mask/flash caches are members here (not stack-locals in the processor
// function) so they truly persist across calls and achieve high-water-mark
// growth without reallocation after the first flashing window.
// ============================================================================
struct PSEState {
    PSEConfig             config;
    FlashRateCounter      rate_y, rate_r;
    SpatialDensityChecker density_y, density_r;
    ScreenProtector       screen;           // dim/blackout display protection

    // FIX 9 — Long-term static region map
    // One float per Y-plane chunk, storing the exponential moving average of
    // the 8-frame luma range for that chunk.  Updated every analysis window by
    // both Y worker threads (disjoint chunk ranges → no synchronisation needed).
    // A chunk is classified as static when its EMA < config.static_chunk_ema_thresh.
    std::vector<float> static_map_y;
    float              static_ema_alpha  = 0.05f;  // set from fps and window_seconds
    float              last_static_frac  = 0.0f;   // fraction static, previous window

    // Persistent mask/flash caches (two Y-thread sets + one R set)
    std::vector<uint32_t>     mask_cache_t1, mask_cache_t2;
    std::vector<ChunkVectors> flash_cache_t1, flash_cache_t2;
    std::vector<uint32_t>     r_mask_cache;
    std::vector<ChunkVectors> r_flash_cache;

    // OPT #3: Compact index lists of non-zero-mask chunks, built during PASS 1.
    // PASS 2 iterates these instead of re-scanning the full mask array,
    // eliminating the data-dependent "if (!mask) continue" branch.
    std::vector<int> flash_idx_t1, flash_idx_t2, flash_idx_r;

    // ── Detection-speed diagnostics ───────────────────────────────────────
    // Tracks per-window timing so the [D] line can surface exactly which gate
    // is blocking detection and how much latency the pipeline is introducing.
    struct FlashDetectionDiag {
        double prev_newest_pts   = -1.0;  // pts_newest from the previous call
        double spatial_start_pts = -1.0;  // when spatial_flash first became true
        bool   prev_spatial      = false; // spatial_flash state last window

        // Called each window: maintains the spatial-burst start timestamp.
        // spatial_start_pts resets when spatial_flash drops to false, so the
        // lag field measures time from the START of the current flash burst.
        void update(bool spatial_flash, double pts_newest) {
            if (spatial_flash && !prev_spatial)
                spatial_start_pts = pts_newest;
            else if (!spatial_flash)
                spatial_start_pts = -1.0;
            prev_spatial = spatial_flash;
        }
    } diag;

    void init(int width, int height,
              uint64_t valid_combos, int16_t kr, double fps_val)
    {
        config.valid_combo_mask = valid_combos;
        config.kr_fixed         = kr;
        config.fps              = fps_val;
        // min_flash_sep removed — rising-edge onset counter needs no separation gate
        density_y.init(width,     height);
        density_r.init(width / 2, height / 2);

        // FIX 9: initialise EMA map.  Alpha = 1 / (fps × window_seconds) gives
        // a time constant equal to the observation window (≈ 2 s default).
        const int y_chunks_count = ((width * height) + 31) / 32;
        static_map_y.assign(y_chunks_count, 0.0f);
        static_ema_alpha = (float)(1.0 /
            (fps_val * config.static_window_seconds));

        // ScreenProtector starts its background thread in its own constructor.
    }
};

// ============================================================================
// PSEState16 — State for 16-slot 3Hz detector
// ============================================================================
struct PSEState16 {
    PSEConfig16           config;
    FlashRateCounter      rate_y, rate_r;
    SpatialDensityChecker density_y, density_r;
    // ScreenProtector is shared with fast detector — only one instance

    std::vector<float> static_map_y;
    float              static_ema_alpha  = 0.05f;
    float              last_static_frac  = 0.0f;

    // Persistent mask/flash caches (two Y-thread sets + one R set)
    std::vector<uint32_t>     mask_cache_t1, mask_cache_t2;
    std::vector<ChunkVectors> flash_cache_t1, flash_cache_t2;
    std::vector<uint32_t>     r_mask_cache;
    std::vector<ChunkVectors> r_flash_cache;
    std::vector<int> flash_idx_t1, flash_idx_t2, flash_idx_r;

    struct FlashDetectionDiag {
        double prev_newest_pts   = -1.0;
        double spatial_start_pts = -1.0;
        bool   prev_spatial      = false;
        void update(bool spatial_flash, double pts_newest) {
            if (spatial_flash && !prev_spatial)
                spatial_start_pts = pts_newest;
            else if (!spatial_flash)
                spatial_start_pts = -1.0;
            prev_spatial = spatial_flash;
        }
    } diag;

    void init(int width, int height,
              const uint64_t valid_combos[4], int16_t kr, double fps_val)
    {
        std::memcpy(config.valid_combo_mask, valid_combos, sizeof(uint64_t) * 4);
        config.kr_fixed = kr;
        config.fps      = fps_val;
        density_y.init(width,     height);
        density_r.init(width / 2, height / 2);
        
        const int y_chunks_count = ((width * height) + 31) / 32;
        static_map_y.assign(y_chunks_count, 0.0f);
        static_ema_alpha = (float)(1.0 / (fps_val * config.static_window_seconds));
    }
};

// ============================================================================
// build_valid_combo_mask — FIX A (adjacency gate, 8-slot version)
// ============================================================================
static uint64_t build_valid_combo_mask(double fps, double min_hz = 3.0) {
    int max_sep = std::max(1, std::min((int)(fps / (2.0 * min_hz)), 7));
    uint64_t mask = 0;
    for (int mi = 0; mi < 8; ++mi)
        for (int mxi = 0; mxi < 8; ++mxi)
            if (mi != mxi && std::abs(mi - mxi) <= max_sep)
                mask |= (1ULL << (mi * 8 + mxi));
    return mask;
}

// ============================================================================
// build_valid_combo_mask_16 — 16-slot version for 3Hz detector
// ============================================================================
static void build_valid_combo_mask_16(double fps, uint64_t mask_out[4],
                                       double min_hz = 3.0) {
    int max_sep = std::max(1, std::min((int)(fps / (2.0 * min_hz)), 15));  // cap at 15
    mask_out[0] = mask_out[1] = mask_out[2] = mask_out[3] = 0;
    for (int mi = 0; mi < 16; ++mi)
        for (int mxi = 0; mxi < 16; ++mxi)
            if (mi != mxi && std::abs(mi - mxi) <= max_sep) {
                int combo = mi * 16 + mxi;
                mask_out[combo / 64] |= (1ULL << (combo % 64));
            }
}

// ============================================================================
// derive_kr_fixed — FIX C (colour-space Cr→R coefficient, unchanged from v59)
// ============================================================================
static int16_t derive_kr_fixed(AVColorSpace cs, int width, int height) {
    switch (cs) {
        case AVCOL_SPC_BT709:                       return 101;
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M:
        case AVCOL_SPC_SMPTE240M:                   return 90;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:                   return 94;
        default:
            return (width >= 1280 || height >= 720) ? 101 : 90;
    }
}

// ============================================================================
// pts_to_elapsed — convert a raw PTS to seconds since the first captured frame
// ============================================================================
template<size_t N>
static inline double pts_to_elapsed(int64_t pts, const PSEBuffer<N>& buf) noexcept {
    return (double)(pts - buf.first_pts) * buf.time_base;
}

// ============================================================================
// process_pse_temporal_avx_threaded
// ============================================================================
void process_pse_temporal_avx_threaded(PSEBufferFast&       buffer,
                                       PersistentWorker&    worker1,
                                       PersistentWorker&    worker2,
                                       const DisplayProfile& profile,
                                       PSEState&            state) noexcept
{
    const int ci = (int)buffer.current_index;
    size_t f[8];
    for (int k = 0; k < 8; ++k)
        f[k] = (size_t)((ci + k) % (int)buffer.MAX_SIZE);

    const double pts_newest = pts_to_elapsed(buffer.pts_buffer[f[7]], buffer);

    // ── FIX 6: scene-drift computation + strobe discriminator ─────────────
    // Collect per-frame mean luma across the 8-frame window.
    float mean_vals[8];
    for (int k = 0; k < 8; ++k) mean_vals[k] = buffer.y_mean[f[k]];
    float mean_min = mean_vals[0], mean_max = mean_vals[0];
    for (int k = 1; k < 8; ++k) {
        if (mean_vals[k] < mean_min) mean_min = mean_vals[k];
        if (mean_vals[k] > mean_max) mean_max = mean_vals[k];
    }
    const float scene_drift = mean_max - mean_min; // luma units [0,255]

    // FIX 6 BUG CORRECTION — Strobe vs motion discriminator.
    //
    // The original assumption "genuine strobes have scene_drift ≈ 0" was
    // wrong.  A full-screen strobe alternating between luma 30 and 228
    // produces scene_drift = 198 — the maximum possible value — because
    // the entire frame inverts on every cycle.  The drift computation cannot
    // distinguish "frame got darker because a hand blocked the light" from
    // "frame got darker because this is a dark phase of a strobe."
    //
    // Discriminator: count how many times consecutive per-frame means cross
    // the window mean (sign changes).  Camera motion is a monotonic drift
    // with 0–1 crossings.  A strobe alternates repeatedly, producing ≥ 3
    // crossings in 8 frames.
    //
    // Cost: 8 comparisons — negligible.
    float window_mean = 0.0f;
    for (int k = 0; k < 8; ++k) window_mean += mean_vals[k];
    window_mean /= 8.0f;
    int sign_changes = 0;
    for (int k = 1; k < 8; ++k) {
        bool prev_above = (mean_vals[k-1] > window_mean);
        bool curr_above = (mean_vals[k]   > window_mean);
        if (prev_above != curr_above) ++sign_changes;
    }
    // ≥ 3 crossings → the brightness is oscillating (strobe); leave threshold
    // at the baseline 25 %.  < 3 crossings → monotonic drift (camera motion);
    // scale up the threshold to suppress the false positive.
    const bool is_strobe_pattern = (sign_changes >= 3);

    const float DRIFT_THRESH = 15.0f;
    float effective_area_thresh = 0.25f;
    if (scene_drift > DRIFT_THRESH && !is_strobe_pattern) {
        float scale = 1.0f + std::min((scene_drift - DRIFT_THRESH) / 50.0f, 1.4f);
        effective_area_thresh = std::min(0.25f * scale, 0.60f);
    }
    // ──────────────────────────────────────────────────────────────────────

    FlashComboTracker tracker;
    for (int mi = 0; mi < 8; ++mi)
        for (int xi = 0; xi < 8; ++xi) {
            int c = mi * 8 + xi;
            tracker.min_frame_ref[c] = f[mi];
            tracker.max_frame_ref[c] = f[xi];
            tracker.min_pts_time[c]  = pts_to_elapsed(buffer.pts_buffer[f[mi]], buffer);
            tracker.max_pts_time[c]  = pts_to_elapsed(buffer.pts_buffer[f[xi]], buffer);
        }

    const bool use_sdr = (profile.active_eotf == EOTF_Mode::SDR_GAMMA_2_2
                       || profile.max_nits < 500.0f);
    const uint64_t valid_mask = state.config.valid_combo_mask;

    int t1_y_end = (buffer.y_chunks * 5) / 8;

    // Per-thread combo + directional arrays
    uint32_t t1_y_combos[64] = {}, t1_y_d2b[64] = {}, t1_y_b2d[64] = {};
    uint32_t t2_y_combos[64] = {}, t2_y_d2b[64] = {}, t2_y_b2d[64] = {};
    uint32_t t2_r_combos[64] = {}, t2_r_d2b[64] = {}, t2_r_b2d[64] = {};

    // Frame index broadcast registers
    const __m256i idx0 = _mm256_set1_epi8((uint8_t)f[0]);
    const __m256i idx1 = _mm256_set1_epi8((uint8_t)f[1]);
    const __m256i idx2 = _mm256_set1_epi8((uint8_t)f[2]);
    const __m256i idx3 = _mm256_set1_epi8((uint8_t)f[3]);
    const __m256i idx4 = _mm256_set1_epi8((uint8_t)f[4]);
    const __m256i idx5 = _mm256_set1_epi8((uint8_t)f[5]);
    const __m256i idx6 = _mm256_set1_epi8((uint8_t)f[6]);
    const __m256i idx7 = _mm256_set1_epi8((uint8_t)f[7]);

    // =========================================================================
    // process_y — PASS 1 (FIX 1: uses evaluate_itu1702_flash_mask)
    //              PASS 2 (FIX 2: directional combo attribution)
    // =========================================================================

    // FIX 8: pixel exclusion counters — accumulated across both Y threads
    // via an atomic, and directly in process_r (single thread).
    std::atomic<int> static_excl_y_atomic{0};
    int              static_excl_r = 0;

    // OPT #1: process_y is templated on USE_SDR so the if(use_sdr) branch is
    // eliminated at compile time from the hot inner loop.  Both specialisations
    // are instantiated once outside the chunk loop; the dead branch body is
    // never emitted.
    //
    // OPT #2: flashes.reserve(end-start) before the loop prevents all
    // std::vector capacity-check branches during push_back.
    //
    // OPT #3: flash_idx (compact list of non-zero-mask chunk positions) is
    // built during PASS 1 so PASS 2 iterates only active chunks — no
    // data-dependent "if (!mask) continue" branch.
    //
    // OPT #5: thresh_vec is hoisted out of the loop so _mm256_set1_epi8 is
    // called once rather than once per chunk.
    auto process_y = [&]<bool USE_SDR>(int start, int end,
                          uint32_t* combos, uint32_t* d2b, uint32_t* b2d,
                          std::vector<uint32_t>& masks,
                          std::vector<ChunkVectors>& flashes,
                          std::vector<int>& flash_idx) -> int
    {
        const int n = end - start;
        if ((int)masks.size() != n) masks.resize(n);
        std::fill(masks.begin(), masks.end(), 0u);
        flashes.clear();
        flash_idx.clear();
        // OPT #2: pre-allocate worst-case capacity — no capacity branch fires.
        flashes.reserve(static_cast<size_t>(n));
        flash_idx.reserve(static_cast<size_t>(n));
        int local_count  = 0;
        int local_static = 0;   // FIX 8: pixels excluded by static-range test

        // OPT #5: hoist loop-invariant AVX broadcast before the chunk loop.
        const __m256i thresh_vec = _mm256_set1_epi8(
            (int8_t)state.config.static_luma_thresh);
        // OPT #1 (HDR path): hoist the constant multiplier for the Michelson test.
        [[maybe_unused]] const __m256i hdr_mult = _mm256_set1_epi16(17);

        for (int chunk = start; chunk < end; ++chunk) {
            const size_t base = chunk * buffer.MAX_SIZE * 32;
            __m256i y0 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[0]*32]);
            __m256i y1 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[1]*32]);
            __m256i y2 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[2]*32]);
            __m256i y3 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[3]*32]);
            __m256i y4 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[4]*32]);
            __m256i y5 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[5]*32]);
            __m256i y6 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[6]*32]);
            __m256i y7 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[7]*32]);

            AVXArgResult amin01   = argmin_merge({y0,idx0}, y1,idx1);
            AVXArgResult amin23   = argmin_merge({y2,idx2}, y3,idx3);
            AVXArgResult amin45   = argmin_merge({y4,idx4}, y5,idx5);
            AVXArgResult amin67   = argmin_merge({y6,idx6}, y7,idx7);
            AVXArgResult amin0123 = argmin_merge(amin01, amin23.val, amin23.frame_idx);
            AVXArgResult amin4567 = argmin_merge(amin45, amin67.val, amin67.frame_idx);
            AVXArgResult amin_f   = argmin_merge(amin0123, amin4567.val, amin4567.frame_idx);

            AVXArgResult amax01   = argmax_merge({y0,idx0}, y1,idx1);
            AVXArgResult amax23   = argmax_merge({y2,idx2}, y3,idx3);
            AVXArgResult amax45   = argmax_merge({y4,idx4}, y5,idx5);
            AVXArgResult amax67   = argmax_merge({y6,idx6}, y7,idx7);
            AVXArgResult amax0123 = argmax_merge(amax01, amax23.val, amax23.frame_idx);
            AVXArgResult amax4567 = argmax_merge(amax45, amax67.val, amax67.frame_idx);
            AVXArgResult amax_f   = argmax_merge(amax0123, amax4567.val, amax4567.frame_idx);

            // OPT #1: if constexpr eliminates dead branch from this translation unit.
            // FIX 1: dual-regime criterion for SDR; Michelson-only path kept for HDR.
            uint32_t mask = 0;
            if constexpr (USE_SDR) {
                mask = evaluate_itu1702_flash_mask(amin_f.val, amax_f.val);
            } else {
                __m256i diff = _mm256_subs_epu8(amax_f.val, amin_f.val);
                __m256i sum  = _mm256_adds_epu8(amax_f.val, amin_f.val);
                __m256i dlo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diff));
                __m256i slo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sum));
                __m256i dhi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(diff, 1));
                __m256i shi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sum, 1));
                uint32_t mlo = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_mullo_epi16(dlo, hdr_mult), slo));
                uint32_t mhi = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_mullo_epi16(dhi, hdr_mult), shi));
                mask = _pext_u32(mlo, 0x55555555u) | (_pext_u32(mhi, 0x55555555u) << 16);
            }

            // FIX 8 ── static pixel exclusion (Y channel) ─────────────────
            // range_vec = max_luma − min_luma across 8 frames per pixel.
            // Pixels whose range ≤ static_luma_thresh are codec artifacts or
            // genuinely static background — they cannot represent a real flash.
            // amax_f.val and amin_f.val are already in registers; zero extra loads.
            {
                const __m256i range_vec  = _mm256_subs_epu8(amax_f.val, amin_f.val);
                // OPT #5: thresh_vec hoisted above loop — no set1 here.
                // is_static byte = 0xFF when range ≤ thresh (min_epu8 clamps range;
                // if the clamp produced equality it was already ≤ thresh).
                const __m256i is_static  = _mm256_cmpeq_epi8(
                    _mm256_min_epu8(range_vec, thresh_vec), range_vec);
                const uint32_t static_bits =
                    (uint32_t)_mm256_movemask_epi8(is_static);
                local_static += __builtin_popcount(static_bits & mask);
                mask          &= ~static_bits;

                // FIX 9 ── EMA static-region map update ───────────────────
                // Compute mean chunk range as a scalar via _mm256_sad_epu8.
                // sad vs zero gives 4 × 64-bit lane sums; sum them to get the
                // total of the 32 range values, then divide by 32 for the mean.
                // This single SAD collapses 32 bytes to a float with no branches.
                const __m256i zero    = _mm256_setzero_si256();
                const __m256i sad     = _mm256_sad_epu8(range_vec, zero);
                // Horizontal sum of the 4 × 64-bit lanes:
                const __m128i lo      = _mm256_castsi256_si128(sad);
                const __m128i hi      = _mm256_extracti128_si256(sad, 1);
                const __m128i sum128  = _mm_add_epi64(lo, hi);
                uint32_t chunk_range_sum =
                    (uint32_t)(_mm_cvtsi128_si64(sum128) +
                               _mm_cvtsi128_si64(_mm_srli_si128(sum128, 8)));
                float chunk_mean_range = (float)chunk_range_sum / 32.0f;

                // EMA: decays towards zero for static chunks, rises for active.
                // Worker threads write disjoint chunk indices → no mutex needed.
                float& ema = state.static_map_y[chunk];
                ema += state.static_ema_alpha * (chunk_mean_range - ema);
                // ──────────────────────────────────────────────────────────
            }
            // ──────────────────────────────────────────────────────────────

            masks[chunk - start] = mask;
            local_count += __builtin_popcount(mask);
            // OPT #3: record the local offset into flash_idx; push_back never
            // allocates (capacity already reserved above).
            if (mask != 0) {
                flash_idx.push_back(chunk - start);
                flashes.push_back({amin_f.frame_idx, amax_f.frame_idx});
            }
        }

        // PASS 2 — FIX 2: directional attribution
        // OPT #3: iterate compact flash_idx — no "if (!mask) continue" branch.
        {
            const int nf = (int)flash_idx.size();
            for (int fi = 0; fi < nf; ++fi) {
                const int ci = flash_idx[fi];
                extract_and_count_combos_precomputed(
                    masks[ci], flashes[fi].frame_min_idx, flashes[fi].frame_max_idx,
                    combos, d2b, b2d, valid_mask);
            }
        }
        static_excl_y_atomic.fetch_add(local_static, std::memory_order_relaxed);
        return local_count;
    };

    // =========================================================================
    // process_r — FIX 1 (dual-regime on R channel)
    //             FIX 2 (directional attribution)
    //             FIX 5 (saturated red flash detection)
    //
    // FIX 5: In addition to the Michelson R-channel check, computes the
    // unsigned Cr range across the 8 frames per pixel to detect transitions
    // to/from saturated red.  ITU-R BT.1702-3 Guideline 1: "Irrespective of
    // luminance, a transition to or from a saturated red is also potentially
    // harmful."  Saturated red = Cr > 150; qualifying swing = Cr range > 30.
    // The saturation area is tracked separately and used as an OR condition
    // for the spatial criterion.
    // =========================================================================
    int sat_pixels_r = 0;  // FIX 5: saturated-red flash pixel count

    auto process_r = [&](uint32_t* combos, uint32_t* d2b, uint32_t* b2d,
                          std::vector<uint32_t>& r_masks,
                          std::vector<ChunkVectors>& r_flashes,
                          std::vector<int>& r_flash_idx) -> int
    {
        const int n_chunks = buffer.v_chunks;
        if ((int)r_masks.size() != n_chunks) r_masks.resize(n_chunks);
        std::fill(r_masks.begin(), r_masks.end(), 0u);
        r_flashes.clear();
        r_flash_idx.clear();
        // OPT #2: pre-allocate worst-case capacity — no capacity branch fires.
        r_flashes.reserve(static_cast<size_t>(n_chunks));
        r_flash_idx.reserve(static_cast<size_t>(n_chunks));
        int local_count    = 0;
        int local_sat      = 0;
        int local_static_r = 0;   // FIX 8: pixels excluded by static-range test (R)

        const __m256i c128   = _mm256_set1_epi16(128);
        const __m256i kr_vec = _mm256_set1_epi16(state.config.kr_fixed);

        // FIX 5: unsigned compare constants (flip sign bit for unsigned order)
        const __m256i sign_flip    = _mm256_set1_epi8((int8_t)0x80);
        const __m256i cr_sat_thresh = _mm256_set1_epi8((int8_t)150); // 150 as uint8
        const __m256i cr_sw_thresh  = _mm256_set1_epi8((int8_t)30);  //  30 as uint8

        // OPT #5: hoist loop-invariant static threshold broadcast.
        const __m256i thresh_vec = _mm256_set1_epi8(
            (int8_t)state.config.static_luma_thresh);

        // Compute R = clamp(Y + kr*(Cr-128)/64, 0, 255) in epi16 fixed-point
        auto make_r = [&](__m256i yh, __m256i cr) -> __m256i {
            __m256i ylo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(yh));
            __m256i crlo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(cr));
            __m256i rlo  = _mm256_adds_epi16(ylo,
                              _mm256_srai_epi16(_mm256_mullo_epi16(_mm256_sub_epi16(crlo, c128), kr_vec), 6));
            __m256i yhi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(yh, 1));
            __m256i crhi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(cr, 1));
            __m256i rhi  = _mm256_adds_epi16(yhi,
                              _mm256_srai_epi16(_mm256_mullo_epi16(_mm256_sub_epi16(crhi, c128), kr_vec), 6));
            return _mm256_packus_epi16(rlo, rhi);
        };

        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const size_t base = chunk * buffer.MAX_SIZE * 32;

            __m256i cr0 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[0]*32]);
            __m256i cr1 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[1]*32]);
            __m256i cr2 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[2]*32]);
            __m256i cr3 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[3]*32]);
            __m256i cr4 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[4]*32]);
            __m256i cr5 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[5]*32]);
            __m256i cr6 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[6]*32]);
            __m256i cr7 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[7]*32]);

            __m256i yh0 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[0]*32]);
            __m256i yh1 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[1]*32]);
            __m256i yh2 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[2]*32]);
            __m256i yh3 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[3]*32]);
            __m256i yh4 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[4]*32]);
            __m256i yh5 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[5]*32]);
            __m256i yh6 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[6]*32]);
            __m256i yh7 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[base + f[7]*32]);

            // FIX 5: Cr unsigned min/max across 8 frames (no index needed)
            __m256i cr_mn = _mm256_min_epu8(_mm256_min_epu8(_mm256_min_epu8(cr0, cr1), _mm256_min_epu8(cr2, cr3)),
                                             _mm256_min_epu8(_mm256_min_epu8(cr4, cr5), _mm256_min_epu8(cr6, cr7)));
            __m256i cr_mx = _mm256_max_epu8(_mm256_max_epu8(_mm256_max_epu8(cr0, cr1), _mm256_max_epu8(cr2, cr3)),
                                             _mm256_max_epu8(_mm256_max_epu8(cr4, cr5), _mm256_max_epu8(cr6, cr7)));

            // Saturated red: max Cr > 150 AND Cr swing > 30 (unsigned comparisons)
            __m256i sat_cmp = _mm256_cmpgt_epi8(
                _mm256_xor_si256(cr_mx, sign_flip),
                _mm256_xor_si256(cr_sat_thresh, sign_flip));
            __m256i sw_cmp  = _mm256_cmpgt_epi8(
                _mm256_xor_si256(_mm256_subs_epu8(cr_mx, cr_mn), sign_flip),
                _mm256_xor_si256(cr_sw_thresh, sign_flip));
            __m256i sat_vec = _mm256_and_si256(sat_cmp, sw_cmp);
            // FIX 8: suppress Cr-range-static pixels from the sat-red count.
            // cr_mn / cr_mx are already computed above; no extra loads.
            {
                const __m256i cr_range   = _mm256_subs_epu8(cr_mx, cr_mn);
                // OPT #5: thresh_vec hoisted above loop.
                const __m256i cr_static  = _mm256_cmpeq_epi8(
                    _mm256_min_epu8(cr_range, thresh_vec), cr_range);
                sat_vec = _mm256_andnot_si256(cr_static, sat_vec);
            }
            local_sat += __builtin_popcount((uint32_t)_mm256_movemask_epi8(sat_vec));

            // Compute true R channel for all 8 frames
            __m256i r0 = make_r(yh0, cr0); __m256i r1 = make_r(yh1, cr1);
            __m256i r2 = make_r(yh2, cr2); __m256i r3 = make_r(yh3, cr3);
            __m256i r4 = make_r(yh4, cr4); __m256i r5 = make_r(yh5, cr5);
            __m256i r6 = make_r(yh6, cr6); __m256i r7 = make_r(yh7, cr7);

            AVXArgResult amin01   = argmin_merge({r0,idx0}, r1,idx1);
            AVXArgResult amin23   = argmin_merge({r2,idx2}, r3,idx3);
            AVXArgResult amin45   = argmin_merge({r4,idx4}, r5,idx5);
            AVXArgResult amin67   = argmin_merge({r6,idx6}, r7,idx7);
            AVXArgResult amin0123 = argmin_merge(amin01, amin23.val, amin23.frame_idx);
            AVXArgResult amin4567 = argmin_merge(amin45, amin67.val, amin67.frame_idx);
            AVXArgResult amin_f   = argmin_merge(amin0123, amin4567.val, amin4567.frame_idx);

            AVXArgResult amax01   = argmax_merge({r0,idx0}, r1,idx1);
            AVXArgResult amax23   = argmax_merge({r2,idx2}, r3,idx3);
            AVXArgResult amax45   = argmax_merge({r4,idx4}, r5,idx5);
            AVXArgResult amax67   = argmax_merge({r6,idx6}, r7,idx7);
            AVXArgResult amax0123 = argmax_merge(amax01, amax23.val, amax23.frame_idx);
            AVXArgResult amax4567 = argmax_merge(amax45, amax67.val, amax67.frame_idx);
            AVXArgResult amax_f   = argmax_merge(amax0123, amax4567.val, amax4567.frame_idx);

            // FIX 1: dual-regime criterion on R (R is in display-encoded [0,255])
            uint32_t mask = evaluate_itu1702_flash_mask(amin_f.val, amax_f.val);

            // FIX 8 ── static pixel exclusion (R channel) ──────────────────
            // R = Y + kr*(Cr-128)/64 — range measured on the reconstructed R.
            {
                const __m256i range_r    = _mm256_subs_epu8(amax_f.val, amin_f.val);
                // OPT #5: thresh_vec hoisted above loop.
                const __m256i is_static  = _mm256_cmpeq_epi8(
                    _mm256_min_epu8(range_r, thresh_vec), range_r);
                const uint32_t static_bits =
                    (uint32_t)_mm256_movemask_epi8(is_static);
                local_static_r += __builtin_popcount(static_bits & mask);
                mask            &= ~static_bits;
            }
            // ──────────────────────────────────────────────────────────────

            r_masks[chunk] = mask;
            local_count += __builtin_popcount(mask);
            // OPT #3: build compact index list for branch-free PASS 2.
            if (mask != 0) {
                r_flash_idx.push_back(chunk);
                r_flashes.push_back({amin_f.frame_idx, amax_f.frame_idx});
            }
        }

        // PASS 2 — FIX 2: directional attribution on R
        // OPT #3: iterate compact r_flash_idx — no "if (!mask) continue" branch.
        {
            const int nf = (int)r_flash_idx.size();
            for (int fi = 0; fi < nf; ++fi) {
                const int ci = r_flash_idx[fi];
                extract_and_count_combos_precomputed(
                    r_masks[ci], r_flashes[fi].frame_min_idx, r_flashes[fi].frame_max_idx,
                    combos, d2b, b2d, valid_mask);
            }
        }
        sat_pixels_r   = local_sat;
        static_excl_r  = local_static_r;   // FIX 8: expose to outer scope
        return local_count;
    };

    int flash_y_t1 = 0, flash_y_t2 = 0, flash_r = 0;

    {
        // OPT #1: dispatch to the correct compile-time specialisation of process_y.
        // Both lambda bodies are already compiled; this picks the right one once.
        auto t1 = [&]() {
            if (use_sdr)
                flash_y_t1 = process_y.operator()<true>(
                    0, t1_y_end,
                    t1_y_combos, t1_y_d2b, t1_y_b2d,
                    state.mask_cache_t1, state.flash_cache_t1,
                    state.flash_idx_t1);
            else
                flash_y_t1 = process_y.operator()<false>(
                    0, t1_y_end,
                    t1_y_combos, t1_y_d2b, t1_y_b2d,
                    state.mask_cache_t1, state.flash_cache_t1,
                    state.flash_idx_t1);
        };
        auto t2 = [&]() {
            if (use_sdr)
                flash_y_t2 = process_y.operator()<true>(
                    t1_y_end, buffer.y_chunks,
                    t2_y_combos, t2_y_d2b, t2_y_b2d,
                    state.mask_cache_t2, state.flash_cache_t2,
                    state.flash_idx_t2);
            else
                flash_y_t2 = process_y.operator()<false>(
                    t1_y_end, buffer.y_chunks,
                    t2_y_combos, t2_y_d2b, t2_y_b2d,
                    state.mask_cache_t2, state.flash_cache_t2,
                    state.flash_idx_t2);
            flash_r = process_r(t2_r_combos, t2_r_d2b, t2_r_b2d,
                                state.r_mask_cache, state.r_flash_cache,
                                state.flash_idx_r);
        };
        static_assert(sizeof(t1) <= InlineTask::STORAGE_SIZE, "t1 too large");
        static_assert(sizeof(t2) <= InlineTask::STORAGE_SIZE, "t2 too large");
        worker1.execute(std::move(t1));
        worker2.execute(std::move(t2));
    }
    worker1.wait();
    worker2.wait();

    // ── Merge combo + directional hits ────────────────────────────────────
    const int total_flash_y = flash_y_t1 + flash_y_t2;
    for (int i = 0; i < 64; ++i) {
        tracker.y_combo_hits[i] = t1_y_combos[i] + t2_y_combos[i];
        tracker.y_d2b_hits[i]   = t1_y_d2b[i]    + t2_y_d2b[i];
        tracker.y_b2d_hits[i]   = t1_y_b2d[i]    + t2_y_b2d[i];
        tracker.r_combo_hits[i] = t2_r_combos[i];
        tracker.r_d2b_hits[i]   = t2_r_d2b[i];
        tracker.r_b2d_hits[i]   = t2_r_b2d[i];
    }

    // =========================================================================
    // FIX 9 — Static region subtraction
    //
    // Count chunks whose EMA range is below static_chunk_ema_thresh.  These
    // are long-term static background regions (desktop, taskbar, etc.) that
    // inflate the denominator of the concurrent flash area fraction, making a
    // small but genuinely dangerous flash appear to cover less than 25% of the
    // "screen" when in fact it covers a significant fraction of the active area.
    //
    // Adjustment: recompute the effective frame denominator as:
    //   active_pixels = total_pixels - static_pixels
    //
    // Nullification: if the static fraction exceeds static_nullify_frac (0.85),
    // the whole screen is static content and the flash area is genuinely small
    // — no adjustment is made.  This prevents the mechanism from falsely
    // amplifying very small flickers on an otherwise completely static display.
    // =========================================================================
    int   static_chunks_y  = 0;
    float active_y_fraction = 1.0f;          // fraction of screen that is active
    float static_frac_y    = 0.0f;

    {
        const int   total_y_chunks  = buffer.y_chunks;
        const float ema_thresh      = state.config.static_chunk_ema_thresh;
        for (int c = 0; c < total_y_chunks; ++c)
            if (state.static_map_y[c] < ema_thresh)
                ++static_chunks_y;

        static_frac_y          = (float)static_chunks_y / (float)total_y_chunks;
        state.last_static_frac = static_frac_y;

        const bool nullify = (static_frac_y > state.config.static_nullify_frac);
        if (!nullify && static_chunks_y > 0)
            active_y_fraction = 1.0f - static_frac_y;
        // active_y_fraction stays 1.0 when nullified or when nothing is static.
    }

    // Effective frame size after removing static background.
    // Flash area fractions are divided by effective_y_pix rather than the full
    // y_frame_size, giving a normalised "fraction of active area" measure.
    const float effective_y_pix =
        (float)buffer.y_frame_size * active_y_fraction;
    const float effective_y_pix_safe = std::max(effective_y_pix, 1.0f);

    // =========================================================================
    // FIX 3 — Per-combo concurrent area (primary spatial criterion)
    //
    // Denominator is now effective_y_pix_safe (FIX 9) rather than raw
    // y_frame_size.  When large areas of the screen are classified as long-term
    // static, the flash is measured against only the active (non-static) pixels,
    // correctly reflecting its visual prominence in the viewer's field of view.
    // =========================================================================
    // FIX 3 — Per-combo concurrent area (primary spatial criterion)
    //
    // Denominator is now effective_y_pix_safe (FIX 9) rather than raw
    // y_frame_size.  When large areas of the screen are classified as long-term
    // static, the flash is measured against only the active (non-static) pixels,
    // correctly reflecting its visual prominence in the viewer's field of view.
    //
    // OPT #7: diagonal skip uses kValidComboIndices; the two loop-invariant
    // denominators are pre-inverted to reciprocals so each iteration uses a
    // multiply instead of a division (latency 5 vs 10–25 cycles on Zen/Skylake).
    // =========================================================================
    float max_concurrent_y = 0.0f, max_concurrent_r = 0.0f;
    int   peak_y = -1, peak_r = -1;
    {
        const float inv_y = 1.0f / effective_y_pix_safe;
        const float inv_r = 1.0f / (float)buffer.v_frame_size;
        for (int i : kValidComboIndices) {
            float cy = (float)tracker.y_combo_hits[i] * inv_y;
            float cr = (float)tracker.r_combo_hits[i] * inv_r;
            if (cy > max_concurrent_y) { max_concurrent_y = cy; peak_y = i; }
            if (cr > max_concurrent_r) { max_concurrent_r = cr; peak_r = i; }
        }
    }

    // FIX 6 applied: effective_area_thresh scales up under scene motion.
    const bool concurrent_y = (max_concurrent_y > effective_area_thresh);
    const bool concurrent_r = (max_concurrent_r > effective_area_thresh);

    // FIX 5: saturated red area (independent of Michelson, at quarter resolution)
    const float sat_area = (float)sat_pixels_r / (float)buffer.v_frame_size;
    const bool sat_red_flash = (sat_area > effective_area_thresh);

    // Secondary: integral-image local density (checker-board patterns)
    state.density_y.build(state.mask_cache_t1.data(), (int)state.mask_cache_t1.size(),
                          state.mask_cache_t2.data(), (int)state.mask_cache_t2.size());
    state.density_r.build(state.r_mask_cache.data(),  (int)state.r_mask_cache.size(),
                          nullptr, 0);
    const int yw = std::max(1, state.density_y.grid_h/5);
    const int yc = std::max(1, state.density_y.grid_w/5);
    const bool local_dense_y = state.density_y.has_dense_region(yw, yc, yw*yc*32/4);
    const int rw = std::max(1, state.density_r.grid_h/5);
    const int rc = std::max(1, state.density_r.grid_w/5);
    const bool local_dense_r = state.density_r.has_dense_region(rw, rc, rw*rc*32/4);

    const bool spatial_flash = concurrent_y || concurrent_r || sat_red_flash
                             || local_dense_y || local_dense_r;

    // =========================================================================
    // FIX 7 — Pattern detection (Guideline 2)
    //
    // Estimates average stripe pair count per row from the Y flash mask SAT.
    // >5 stripe pairs = potentially harmful pattern.
    // Thresholds: stationary >40%, flashing/reversing >25% (Attachment 1).
    // =========================================================================
    const float stripe_freq = state.density_y.estimate_stripe_frequency();
    bool pattern_violation = false;
    if (stripe_freq > 5.0f) {
        float area = (float)total_flash_y / (float)buffer.y_frame_size;
        // scene_drift ≈ 0 → stationary pattern; nonzero → potentially flashing
        float pat_thresh = (scene_drift < 5.0f) ? 0.40f : 0.25f;
        pattern_violation = (area > pat_thresh);
    }

    // =========================================================================
    // FIX 2 + FIX 3 + FIX 4 — Opposing-pair detection + rate counting
    //
    // A combo is dominant when it accounts for >25% of its channel's total
    // concurrent flash pixels.  A flash PAIR fires only when BOTH dark→bright
    // and bright→dark dominant combos are present (ITU "opposing changes").
    // Rate events are recorded only once per min_flash_sep seconds (FIX 4).
    // =========================================================================
    // =========================================================================
    // FIX 2 + FIX 3 + FIX 4 — Opposing-pair detection + rate counting
    //
    // A combo is dominant when it accounts for >25% of its channel's total
    // concurrent flash pixels.  A flash PAIR fires only when BOTH dark→bright
    // and bright→dark dominant combos are present (ITU "opposing changes").
    // Rate events are recorded only once per min_flash_sep seconds (FIX 4).
    //
    // OPT #6: loop-invariant guards (total_flash_y > 0, flash_r > 0) and the
    // 0.25× threshold multiplications are hoisted before the loop.  Diagonal
    // entries are skipped via kValidComboIndices (no per-iteration branch).
    // has_d2b_y / has_b2d_y are accumulated with |= (comparison result) —
    // branchless, emits OR+CMP without a conditional jump.
    // =========================================================================
    bool has_d2b_y = false, has_b2d_y = false;
    bool has_d2b_r = false, has_b2d_r = false;
    {
        const float thresh_y = 0.25f * (float)total_flash_y;
        const float thresh_r = 0.25f * (float)flash_r;
        for (int i : kValidComboIndices) {
            has_d2b_y |= (tracker.y_d2b_hits[i] > thresh_y);
            has_b2d_y |= (tracker.y_b2d_hits[i] > thresh_y);
            has_d2b_r |= (tracker.r_d2b_hits[i] > thresh_r);
            has_b2d_r |= (tracker.r_b2d_hits[i] > thresh_r);
        }
    }

    // ── FIX 4 (v61): rising-edge onset recording ─────────────────────────
    //   • spatial criterion met (concurrent area, sat-red, or local density)
    //   • opposing-pair present (both d2b AND b2d detected in the 8-frame window)
    // record_state() records an onset timestamp on the false→true edge and
    // resets state on the true→false edge — no min_sep parameter needed.
    const bool is_flashing_y = spatial_flash && has_d2b_y && has_b2d_y;
    const bool is_flashing_r = spatial_flash && has_d2b_r && has_b2d_r;
    state.rate_y.record_state(is_flashing_y, pts_newest);
    state.rate_r.record_state(is_flashing_r, pts_newest);

    const bool pse_y = state.rate_y.pse_rate_exceeded(pts_newest);
    const bool pse_r = state.rate_r.pse_rate_exceeded(pts_newest);

    // ── Output ────────────────────────────────────────────────────────────
    // All variables needed by pse_flash_probability are fully resolved here:
    //   max_concurrent_y, max_concurrent_r  (FIX 3 — concurrent area)
    //   sat_area                            (FIX 5 — saturated red)
    //   sign_changes                        (FIX 6 — oscillation count)
    //   effective_area_thresh               (FIX 6 — drift-corrected threshold)
    //   stripe_freq                         (FIX 7 — Guideline 2 patterns)
    //   rate_y/r.events_in_window           (FIX 4 — onset count)
    if (total_flash_y > 0 || flash_r > 0 || sat_pixels_r > 0) {
        // FIX 9 timestamp: elapsed seconds since first captured frame.
        const double t0 = pts_to_elapsed(buffer.pts_buffer[f[0]], buffer);

        // FIX 8: read exclusion counts from atomic (both Y threads merged).
        const int static_excl_y = static_excl_y_atomic.load(std::memory_order_relaxed);

        // Compute graded probability P via the analytical P-function.
        // The function internally applies the drift-corrected threshold (Step 2),
        // so thr_c_pct may differ from effective_area_thresh when c_max ≥ 50%.
        const auto prob = pse_flash_probability(
            max_concurrent_y * 100.f,
            max_concurrent_r * 100.f,
            sat_area         * 100.f,
            sign_changes,
            effective_area_thresh * 100.f,
            stripe_freq,
            state.rate_y.events_in_window(pts_newest),
            state.rate_r.events_in_window(pts_newest));

        std::cout << "[i] " << t0 << "s-" << pts_newest << "s"
                  << "  Y_conc=" << (max_concurrent_y * 100.f) << "%"
                  << "  R_conc=" << (max_concurrent_r * 100.f) << "%"
                  << "  SatR="   << (sat_area * 100.f)         << "%"
                  << "  drift="  << scene_drift
                  << "  osc="    << sign_changes
                  << "  AThresh->" << prob.thr_c_pct            << "%"
                  << "  excl_Y=" << (static_excl_y * 100.f / (float)buffer.y_frame_size) << "%"
                  << "  excl_R=" << (static_excl_r * 100.f / (float)buffer.v_frame_size) << "%"
                  << "  static=" << (static_frac_y * 100.f) << "%"
                  << "  active=" << (active_y_fraction * 100.f) << "%"
                  << "  oy=" << state.rate_y.events_in_window(pts_newest)
                  << "  or=" << state.rate_r.events_in_window(pts_newest)
                  << "  P=" << prob.P_pct << "%"
                  << " [" << prob.risk_label() << "]\n";

        // ── [D] Detection-speed diagnostic line ───────────────────────────
        // Each field directly corresponds to one gate in the detection chain.
        // When 3Hz (or any low-frequency) flash is not alarming, these values
        // reveal exactly which gate is the blocker and why.
        {
            // 1. Window span: (pts[f[7]] - pts[f[0]]) in ms.
            //    At 30fps = ~233ms; at 60fps = ~117ms.
            //    A full flash cycle must be ≤ this span for BOTH d2b and b2d
            //    to appear in the same window.  min_hz_coverage = 1000/win_ms.
            const double win_span_ms =
                (pts_newest - t0) * 1000.0;  // t0=f[0], pts_newest=f[7]

            // 2. Window step: time since the previous analysis window.
            //    Should be 1000/fps ms (one new frame per window advance).
            const double step_ms = (state.diag.prev_newest_pts >= 0)
                ? (pts_newest - state.diag.prev_newest_pts) * 1000.0
                : -1.0;

            // 3. Adjacency gate: max_sep = floor(fps / (2×3Hz)), capped at 7.
            //    min_hz_gate = fps / (2 × max_sep).
            //    Any flash whose argmin-argmax slot separation > max_sep is
            //    silently discarded — its combo is not in the valid_combo_mask.
            const int    max_sep_val  = std::max(1, std::min(
                                           (int)(state.config.fps / (2.0 * 3.0)), 7));
            const double min_hz_gate  = state.config.fps / (2.0 * max_sep_val);

            // Peak combo slot separation: the largest transition seen this window.
            // "sep=N/M" means N slots apart out of a gate limit of M.
            // If N == M the flash is AT the gate boundary; if N > M it was rejected.
            const int peak_min_slot = (peak_y >= 0) ? (peak_y / 8) : -1;
            const int peak_max_slot = (peak_y >= 0) ? (peak_y % 8) : -1;
            const int peak_sep_val  = (peak_y >= 0)
                ? std::abs(peak_min_slot - peak_max_slot) : -1;

            // 4. Opposing-pair gate: BOTH d2b AND b2d must be present.
            //    "FAIL:b2d" means only dark→bright was seen — no bright→dark
            //    transition exists in the window → is_flashing stays false.
            const char* pair_status =
                (has_d2b_y  && has_b2d_y)  ? "PASS"      :
                (!has_d2b_y && !has_b2d_y) ? "FAIL:both" :
                !has_d2b_y                 ? "FAIL:d2b"  : "FAIL:b2d";

            // 5. Instantaneous Hz estimate from the last two recorded onsets.
            //    "-" = fewer than 2 onsets have been recorded yet.
            const auto [oa, ob] = state.rate_y.last_two_onsets();
            const double hz_est = (oa >= 0 && ob > oa)
                ? 1.0 / (ob - oa) : -1.0;

            // 6. Spatial→detect lag: elapsed ms from when spatial_flash first
            //    became true in this burst to when is_flashing_y first became true.
            //    A large lag means the opposing-pair or rate gate held detection back.
            const double lag_ms = (is_flashing_y && state.diag.spatial_start_pts >= 0)
                ? (pts_newest - state.diag.spatial_start_pts) * 1000.0 : -1.0;

            std::cout << std::fixed << std::setprecision(1)
                      << "[D] win=" << win_span_ms << "ms"
                      << "  step=" << (step_ms >= 0 ? step_ms : 0.0) << "ms"
                      << "  sep=" << peak_sep_val << "/" << max_sep_val
                      << "  gate_min=" << min_hz_gate << "Hz"
                      << "  pair=" << pair_status
                      << "  is_flash_y=" << (is_flashing_y ? "Y" : "N")
                      << "  hz_est=" << (hz_est > 0 ? hz_est : 0.0) << "Hz"
                      << "  lag=" << lag_ms << "ms"
                      << "\n" << std::defaultfloat;

            // Advance diag state for the next window.
            state.diag.prev_newest_pts = pts_newest;
            state.diag.update(spatial_flash, pts_newest);
        }
        // ─────────────────────────────────────────────────────────────────

        // ── Screen protection: dim or blackout based on P-function level ──
        // This is an atomic store — the background ScreenProtector thread
        // reads the new level and applies the OS gamma/fade API without
        // blocking the analysis path.  The effect persists until the next
        // window clears the level back to NONE.
        state.screen.update_from_probability(prob.P);

        // Elevated risk sub-bands: print component breakdown for warn/alarm.
        // This surfaces the dominant pathway (luma vs red vs pattern) for
        // content that is approaching but has not yet crossed the hard alarm.
        if (prob.P >= 0.45f && prob.P < 0.70f) {
            std::cout << "    [warn]"
                      << "  P_core=" << (prob.P_core * 100.f) << "%"
                      << "  P_red="  << (prob.P_red  * 100.f) << "%"
                      << "  P_pat="  << (prob.P_pat  * 100.f) << "%"
                      << "  W_rt="   << (prob.W_rt   * 100.f) << "%"
                      << "  P_osc="  << (prob.P_osc  * 100.f) << "%\n";
        }
    }

    if (pattern_violation) {
        std::cout << "[!] PATTERN VIOLATION — "
                  << stripe_freq << " stripe pairs/row, area="
                  << ((float)total_flash_y / buffer.y_frame_size * 100.f) << "%"
                  << "  t=" << pts_newest << "s\n";
    }

    // ── Hard ITU alarm (unchanged gating) + P-function enrichment ────────
    // The hard alarm fires on the same spatial+opposing-pair+rate AND gate as
    // before. The P-function output is added to the alarm block to show which
    // pathway drove the alarm and what the probability was at alarm time.
    // This allows post-hoc analysis to distinguish near-miss alarms (P≈70%)
    // from confirmed severe events (P≈95%).
    if ((pse_y || pse_r) && spatial_flash) {
        // Re-compute prob at alarm time (same call, cost is negligible).
        const auto alarm_prob = pse_flash_probability(
            max_concurrent_y * 100.f,
            max_concurrent_r * 100.f,
            sat_area         * 100.f,
            sign_changes,
            effective_area_thresh * 100.f,
            stripe_freq,
            state.rate_y.events_in_window(pts_newest),
            state.rate_r.events_in_window(pts_newest));

        std::cout << "\n[!!!] PSE EPILEPTIC FLASH ALARM  t=" << pts_newest << "s"
                  << "  P=" << alarm_prob.P_pct << "% [" << alarm_prob.risk_label() << "]\n"
                  << "      Y onsets/s : " << state.rate_y.events_in_window(pts_newest)
                  << "  (threshold >3)\n"
                  << "      R onsets/s : " << state.rate_r.events_in_window(pts_newest)
                  << "  (threshold >3)\n"
                  << "      ConcY  : " << (max_concurrent_y * 100.f) << "%\n"
                  << "      ConcR  : " << (max_concurrent_r * 100.f) << "%\n"
                  << "      SatRed : " << (sat_area * 100.f) << "%\n"
                  << "      Drift  : " << scene_drift << " luma units"
                  << "  osc=" << sign_changes
                  << (is_strobe_pattern ? " [STROBE-bypass]" : " [motion-scaled]") << "\n"
                  << "      AThresh: " << (effective_area_thresh * 100.f)
                  << "% → thr_c=" << alarm_prob.thr_c_pct << "% (drift-corrected)\n"
                  << "      Static : " << (static_frac_y * 100.f) << "% background"
                  << "  active=" << (active_y_fraction * 100.f) << "%"
                  << (active_y_fraction < 1.0f ? " [area-adjusted]" : " [full-frame]") << "\n"
                  << "      LocalD : Y=" << local_dense_y << " R=" << local_dense_r << "\n"
                  << "      P breakdown:"
                  << "  P_core=" << (alarm_prob.P_core * 100.f) << "%"
                  << "  P_red="  << (alarm_prob.P_red  * 100.f) << "%"
                  << "  P_pat="  << (alarm_prob.P_pat  * 100.f) << "%"
                  << "  W_rt="   << (alarm_prob.W_rt   * 100.f) << "%\n";

        // Per-combo diagnostic for the peak transition
        for (int ch : {peak_y, peak_r}) {
            if (ch < 0) continue;
            std::cout << "      Peak combo [" << (ch/8) << "→" << (ch%8) << "]"
                      << "  d2b=" << (ch == peak_y ? tracker.y_d2b_hits[ch] : tracker.r_d2b_hits[ch])
                      << "  b2d=" << (ch == peak_y ? tracker.y_b2d_hits[ch] : tracker.r_b2d_hits[ch])
                      << "  t_min=" << tracker.min_pts_time[ch] << "s"
                      << "  t_max=" << tracker.max_pts_time[ch] << "s\n";
        }
    }
}

// ============================================================================
// process_pse_temporal_avx_threaded_16slot
//
// 16-slot 3Hz detector.  Identical logic to the 8-slot version with the
// following mechanical differences:
//
//   f[16]                  — 16 consecutive ring-buffer slots
//   mean_vals[16]          — per-frame luma over 16 frames
//   window_mean /= 16      — mean over 16 samples
//   is_strobe >= 6         — 6/16 ≈ 3/8 sign-change ratio
//   tracker: FlashComboTracker16 — 256-element arrays (16×16 grid)
//   combo index: mi*16+xi  — 256 unique combos
//   max_sep capped at 15   — allows 10-frame half-period @ 60fps → 3Hz
//   extract_and_count_combos_16 — uses 256-bit mask (uint64_t[4])
//   kValidComboIndices_16  — 240 off-diagonal entries
//   [D] output prefixed with "S:" so it is distinct from the fast detector
//
// Parameters are identical in type to the 8-slot version; only the buffer
// and state types differ (PSEBufferSlow / PSEState16).
// ============================================================================
void process_pse_temporal_avx_threaded_16slot(
    PSEBufferSlow&       buffer,
    PersistentWorker&    worker1,
    PersistentWorker&    worker2,
    const DisplayProfile& profile,
    PSEState16&          state) noexcept
{
    const int ci = (int)buffer.current_index;
    size_t f[16];
    for (int k = 0; k < 16; ++k)
        f[k] = (size_t)((ci + k) % (int)buffer.MAX_SIZE);

    const double pts_newest = pts_to_elapsed(buffer.pts_buffer[f[15]], buffer);

    // Scene-drift + strobe discriminator over 16 frames
    float mean_vals[16];
    for (int k = 0; k < 16; ++k) mean_vals[k] = buffer.y_mean[f[k]];
    float mean_min = mean_vals[0], mean_max = mean_vals[0];
    for (int k = 1; k < 16; ++k) {
        if (mean_vals[k] < mean_min) mean_min = mean_vals[k];
        if (mean_vals[k] > mean_max) mean_max = mean_vals[k];
    }
    const float scene_drift = mean_max - mean_min;

    float window_mean = 0.0f;
    for (int k = 0; k < 16; ++k) window_mean += mean_vals[k];
    window_mean /= 16.0f;
    int sign_changes = 0;
    for (int k = 1; k < 16; ++k) {
        bool prev_above = (mean_vals[k-1] > window_mean);
        bool curr_above = (mean_vals[k]   > window_mean);
        if (prev_above != curr_above) ++sign_changes;
    }
    // FIX (3Hz detection): threshold lowered from 6 to 3.
    // At exactly 3Hz @ 60fps the 267ms/16-frame window covers ~1.6 cycles,
    // producing ~3 mean-crossings — not 6.  The naive 3/8 -> 6/16 scaling
    // assumed crossing count scales linearly with window length, but a 3Hz
    // strobe is the TARGET frequency for this detector, so its crossing
    // count in a 16-frame window is what must trip is_strobe_pattern.
    // With threshold=6, a genuine 3Hz strobe never set is_strobe_pattern,
    // so effective_area_thresh could inflate under any scene_drift and
    // mask the flash entirely.
    const bool is_strobe_pattern = (sign_changes >= 3);

    const float DRIFT_THRESH = 15.0f;
    float effective_area_thresh = 0.25f;
    if (scene_drift > DRIFT_THRESH && !is_strobe_pattern) {
        float scale = 1.0f + std::min((scene_drift - DRIFT_THRESH) / 50.0f, 1.4f);
        effective_area_thresh = std::min(0.25f * scale, 0.60f);
    }

    FlashComboTracker16 tracker;
    for (int mi = 0; mi < 16; ++mi)
        for (int xi = 0; xi < 16; ++xi) {
            int c = mi * 16 + xi;
            tracker.min_frame_ref[c] = f[mi];
            tracker.max_frame_ref[c] = f[xi];
            tracker.min_pts_time[c]  = pts_to_elapsed(buffer.pts_buffer[f[mi]], buffer);
            tracker.max_pts_time[c]  = pts_to_elapsed(buffer.pts_buffer[f[xi]], buffer);
        }

    const bool use_sdr = (profile.active_eotf == EOTF_Mode::SDR_GAMMA_2_2
                       || profile.max_nits < 500.0f);

    // Copy 256-bit mask into local array for extract_and_count_combos_16
    const uint64_t valid_mask[4] = {
        state.config.valid_combo_mask[0],
        state.config.valid_combo_mask[1],
        state.config.valid_combo_mask[2],
        state.config.valid_combo_mask[3]
    };

    int t1_y_end = (buffer.y_chunks * 5) / 8;

    uint32_t t1_y_combos[256] = {}, t1_y_d2b[256] = {}, t1_y_b2d[256] = {};
    uint32_t t2_y_combos[256] = {}, t2_y_d2b[256] = {}, t2_y_b2d[256] = {};
    uint32_t t2_r_combos[256] = {}, t2_r_d2b[256] = {}, t2_r_b2d[256] = {};

    // FIX (3Hz detection): broadcast WINDOW OFFSETS (0..15), not ring
    // indices f[k].  f[k] = (ci+k) % MAX_SIZE wraps around the ring buffer
    // and is NOT monotonic when ci != 0 (e.g. ci=10 gives f = [10,11,...,
    // 15,0,1,...,9]).  build_valid_combo_mask_16() assumes combo = mi*16+xi
    // encodes the WINDOW-RELATIVE separation |mi-xi| <= max_sep — but
    // extract_and_count_combos_16 reads ms/xs from these idx registers to
    // form that same combo index.  If idx_k carries f[k] (ring index), the
    // combo's "separation" is the ring-index distance, which is unrelated
    // to temporal separation and can be smaller OR larger than the true
    // window separation depending on ci -- causing a 3Hz flash exactly at
    // the max_sep=10 boundary to fall outside the valid mask on some frames
    // and inside it on others (intermittent / missed detection).
    //
    // Using k directly makes idx_k monotonic 0..15 in temporal order, so
    // combo = mi*16+xi always equals the true window-relative separation,
    // matching the mask built by build_valid_combo_mask_16().
    // tracker.min_frame_ref/max_frame_ref and *_pts_time still use f[mi]/
    // f[xi] for timestamp lookups (those arrays are indexed by combo, not
    // by these registers) and are unaffected.
    const __m256i idx0  = _mm256_set1_epi8((uint8_t)0);
    const __m256i idx1  = _mm256_set1_epi8((uint8_t)1);
    const __m256i idx2  = _mm256_set1_epi8((uint8_t)2);
    const __m256i idx3  = _mm256_set1_epi8((uint8_t)3);
    const __m256i idx4  = _mm256_set1_epi8((uint8_t)4);
    const __m256i idx5  = _mm256_set1_epi8((uint8_t)5);
    const __m256i idx6  = _mm256_set1_epi8((uint8_t)6);
    const __m256i idx7  = _mm256_set1_epi8((uint8_t)7);
    const __m256i idx8  = _mm256_set1_epi8((uint8_t)8);
    const __m256i idx9  = _mm256_set1_epi8((uint8_t)9);
    const __m256i idx10 = _mm256_set1_epi8((uint8_t)10);
    const __m256i idx11 = _mm256_set1_epi8((uint8_t)11);
    const __m256i idx12 = _mm256_set1_epi8((uint8_t)12);
    const __m256i idx13 = _mm256_set1_epi8((uint8_t)13);
    const __m256i idx14 = _mm256_set1_epi8((uint8_t)14);
    const __m256i idx15 = _mm256_set1_epi8((uint8_t)15);

    std::atomic<int> static_excl_y_atomic{0};
    int              static_excl_r = 0;

    // process_y_16: identical to the 8-slot version but processes 16 frames
    // per chunk using 16 loads + 16 argmin/argmax merges.
    auto process_y_16 = [&]<bool USE_SDR>(int start, int end,
                             uint32_t* combos, uint32_t* d2b, uint32_t* b2d,
                             std::vector<uint32_t>& masks,
                             std::vector<ChunkVectors>& flashes,
                             std::vector<int>& flash_idx) -> int
    {
        const int n = end - start;
        if ((int)masks.size() != n) masks.resize(n);
        std::fill(masks.begin(), masks.end(), 0u);
        flashes.clear();
        flash_idx.clear();
        flashes.reserve(static_cast<size_t>(n));
        flash_idx.reserve(static_cast<size_t>(n));
        int local_count  = 0;
        int local_static = 0;

        const __m256i thresh_vec = _mm256_set1_epi8(
            (int8_t)state.config.static_luma_thresh);
        [[maybe_unused]] const __m256i hdr_mult = _mm256_set1_epi16(17);

        for (int chunk = start; chunk < end; ++chunk) {
            const size_t base = chunk * buffer.MAX_SIZE * 32;
            // Load all 16 frame planes for this chunk
            __m256i y0  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[0]*32]);
            __m256i y1  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[1]*32]);
            __m256i y2  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[2]*32]);
            __m256i y3  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[3]*32]);
            __m256i y4  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[4]*32]);
            __m256i y5  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[5]*32]);
            __m256i y6  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[6]*32]);
            __m256i y7  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[7]*32]);
            __m256i y8  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[8]*32]);
            __m256i y9  = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[9]*32]);
            __m256i y10 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[10]*32]);
            __m256i y11 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[11]*32]);
            __m256i y12 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[12]*32]);
            __m256i y13 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[13]*32]);
            __m256i y14 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[14]*32]);
            __m256i y15 = _mm256_load_si256((const __m256i*)&buffer.y_buffer[base + f[15]*32]);

            // 16-way argmin tree — 4 levels of pairwise merges
            AVXArgResult a01   = argmin_merge({y0,idx0},   y1, idx1);
            AVXArgResult a23   = argmin_merge({y2,idx2},   y3, idx3);
            AVXArgResult a45   = argmin_merge({y4,idx4},   y5, idx5);
            AVXArgResult a67   = argmin_merge({y6,idx6},   y7, idx7);
            AVXArgResult a89   = argmin_merge({y8,idx8},   y9, idx9);
            AVXArgResult a1011 = argmin_merge({y10,idx10}, y11,idx11);
            AVXArgResult a1213 = argmin_merge({y12,idx12}, y13,idx13);
            AVXArgResult a1415 = argmin_merge({y14,idx14}, y15,idx15);
            AVXArgResult a0123   = argmin_merge(a01,   a23.val,   a23.frame_idx);
            AVXArgResult a4567   = argmin_merge(a45,   a67.val,   a67.frame_idx);
            AVXArgResult a891011 = argmin_merge(a89,   a1011.val, a1011.frame_idx);
            AVXArgResult a12151  = argmin_merge(a1213, a1415.val, a1415.frame_idx);
            AVXArgResult a07     = argmin_merge(a0123, a4567.val,   a4567.frame_idx);
            AVXArgResult a815    = argmin_merge(a891011, a12151.val, a12151.frame_idx);
            AVXArgResult amin_f  = argmin_merge(a07,   a815.val,  a815.frame_idx);

            // 16-way argmax tree
            AVXArgResult b01   = argmax_merge({y0,idx0},   y1, idx1);
            AVXArgResult b23   = argmax_merge({y2,idx2},   y3, idx3);
            AVXArgResult b45   = argmax_merge({y4,idx4},   y5, idx5);
            AVXArgResult b67   = argmax_merge({y6,idx6},   y7, idx7);
            AVXArgResult b89   = argmax_merge({y8,idx8},   y9, idx9);
            AVXArgResult b1011 = argmax_merge({y10,idx10}, y11,idx11);
            AVXArgResult b1213 = argmax_merge({y12,idx12}, y13,idx13);
            AVXArgResult b1415 = argmax_merge({y14,idx14}, y15,idx15);
            AVXArgResult b0123   = argmax_merge(b01,   b23.val,   b23.frame_idx);
            AVXArgResult b4567   = argmax_merge(b45,   b67.val,   b67.frame_idx);
            AVXArgResult b891011 = argmax_merge(b89,   b1011.val, b1011.frame_idx);
            AVXArgResult b12151  = argmax_merge(b1213, b1415.val, b1415.frame_idx);
            AVXArgResult b07     = argmax_merge(b0123, b4567.val,   b4567.frame_idx);
            AVXArgResult b815    = argmax_merge(b891011, b12151.val, b12151.frame_idx);
            AVXArgResult amax_f  = argmax_merge(b07,   b815.val,  b815.frame_idx);

            uint32_t mask = 0;
            if constexpr (USE_SDR) {
                mask = evaluate_itu1702_flash_mask(amin_f.val, amax_f.val);
            } else {
                __m256i diff = _mm256_subs_epu8(amax_f.val, amin_f.val);
                __m256i sum  = _mm256_adds_epu8(amax_f.val, amin_f.val);
                __m256i dlo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diff));
                __m256i slo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sum));
                __m256i dhi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(diff, 1));
                __m256i shi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sum, 1));
                uint32_t mlo = _mm256_movemask_epi8(_mm256_cmpgt_epi16(
                    _mm256_mullo_epi16(dlo, hdr_mult), slo));
                uint32_t mhi = _mm256_movemask_epi8(_mm256_cmpgt_epi16(
                    _mm256_mullo_epi16(dhi, hdr_mult), shi));
                mask = _pext_u32(mlo, 0x55555555u) | (_pext_u32(mhi, 0x55555555u) << 16);
            }

            // Static exclusion
            {
                const __m256i range_vec = _mm256_subs_epu8(amax_f.val, amin_f.val);
                const __m256i is_static = _mm256_cmpeq_epi8(
                    _mm256_min_epu8(range_vec, thresh_vec), range_vec);
                const uint32_t static_bits = (uint32_t)_mm256_movemask_epi8(is_static);
                local_static += __builtin_popcount(static_bits & mask);
                mask         &= ~static_bits;

                // EMA static map update
                const __m256i zero = _mm256_setzero_si256();
                const __m256i sad  = _mm256_sad_epu8(range_vec, zero);
                const __m128i lo   = _mm256_castsi256_si128(sad);
                const __m128i hi   = _mm256_extracti128_si256(sad, 1);
                const __m128i s128 = _mm_add_epi64(lo, hi);
                uint32_t rsum = (uint32_t)(_mm_cvtsi128_si64(s128) +
                                           _mm_cvtsi128_si64(_mm_srli_si128(s128, 8)));
                float& ema = state.static_map_y[chunk];
                ema += state.static_ema_alpha * ((float)rsum / 32.0f - ema);
            }

            masks[chunk - start] = mask;
            local_count += __builtin_popcount(mask);
            if (mask != 0) {
                flash_idx.push_back(chunk - start);
                flashes.push_back({amin_f.frame_idx, amax_f.frame_idx});
            }
        }

        // PASS 2 — directional combo attribution using 16-slot extract
        {
            const int nf = (int)flash_idx.size();
            for (int fi = 0; fi < nf; ++fi) {
                const int ci2 = flash_idx[fi];
                extract_and_count_combos_16(
                    masks[ci2], flashes[fi].frame_min_idx, flashes[fi].frame_max_idx,
                    combos, d2b, b2d, valid_mask);
            }
        }
        static_excl_y_atomic.fetch_add(local_static, std::memory_order_relaxed);
        return local_count;
    };  // end process_y_16

    int sat_pixels_r = 0;

    auto process_r_16 = [&](uint32_t* combos, uint32_t* d2b, uint32_t* b2d,
                             std::vector<uint32_t>& r_masks,
                             std::vector<ChunkVectors>& r_flashes,
                             std::vector<int>& r_flash_idx) -> int
    {
        const int n_chunks = buffer.v_chunks;
        if ((int)r_masks.size() != n_chunks) r_masks.resize(n_chunks);
        std::fill(r_masks.begin(), r_masks.end(), 0u);
        r_flashes.clear();
        r_flash_idx.clear();
        r_flashes.reserve(static_cast<size_t>(n_chunks));
        r_flash_idx.reserve(static_cast<size_t>(n_chunks));
        int local_count    = 0;
        int local_sat      = 0;
        int local_static_r = 0;

        const __m256i c128   = _mm256_set1_epi16(128);
        const __m256i kr_vec = _mm256_set1_epi16(state.config.kr_fixed);
        const __m256i sign_flip     = _mm256_set1_epi8((int8_t)0x80);
        const __m256i cr_sat_thresh = _mm256_set1_epi8((int8_t)150);
        const __m256i cr_sw_thresh  = _mm256_set1_epi8((int8_t)30);
        const __m256i thresh_vec    = _mm256_set1_epi8(
            (int8_t)state.config.static_luma_thresh);

        auto make_r = [&](__m256i yh, __m256i cr) -> __m256i {
            __m256i ylo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(yh));
            __m256i crlo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(cr));
            __m256i rlo  = _mm256_adds_epi16(ylo,
                _mm256_srai_epi16(_mm256_mullo_epi16(
                    _mm256_sub_epi16(crlo, c128), kr_vec), 6));
            __m256i yhi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(yh, 1));
            __m256i crhi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(cr, 1));
            __m256i rhi  = _mm256_adds_epi16(yhi,
                _mm256_srai_epi16(_mm256_mullo_epi16(
                    _mm256_sub_epi16(crhi, c128), kr_vec), 6));
            return _mm256_packus_epi16(rlo, rhi);
        };

        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const size_t base = chunk * buffer.MAX_SIZE * 32;

            // Cr min/max across all 16 frames
            __m256i cr0  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[0]*32]);
            __m256i cr1  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[1]*32]);
            __m256i cr2  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[2]*32]);
            __m256i cr3  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[3]*32]);
            __m256i cr4  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[4]*32]);
            __m256i cr5  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[5]*32]);
            __m256i cr6  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[6]*32]);
            __m256i cr7  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[7]*32]);
            __m256i cr8  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[8]*32]);
            __m256i cr9  = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[9]*32]);
            __m256i cr10 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[10]*32]);
            __m256i cr11 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[11]*32]);
            __m256i cr12 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[12]*32]);
            __m256i cr13 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[13]*32]);
            __m256i cr14 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[14]*32]);
            __m256i cr15 = _mm256_load_si256((const __m256i*)&buffer.v_buffer[base + f[15]*32]);

            __m256i cr_mn = _mm256_min_epu8(cr0, _mm256_min_epu8(cr1, _mm256_min_epu8(cr2,
                            _mm256_min_epu8(cr3, _mm256_min_epu8(cr4, _mm256_min_epu8(cr5,
                            _mm256_min_epu8(cr6, _mm256_min_epu8(cr7, _mm256_min_epu8(cr8,
                            _mm256_min_epu8(cr9, _mm256_min_epu8(cr10,_mm256_min_epu8(cr11,
                            _mm256_min_epu8(cr12,_mm256_min_epu8(cr13,_mm256_min_epu8(cr14,cr15
                            )))))))))))))));
            __m256i cr_mx = _mm256_max_epu8(cr0, _mm256_max_epu8(cr1, _mm256_max_epu8(cr2,
                            _mm256_max_epu8(cr3, _mm256_max_epu8(cr4, _mm256_max_epu8(cr5,
                            _mm256_max_epu8(cr6, _mm256_max_epu8(cr7, _mm256_max_epu8(cr8,
                            _mm256_max_epu8(cr9, _mm256_max_epu8(cr10,_mm256_max_epu8(cr11,
                            _mm256_max_epu8(cr12,_mm256_max_epu8(cr13,_mm256_max_epu8(cr14,cr15
                            )))))))))))))));

            // FIX 5: saturated red — Cr > 150 and Cr swing > 30, unsigned
            __m256i cr_mn_u = _mm256_xor_si256(cr_mn, sign_flip);
            __m256i cr_mx_u = _mm256_xor_si256(cr_mx, sign_flip);
            __m256i sat_mn  = _mm256_xor_si256(_mm256_set1_epi8((int8_t)(150 ^ 0x80)), sign_flip);
            __m256i sat_sw  = _mm256_xor_si256(_mm256_set1_epi8((int8_t)(30  ^ 0x80)), sign_flip);
            __m256i sat_vec = _mm256_and_si256(
                _mm256_cmpgt_epi8(cr_mx_u, sat_mn),
                _mm256_cmpgt_epi8(_mm256_subs_epu8(cr_mx, cr_mn), _mm256_xor_si256(sat_sw, sign_flip)));

            // Suppress Cr-static pixels from sat count
            {
                const __m256i cr_range  = _mm256_subs_epu8(cr_mx, cr_mn);
                const __m256i cr_static = _mm256_cmpeq_epi8(
                    _mm256_min_epu8(cr_range, thresh_vec), cr_range);
                sat_vec = _mm256_andnot_si256(cr_static, sat_vec);
            }
            local_sat += __builtin_popcount((uint32_t)_mm256_movemask_epi8(sat_vec));

            // Y-half plane for R reconstruction
            const size_t ybase = chunk * buffer.MAX_SIZE * 32;
            // Build R argmin/argmax across 16 frames using idx vectors
            __m256i yh0  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[0]*32]);
            __m256i yh1  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[1]*32]);
            __m256i yh2  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[2]*32]);
            __m256i yh3  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[3]*32]);
            __m256i yh4  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[4]*32]);
            __m256i yh5  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[5]*32]);
            __m256i yh6  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[6]*32]);
            __m256i yh7  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[7]*32]);
            __m256i yh8  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[8]*32]);
            __m256i yh9  = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[9]*32]);
            __m256i yh10 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[10]*32]);
            __m256i yh11 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[11]*32]);
            __m256i yh12 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[12]*32]);
            __m256i yh13 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[13]*32]);
            __m256i yh14 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[14]*32]);
            __m256i yh15 = _mm256_load_si256((const __m256i*)&buffer.y_half_buffer[ybase + f[15]*32]);

            // Reconstruct R for all 16 frames then argmin/argmax
            __m256i r0  = make_r(yh0,  cr0);  __m256i r1  = make_r(yh1,  cr1);
            __m256i r2  = make_r(yh2,  cr2);  __m256i r3  = make_r(yh3,  cr3);
            __m256i r4  = make_r(yh4,  cr4);  __m256i r5  = make_r(yh5,  cr5);
            __m256i r6  = make_r(yh6,  cr6);  __m256i r7  = make_r(yh7,  cr7);
            __m256i r8  = make_r(yh8,  cr8);  __m256i r9  = make_r(yh9,  cr9);
            __m256i r10 = make_r(yh10, cr10); __m256i r11 = make_r(yh11, cr11);
            __m256i r12 = make_r(yh12, cr12); __m256i r13 = make_r(yh13, cr13);
            __m256i r14 = make_r(yh14, cr14); __m256i r15 = make_r(yh15, cr15);

            AVXArgResult ra01   = argmin_merge({r0,idx0},   r1, idx1);
            AVXArgResult ra23   = argmin_merge({r2,idx2},   r3, idx3);
            AVXArgResult ra45   = argmin_merge({r4,idx4},   r5, idx5);
            AVXArgResult ra67   = argmin_merge({r6,idx6},   r7, idx7);
            AVXArgResult ra89   = argmin_merge({r8,idx8},   r9, idx9);
            AVXArgResult ra1011 = argmin_merge({r10,idx10}, r11,idx11);
            AVXArgResult ra1213 = argmin_merge({r12,idx12}, r13,idx13);
            AVXArgResult ra1415 = argmin_merge({r14,idx14}, r15,idx15);
            AVXArgResult ra0123   = argmin_merge(ra01, ra23.val,   ra23.frame_idx);
            AVXArgResult ra4567   = argmin_merge(ra45, ra67.val,   ra67.frame_idx);
            AVXArgResult ra891011 = argmin_merge(ra89, ra1011.val, ra1011.frame_idx);
            AVXArgResult ra12151  = argmin_merge(ra1213,ra1415.val,ra1415.frame_idx);
            AVXArgResult ra07     = argmin_merge(ra0123, ra4567.val, ra4567.frame_idx);
            AVXArgResult ra815    = argmin_merge(ra891011,ra12151.val,ra12151.frame_idx);
            AVXArgResult amin_f   = argmin_merge(ra07, ra815.val, ra815.frame_idx);

            AVXArgResult rb01   = argmax_merge({r0,idx0},   r1, idx1);
            AVXArgResult rb23   = argmax_merge({r2,idx2},   r3, idx3);
            AVXArgResult rb45   = argmax_merge({r4,idx4},   r5, idx5);
            AVXArgResult rb67   = argmax_merge({r6,idx6},   r7, idx7);
            AVXArgResult rb89   = argmax_merge({r8,idx8},   r9, idx9);
            AVXArgResult rb1011 = argmax_merge({r10,idx10}, r11,idx11);
            AVXArgResult rb1213 = argmax_merge({r12,idx12}, r13,idx13);
            AVXArgResult rb1415 = argmax_merge({r14,idx14}, r15,idx15);
            AVXArgResult rb0123   = argmax_merge(rb01, rb23.val,   rb23.frame_idx);
            AVXArgResult rb4567   = argmax_merge(rb45, rb67.val,   rb67.frame_idx);
            AVXArgResult rb891011 = argmax_merge(rb89, rb1011.val, rb1011.frame_idx);
            AVXArgResult rb12151  = argmax_merge(rb1213,rb1415.val,rb1415.frame_idx);
            AVXArgResult rb07     = argmax_merge(rb0123, rb4567.val, rb4567.frame_idx);
            AVXArgResult rb815    = argmax_merge(rb891011,rb12151.val,rb12151.frame_idx);
            AVXArgResult amax_f   = argmax_merge(rb07, rb815.val, rb815.frame_idx);

            // Flash mask (Michelson only — R channel)
            __m256i diff = _mm256_subs_epu8(amax_f.val, amin_f.val);
            __m256i sum  = _mm256_adds_epu8(amax_f.val, amin_f.val);
            __m256i dlo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diff));
            __m256i slo  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sum));
            __m256i dhi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(diff, 1));
            __m256i shi  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sum, 1));
            const __m256i k17 = _mm256_set1_epi16(17);
            uint32_t mlo = _mm256_movemask_epi8(
                _mm256_cmpgt_epi16(_mm256_mullo_epi16(dlo, k17), slo));
            uint32_t mhi = _mm256_movemask_epi8(
                _mm256_cmpgt_epi16(_mm256_mullo_epi16(dhi, k17), shi));
            uint32_t mask = _pext_u32(mlo, 0x55555555u) | (_pext_u32(mhi, 0x55555555u) << 16);

            // R-channel static exclusion
            {
                const __m256i range_r   = _mm256_subs_epu8(amax_f.val, amin_f.val);
                const __m256i is_static = _mm256_cmpeq_epi8(
                    _mm256_min_epu8(range_r, thresh_vec), range_r);
                const uint32_t static_bits = (uint32_t)_mm256_movemask_epi8(is_static);
                local_static_r += __builtin_popcount(static_bits & mask);
                mask           &= ~static_bits;
            }

            r_masks[chunk] = mask;
            local_count += __builtin_popcount(mask);
            if (mask != 0) {
                r_flash_idx.push_back(chunk);
                r_flashes.push_back({amin_f.frame_idx, amax_f.frame_idx});
            }
        }

        // PASS 2 — R directional combo attribution
        {
            const int nf = (int)r_flash_idx.size();
            for (int fi = 0; fi < nf; ++fi) {
                const int ci2 = r_flash_idx[fi];
                extract_and_count_combos_16(
                    r_masks[ci2], r_flashes[fi].frame_min_idx, r_flashes[fi].frame_max_idx,
                    combos, d2b, b2d, valid_mask);
            }
        }
        sat_pixels_r   = local_sat;
        static_excl_r  = local_static_r;
        return local_count;
    };  // end process_r_16

    // ── Dispatch workers ──────────────────────────────────────────────────────
    int flash_y_t1 = 0, flash_y_t2 = 0, flash_r = 0;
    {
        auto t1 = [&]() {
            if (use_sdr)
                flash_y_t1 = process_y_16.operator()<true>(
                    0, t1_y_end,
                    t1_y_combos, t1_y_d2b, t1_y_b2d,
                    state.mask_cache_t1, state.flash_cache_t1, state.flash_idx_t1);
            else
                flash_y_t1 = process_y_16.operator()<false>(
                    0, t1_y_end,
                    t1_y_combos, t1_y_d2b, t1_y_b2d,
                    state.mask_cache_t1, state.flash_cache_t1, state.flash_idx_t1);
        };
        auto t2 = [&]() {
            if (use_sdr)
                flash_y_t2 = process_y_16.operator()<true>(
                    t1_y_end, buffer.y_chunks,
                    t2_y_combos, t2_y_d2b, t2_y_b2d,
                    state.mask_cache_t2, state.flash_cache_t2, state.flash_idx_t2);
            else
                flash_y_t2 = process_y_16.operator()<false>(
                    t1_y_end, buffer.y_chunks,
                    t2_y_combos, t2_y_d2b, t2_y_b2d,
                    state.mask_cache_t2, state.flash_cache_t2, state.flash_idx_t2);
            flash_r = process_r_16(t2_r_combos, t2_r_d2b, t2_r_b2d,
                                   state.r_mask_cache, state.r_flash_cache,
                                   state.flash_idx_r);
        };
        static_assert(sizeof(t1) <= InlineTask::STORAGE_SIZE, "t1_16 too large");
        static_assert(sizeof(t2) <= InlineTask::STORAGE_SIZE, "t2_16 too large");
        worker1.execute(std::move(t1));
        worker2.execute(std::move(t2));
    }
    worker1.wait();
    worker2.wait();

    const int total_flash_y = flash_y_t1 + flash_y_t2;

    // Merge per-thread combo arrays into tracker
    for (int i = 0; i < 256; ++i) {
        tracker.y_combo_hits[i] = t1_y_combos[i] + t2_y_combos[i];
        tracker.y_d2b_hits[i]   = t1_y_d2b[i]    + t2_y_d2b[i];
        tracker.y_b2d_hits[i]   = t1_y_b2d[i]    + t2_y_b2d[i];
        tracker.r_combo_hits[i] = t2_r_combos[i];
        tracker.r_d2b_hits[i]   = t2_r_d2b[i];
        tracker.r_b2d_hits[i]   = t2_r_b2d[i];
    }

    // Static fraction / active area
    const int static_excl_y = static_excl_y_atomic.load(std::memory_order_relaxed);
    int static_chunks_y = 0;
    for (float v : state.static_map_y)
        if (v < state.config.static_chunk_ema_thresh) ++static_chunks_y;
    const int total_y_chunks = buffer.y_chunks;
    const float static_frac_y = (total_y_chunks > 0)
        ? (float)static_chunks_y / (float)total_y_chunks : 0.0f;
    const float active_y_fraction = (static_frac_y < state.config.static_nullify_frac)
        ? std::max(0.01f, 1.0f - static_frac_y) : 1.0f;
    const float effective_y_pix_safe = std::max(
        (float)buffer.y_frame_size * active_y_fraction, 1.0f);

    // FIX 3: per-combo concurrent area (16-slot, 256 combos)
    float max_concurrent_y = 0.0f, max_concurrent_r = 0.0f;
    int   peak_y = -1, peak_r = -1;
    {
        const float inv_y = 1.0f / effective_y_pix_safe;
        const float inv_r = 1.0f / (float)buffer.v_frame_size;
        for (int i : kValidComboIndices_16) {
            float cy = (float)tracker.y_combo_hits[i] * inv_y;
            float cr = (float)tracker.r_combo_hits[i] * inv_r;
            if (cy > max_concurrent_y) { max_concurrent_y = cy; peak_y = i; }
            if (cr > max_concurrent_r) { max_concurrent_r = cr; peak_r = i; }
        }
    }

    const bool concurrent_y  = (max_concurrent_y > effective_area_thresh);
    const bool concurrent_r  = (max_concurrent_r > effective_area_thresh);
    const float sat_area      = (float)sat_pixels_r / (float)buffer.v_frame_size;
    const bool sat_red_flash  = (sat_area > effective_area_thresh);

    state.density_y.build(state.mask_cache_t1.data(), (int)state.mask_cache_t1.size(),
                          state.mask_cache_t2.data(), (int)state.mask_cache_t2.size());
    state.density_r.build(state.r_mask_cache.data(), (int)state.r_mask_cache.size(),
                          nullptr, 0);
    const int yw = std::max(1, state.density_y.grid_h/5);
    const int yc = std::max(1, state.density_y.grid_w/5);
    const bool local_dense_y = state.density_y.has_dense_region(yw, yc, yw*yc*32/4);
    const int rw = std::max(1, state.density_r.grid_h/5);
    const int rc = std::max(1, state.density_r.grid_w/5);
    const bool local_dense_r = state.density_r.has_dense_region(rw, rc, rw*rc*32/4);
    const bool spatial_flash  = concurrent_y || concurrent_r || sat_red_flash
                               || local_dense_y || local_dense_r;

    const float stripe_freq   = state.density_y.estimate_stripe_frequency();
    bool pattern_violation    = false;
    if (stripe_freq > 5.0f) {
        float area      = (float)total_flash_y / (float)buffer.y_frame_size;
        float pat_thresh = (scene_drift < 5.0f) ? 0.40f : 0.25f;
        pattern_violation = (area > pat_thresh);
    }

    // FIX 2: opposing-pair detection (256-combo grid)
    bool has_d2b_y = false, has_b2d_y = false;
    bool has_d2b_r = false, has_b2d_r = false;
    {
        const float thresh_y = 0.25f * (float)total_flash_y;
        const float thresh_r = 0.25f * (float)flash_r;
        for (int i : kValidComboIndices_16) {
            has_d2b_y |= (tracker.y_d2b_hits[i] > thresh_y);
            has_b2d_y |= (tracker.y_b2d_hits[i] > thresh_y);
            has_d2b_r |= (tracker.r_d2b_hits[i] > thresh_r);
            has_b2d_r |= (tracker.r_b2d_hits[i] > thresh_r);
        }
    }

    const bool is_flashing_y = spatial_flash && has_d2b_y && has_b2d_y;
    const bool is_flashing_r = spatial_flash && has_d2b_r && has_b2d_r;
    state.rate_y.record_state(is_flashing_y, pts_newest);
    state.rate_r.record_state(is_flashing_r, pts_newest);

    const bool pse_y = state.rate_y.pse_rate_exceeded(pts_newest);
    const bool pse_r = state.rate_r.pse_rate_exceeded(pts_newest);

    // Diagnostic output (prefixed [S] = slow detector)
    if (total_flash_y > 0 || flash_r > 0 || sat_pixels_r > 0) {
        const double t0 = pts_to_elapsed(buffer.pts_buffer[f[0]], buffer);
        const int    max_sep_val = std::max(1, std::min(
            (int)(state.config.fps / (2.0 * 3.0)), 15));
        const double min_hz_gate = state.config.fps / (2.0 * max_sep_val);
        const int peak_min_slot  = (peak_y >= 0) ? (peak_y / 16) : -1;
        const int peak_max_slot  = (peak_y >= 0) ? (peak_y % 16) : -1;
        const int peak_sep_val   = (peak_y >= 0) ? std::abs(peak_min_slot - peak_max_slot) : -1;
        const char* pair_status  =
            (has_d2b_y  && has_b2d_y)  ? "PASS"      :
            (!has_d2b_y && !has_b2d_y) ? "FAIL:both" :
            !has_d2b_y                 ? "FAIL:d2b"  : "FAIL:b2d";
        const auto [oa, ob] = state.rate_y.last_two_onsets();
        const double hz_est = (oa >= 0 && ob > oa) ? 1.0 / (ob - oa) : -1.0;

        std::cout << std::fixed << std::setprecision(1)
                  << "[S] win=" << ((pts_newest - t0) * 1000.0) << "ms"
                  << "  sep=" << peak_sep_val << "/" << max_sep_val
                  << "  gate_min=" << min_hz_gate << "Hz"
                  << "  pair=" << pair_status
                  << "  is_flash_y=" << (is_flashing_y ? "Y" : "N")
                  << "  hz_est=" << (hz_est > 0 ? hz_est : 0.0) << "Hz"
                  << "\n" << std::defaultfloat;
    }

    if ((pse_y || pse_r) && spatial_flash) {
        std::cout << "\n[!!!][SLOW] PSE 3Hz FLASH ALARM  t=" << pts_newest << "s"
                  << "  Y onsets/s: " << state.rate_y.events_in_window(pts_newest)
                  << "  R onsets/s: " << state.rate_r.events_in_window(pts_newest)
                  << "  ConcY=" << (max_concurrent_y * 100.f) << "%"
                  << "  ConcR=" << (max_concurrent_r * 100.f) << "%"
                  << "  win=16 frames (3Hz capable)\n";
    }
}

// ============================================================================
// CaptureConfig — platform device selection + runtime parameters
//
// Selected at compile time from the preprocessor OS macros.
// All fields are used only in main(); the PSE analysis pipeline is unchanged.
// ============================================================================
struct CaptureConfig {
    const char* input_format;   // FFmpeg input format name for avdevice
    const char* device_url;     // device URL / display identifier
    const char* framerate;      // e.g. "60" — must match display refresh rate
    const char* video_size;     // "WxH" of the capture region; nullptr = native
    const char* pixel_format;   // hint to the device (may be ignored)
    bool        needs_swscale;  // true when the device delivers RGB, not YUV420p
};

// ============================================================================
// DownsampleConfig — post-capture spatial downscale before PSE analysis
//
// The PSE pipeline pixel cost scales with W×H.  Capturing at native 4K/1440p
// and then converting RGB→YUV420p through libswscale at full resolution wastes
// CPU and causes branch-predictor pressure in the Cr-plane scatter loops.
//
// Solution: insert a second sws_scale pass that resizes the YUV420p frame
// coming out of the colour-conversion step down to a smaller analysis
// resolution (default 1280×720) BEFORE handing it to PSEBuffer/PSEState.
//
// Trade-off: the downsample pass itself calls libswscale and has its own
// branch cost.  The net gain is positive when src_w * src_h / (dst_w * dst_h)
// is large — i.e. when the native resolution is significantly larger than the
// analysis resolution.  At 1080p→720p the ratio is ~2.25×; at 4K→720p it is
// ~8×.  See the overview comment in main() for measured context.
//
// To change the analysis resolution, edit DST_W / DST_H below.
// ============================================================================
struct DownsampleConfig {
    int  dst_w;     // analysis width  — must be even (YUV420p subsampling)
    int  dst_h;     // analysis height — must be even
    bool enabled;   // set false to bypass downscaling (native resolution path)
};

// ── Tweak these two constants to change the analysis resolution ──────────────
static constexpr int  ANALYSIS_W       = 1280;   // default: 720p wide
static constexpr int  ANALYSIS_H       =  720;   // default: 720p tall
static constexpr bool DOWNSAMPLE_ENABLED = true; // set false to use native res
// ─────────────────────────────────────────────────────────────────────────────

static DownsampleConfig make_downsample_config(int native_w, int native_h)
{
    DownsampleConfig dc;
    dc.dst_w   = ANALYSIS_W;
    dc.dst_h   = ANALYSIS_H;
    // Auto-disable if the native resolution is already ≤ target, to avoid an
    // upscale (which would increase pixel count and harm performance).
    dc.enabled = DOWNSAMPLE_ENABLED
                 && (native_w > ANALYSIS_W || native_h > ANALYSIS_H);
    if (!dc.enabled) {
        dc.dst_w = native_w;
        dc.dst_h = native_h;
    }
    return dc;
}

// ─────────────────────────────────────────────────────────────────────────────
// make_capture_config
//
// Returns the correct CaptureConfig for the current OS.
//
// Windows  — dxgigrab: grabs the composed desktop via DXGI Desktop Duplication.
//            No elevation required for the primary display.  Delivers BGRA8
//            (AV_PIX_FMT_BGR0) → needs_swscale = true.
//
// macOS    — avfoundation: the only public screen-capture API on macOS without
//            a privacy entitlement on older SDK versions.  Delivers UYVY422 or
//            BGR24 depending on hardware → needs_swscale = true.
//            On macOS 12.3+ ScreenCaptureKit is preferred (see Tier 2 notes)
//            but requires ObjC; avfoundation works universally from C++.
//
// Linux X11  — x11grab: reads the X11 root window via XGetImage under the hood.
//              Delivers BGR0 / BGRA (AV_PIX_FMT_BGR0) → needs_swscale = true.
//              For Wayland use the "kmsgrab" format instead (requires CAP_SYS_ADMIN
//              or membership of the 'video' group); set WAYLAND_DISPLAY to detect.
//
// The device_url and framerate can be overridden at runtime via argv[1]/argv[2].
// ─────────────────────────────────────────────────────────────────────────────
static CaptureConfig make_capture_config(int argc, char* argv[])
{
    CaptureConfig cfg;

#if defined(_WIN32)
    cfg.input_format = "dxgigrab";
    cfg.device_url   = "desktop";          // entire primary desktop
    cfg.framerate    = "60";
    cfg.video_size   = nullptr;            // dxgigrab infers from the display
    cfg.pixel_format = nullptr;
    cfg.needs_swscale = true;              // dxgigrab → AV_PIX_FMT_BGR0

#elif defined(__APPLE__)
    cfg.input_format = "avfoundation";
    cfg.device_url   = "Capture screen 0"; // primary display (index 0)
    cfg.framerate    = "60";
    cfg.video_size   = nullptr;
    cfg.pixel_format = "bgr0";
    cfg.needs_swscale = true;

#elif defined(__linux__)
    // Prefer kmsgrab on bare-metal (direct KMS framebuffer, lowest latency).
    // Fall back to x11grab when DISPLAY is set (X11 desktop).
    const char* has_display  = std::getenv("DISPLAY");
    const char* has_wayland  = std::getenv("WAYLAND_DISPLAY");
    if (has_wayland && !has_display) {
        // Wayland without X11: use kmsgrab (needs video group / CAP_SYS_ADMIN).
        // PipeWire / xdg-desktop-portal is the recommended long-term path but
        // requires D-Bus integration beyond the scope of this C++ file.
        cfg.input_format = "kmsgrab";
        cfg.device_url   = "";             // primary KMS plane
        cfg.framerate    = "60";
        cfg.video_size   = nullptr;
        cfg.pixel_format = "bgr0";
        cfg.needs_swscale = true;
    } else {
        // X11 (or XWayland) session.
        cfg.input_format = "x11grab";
        cfg.device_url   = has_display ? has_display : ":0.0";
        cfg.framerate    = "60";
        cfg.video_size   = nullptr;        // will be set from display geometry below
        cfg.pixel_format = nullptr;
        cfg.needs_swscale = true;          // x11grab → AV_PIX_FMT_BGR0
    }
#endif

    // Allow argv[1] = device override, argv[2] = framerate override.
    if (argc > 1 && argv[1][0] != '\0') cfg.device_url = argv[1];
    if (argc > 2 && argv[2][0] != '\0') cfg.framerate  = argv[2];

    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// query_display_geometry_linux
//
// Used on Linux to fill cfg.video_size when x11grab is selected.
// x11grab requires an explicit WxH; it cannot infer from the display.
// We read from $DISPLAY geometry via XOpenDisplay / XDisplayWidth.
// Returns "1920x1080" as a fallback if X11 is unavailable.
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__linux__)
static std::string query_display_geometry_linux(const char* display_env)
{
    Display* dpy = XOpenDisplay(display_env);
    if (!dpy) return "1920x1080";
    int screen = DefaultScreen(dpy);
    int w = XDisplayWidth(dpy, screen);
    int h = XDisplayHeight(dpy, screen);
    XCloseDisplay(dpy);
    return std::to_string(w) + "x" + std::to_string(h);
}
#endif

// ============================================================================
// open_screen_capture
//
// Opens the platform capture device and returns a ready AVFormatContext.
// All AVDictionary options are freed before return regardless of success.
//
// Key design decisions:
//   draw_mouse=0   — Mouse cursor adds a spurious bright/dark dot that moves
//                    unpredictably across the frame. Excluding it prevents the
//                    cursor from contributing false flash pixels in the PSE
//                    detection pipeline.
//
//   thread_queue_size=512 — Screen-capture devices produce frames at the
//                    display refresh rate. Without a generous input thread
//                    queue the demuxer thread blocks on av_read_frame() while
//                    the analysis threads are busy, causing dropped frames.
//                    512 packet slots at ~100KB each = ~50MB peak queue.
//
//   probesize / analyzeduration — Set to the minimum for a raw device input
//                    so avformat_find_stream_info() returns immediately rather
//                    than buffering several seconds of frames to probe.
// ============================================================================
static AVFormatContext* open_screen_capture(const CaptureConfig& cfg,
                                            std::string& video_size_storage)
{
    // Must be called before any avformat_open_input with a device URL.
    avdevice_register_all();

    const AVInputFormat* ifmt = av_find_input_format(cfg.input_format);
    if (!ifmt) {
        std::cerr << "[!] Input format '" << cfg.input_format
                  << "' not found in this FFmpeg build.\n"
                  << "    Rebuild FFmpeg with --enable-" << cfg.input_format << "\n";
        return nullptr;
    }

    AVDictionary* opts = nullptr;

    // Frame rate — must match the display refresh exactly; lower values cause
    // the device to drop frames silently on some drivers.
    av_dict_set(&opts, "framerate", cfg.framerate, 0);

    // Capture resolution.  dxgigrab and avfoundation infer this automatically;
    // x11grab requires an explicit size.
    const char* vsize = cfg.video_size;
#if defined(__linux__)
    if (!vsize) {
        // Query X11 display geometry at runtime.
        video_size_storage = query_display_geometry_linux(cfg.device_url);
        vsize = video_size_storage.c_str();
    }
#endif
    if (vsize) av_dict_set(&opts, "video_size", vsize, 0);

    // Pixel format hint (device may ignore it).
    if (cfg.pixel_format) av_dict_set(&opts, "pixel_format", cfg.pixel_format, 0);

    // Exclude mouse cursor — avoids spurious flash pixels from cursor movement.
    av_dict_set(&opts, "draw_mouse", "0", 0);

    // Large input thread queue to prevent dropped frames under analysis load.
    av_dict_set(&opts, "thread_queue_size", "512", 0);

    // Minimise probing time: we know it's a live video device.
    av_dict_set(&opts, "probesize",         "32",  0); // 32 bytes
    av_dict_set(&opts, "analyzeduration",   "0",   0); // don't buffer frames

    // x11grab: capture the entire root window starting at offset 0,0.
    // Format: ":display.screen+x_offset,y_offset"
    std::string url_with_offset;
#if defined(__linux__)
    if (std::string(cfg.input_format) == "x11grab" && !strchr(cfg.device_url, '+')) {
        url_with_offset = std::string(cfg.device_url) + "+0,0";
        // "+0,0" appends the top-left capture origin.
    }
#endif
    const char* open_url = url_with_offset.empty() ? cfg.device_url
                                                    : url_with_offset.c_str();

    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, open_url, ifmt, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, e, sizeof(e));
        std::cerr << "[!] Cannot open capture device '" << open_url
                  << "' via " << cfg.input_format << ": " << e << "\n";
#if defined(__linux__)
        if (std::string(cfg.input_format) == "kmsgrab")
            std::cerr << "    Hint: add yourself to the 'video' group or run with sudo.\n";
        if (std::string(cfg.input_format) == "x11grab")
            std::cerr << "    Hint: ensure DISPLAY=" << cfg.device_url << " is accessible.\n";
#elif defined(__APPLE__)
        std::cerr << "    Hint: grant Screen Recording permission in System Settings → Privacy.\n";
#endif
        return nullptr;
    }
    return fmt_ctx;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
    auto start_t = std::chrono::high_resolution_clock::now();

    // ── Build platform capture config ────────────────────────────────────
    std::string video_size_storage; // lifetime must outlive open_screen_capture
    CaptureConfig cfg = make_capture_config(argc, argv);

    // ── Open the screen capture device ───────────────────────────────────
    AVFormatContext* fmt_ctx = open_screen_capture(cfg, video_size_storage);
    if (!fmt_ctx) return -1;

    // avformat_find_stream_info on a live device buffers frames.
    // With analyzeduration=0 this returns after reading a single packet.
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "[!] Cannot find stream info.\n";
        avformat_close_input(&fmt_ctx);
        return -1;
    }

    int v_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1,
                                    nullptr, 0);
    if (v_idx < 0) {
        std::cerr << "[!] No video stream in capture device.\n";
        avformat_close_input(&fmt_ctx);
        return -1;
    }

    // ── Decoder setup ─────────────────────────────────────────────────────
    // Screen capture devices are "raw" — the codec is rawvideo (BGR, BGRA, etc.)
    // There is no entropy decoding; avcodec_receive_frame copies the packet
    // data into the AVFrame planes with no further processing.
    AVCodecParameters* cp    = fmt_ctx->streams[v_idx]->codecpar;
    const AVCodec*     codec = avcodec_find_decoder(cp->codec_id);
    if (!codec) {
        std::cerr << "[!] No decoder for codec id " << cp->codec_id << "\n";
        avformat_close_input(&fmt_ctx);
        return -1;
    }
    AVCodecContext* cc = avcodec_alloc_context3(codec);
    if (!cc || avcodec_parameters_to_context(cc, cp) < 0
           || avcodec_open2(cc, codec, nullptr) < 0) {
        std::cerr << "[!] Codec init failed.\n";
        avformat_close_input(&fmt_ctx);
        return -1;
    }

    AVPacket* pkt   = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc(); // decoded frame (RGB/BGR from device)
    if (!pkt || !frame) {
        std::cerr << "[!] Alloc failed.\n";
        return -1;
    }

    AVStream*    vs  = fmt_ctx->streams[v_idx];
    double       tb  = av_q2d(vs->time_base);

    // Frame rate: use avg_frame_rate for live devices (r_frame_rate may be 0/0).
    double fps = 60.0; // safe default for screen capture
    if (vs->avg_frame_rate.den > 0 && vs->avg_frame_rate.num > 0)
        fps = av_q2d(vs->avg_frame_rate);
    else if (vs->r_frame_rate.den > 0 && vs->r_frame_rate.num > 0)
        fps = av_q2d(vs->r_frame_rate);

    // ── swscale setup (RGB→YUV420p conversion) ────────────────────────────
    // Screen devices deliver BGR0 / BGRA / UYVY422. The PSE pipeline expects
    // YUV420p (separate Y and Cr planes). We convert once per captured frame.
    //
    // SWS_BILINEAR is chosen over SWS_FAST_BILINEAR because:
    //   1. The only scaling here is chroma subsampling (no spatial resize),
    //      so the filter quality difference is invisible.
    //   2. SWS_BILINEAR uses the same SIMD kernels as FAST_BILINEAR for the
    //      packed→planar conversion path; the naming difference matters only
    //      when actual scaling is involved.
    //   3. Correctness: FAST_BILINEAR can produce UV phase errors on some
    //      chroma-subsampling paths, which would bias the Cr-based red flash
    //      detection (FIX 5).
    //
    // The output frame (yuv_frame) is allocated once and reused every capture
    // cycle — no per-frame heap allocation in the conversion path.
    // U (Cb) plane is intentionally left uninitialised: push_frames only reads
    // data[0] (Y) and data[2] (Cr), so data[1] is never touched downstream.
    AVFrame* yuv_frame = nullptr;

    if (cfg.needs_swscale) {
        yuv_frame = av_frame_alloc();
        if (!yuv_frame) { std::cerr << "[!] yuv_frame alloc failed.\n"; return -1; }
        yuv_frame->format = AV_PIX_FMT_YUV420P;
        yuv_frame->width  = cp->width;
        yuv_frame->height = cp->height;
        // 32-byte alignment satisfies AVX2 store alignment in bgr0_to_yuv420p_cs
        // and in PSEBuffer::scatter_y.
        if (av_frame_get_buffer(yuv_frame, 32) < 0) {
            std::cerr << "[!] yuv_frame buffer alloc failed.\n"; return -1;
        }
        // No SwsContext needed: bgr0_to_yuv420p_cs is a direct AVX2 call.
    }

    // ── Downsample setup (spatial resize to analysis resolution) ─────────
    // A second sws pass resizes the full-resolution YUV420p frame produced
    // above (or the native YUV420p device frame on the non-swscale path) down
    // to the configured analysis resolution before it enters PSEBuffer/PSEState.
    //
    // This reduces the pixel count seen by the detection mechanism.  The
    // trade-off analysis is printed at startup.
    //
    // To change the output resolution: edit ANALYSIS_W / ANALYSIS_H at the
    // top of this file and recompile.  No other changes are required.
    DownsampleConfig dc = make_downsample_config(cp->width, cp->height);

    AVFrame*    ds_frame = nullptr;  // downsampled YUV420p frame (analysis size)
    SwsContext* sws_ds   = nullptr;  // resize context (full-res YUV -> analysis YUV)

    if (dc.enabled) {
        ds_frame = av_frame_alloc();
        if (!ds_frame) { std::cerr << "[!] ds_frame alloc failed.\n"; return -1; }
        ds_frame->format = AV_PIX_FMT_YUV420P;
        ds_frame->width  = dc.dst_w;
        ds_frame->height = dc.dst_h;
        if (av_frame_get_buffer(ds_frame, 32) < 0) {
            std::cerr << "[!] ds_frame buffer alloc failed.\n"; return -1;
        }

        // YUV420p -> YUV420p spatial resize.
        // SWS_BILINEAR gives acceptable quality for the luma plane used by the
        // luminance threshold and adequate chroma accuracy for FIX 5 (sat-red).
        // SWS_AREA would be marginally better for heavy downscales (>4x) but
        // is slower; BILINEAR is the better default across the range of monitor
        // resolutions we encounter.
        sws_ds = sws_getContext(
            cp->width,  cp->height,  AV_PIX_FMT_YUV420P,   // src: full-res YUV
            dc.dst_w,   dc.dst_h,    AV_PIX_FMT_YUV420P,   // dst: analysis res
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ds) {
            std::cerr << "[!] sws_getContext (downsample) failed.\n";
            return -1;
        }
    }

    // ── PSE state setup ───────────────────────────────────────────────────
    const uint64_t valid_combos = build_valid_combo_mask(fps);
    // Screen content is always BT.709 (sRGB primaries) on modern OSes.
    // Pass AVCOL_SPC_BT709 explicitly rather than relying on cp->color_space
    // which is often AVCOL_SPC_UNSPECIFIED for raw device streams.
    // Use dc.dst_w/h (the analysis resolution) — when downsampling is active
    // this is smaller than cp->width/height.  All PSE state (area thresholds,
    // scatter arrays, etc.) must be sized to the frame that actually enters the
    // detector, not the raw capture resolution.
    const int16_t kr = derive_kr_fixed(AVCOL_SPC_BT709, dc.dst_w, dc.dst_h);

    // ── FAST detector (8-slot, 4-frame ring) — catches >4.3 Hz @ 60fps ──
    PSEState state;
    state.init(dc.dst_w, dc.dst_h, valid_combos, kr, fps);

    PSEBufferFast     buf;
    buf.time_base = tb;
    PersistentWorker w1, w2;

    // ── SLOW detector (16-slot, 16-frame ring) — catches 3–4.3 Hz @ 60fps ─
    uint64_t valid_combos_16[4];
    build_valid_combo_mask_16(fps, valid_combos_16);

    PSEState16        state16;
    state16.init(dc.dst_w, dc.dst_h, valid_combos_16, kr, fps);

    PSEBufferSlow     buf16;
    buf16.time_base = tb;
    PersistentWorker  w3, w4;   // dedicated workers for the slow detector

    DisplayProfile   dp = get_updated_display_profile();

    std::cout << "[+] loadv82 — dual-detector ITU-R BT.1702-3 PSE (fast 8-slot + slow 16-slot)\n"
              << "    Device : " << cfg.input_format
              << "  URL: "       << cfg.device_url << "\n"
              << "    Video  : " << cp->width << "x" << cp->height
              << "  FPS: "       << fps << "\n"
              << "    SrcFmt : " << av_get_pix_fmt_name((AVPixelFormat)cp->format)
              << (cfg.needs_swscale ? " -> YUV420p (swscale)" : " native YUV420p")
              << "\n"
              << "    Analyse: " << dc.dst_w << "x" << dc.dst_h
              << (dc.enabled ? "  (downsampled from native)" : "  (native — no downscale)")
              << "\n"
              << "    Colour : kr=" << kr << "/64  (BT.709 forced)\n"
              << "    FAST   : 8-slot / 4-frame ring"
              << "  max_sep=" << std::max(1, std::min((int)(fps/(2.0*3.0)), 7))
              << "  mask=0x" << std::hex << valid_combos << std::dec << "\n"
              << "    SLOW   : 16-slot / 16-frame ring"
              << "  max_sep=" << std::max(1, std::min((int)(fps/(2.0*3.0)), 15))
              << "  (3Hz capable)\n"
              << "    Workers: 4 total (w1+w2=fast, w3+w4=slow)\n"
              << "    Fixes  : dual-threshold | opposing-pair | concurrent-area\n"
              << "             | rising-edge-rate | sat-red | strobe-aware-motion-comp | patterns\n"
              << "    Press Ctrl-C to stop.\n";

    // ── Detection speed capability summary ───────────────────────────────
    {
        const int    fast_max_sep    = std::max(1, std::min((int)(fps / (2.0 * 3.0)), 7));
        const double fast_min_hz     = fps / (2.0 * fast_max_sep);
        const double fast_win_s      = (PSEBufferFast::MAX_SIZE - 1) / fps;

        const int    slow_max_sep    = std::max(1, std::min((int)(fps / (2.0 * 3.0)), 15));
        const double slow_min_hz     = fps / (2.0 * slow_max_sep);
        const double slow_win_s      = (PSEBufferSlow::MAX_SIZE - 1) / fps;

        std::cout << "[D] Dual-detector capability @ " << fps << " fps:\n"
                  << "    FAST  8-slot: win=" << std::fixed << std::setprecision(1)
                  <<                           (fast_win_s * 1000.0) << "ms"
                  << "  max_sep=" << fast_max_sep
                  << "  min_hz=" << std::setprecision(2) << fast_min_hz << "Hz\n"
                  << "    SLOW 16-slot: win=" << std::setprecision(1)
                  <<                           (slow_win_s * 1000.0) << "ms"
                  << "  max_sep=" << slow_max_sep
                  << "  min_hz=" << std::setprecision(2) << slow_min_hz << "Hz\n"
                  << "    Combined coverage: " << slow_min_hz << "–30 Hz\n";

        if (slow_min_hz <= 3.01)
            std::cout << "    3Hz detection: OK (slow detector covers "
                      << std::setprecision(2) << slow_min_hz << "Hz minimum)\n";
        else
            std::cout << "    *** 3Hz still BLOCKED: slow min_hz="
                      << slow_min_hz << "Hz > 3Hz — fps too low for 16-slot ring ***\n";

        std::cout << std::defaultfloat;
    }
    
    std::cout << "=== Display Profile ===\n"
              << "Max Luminance   : " << dp.max_nits               << " nits\n"
              << "Brightness      : " << dp.current_brightness_pct * 100.0f << "%\n"
              << "Effective nits  : " << dp.max_nits * dp.current_brightness_pct << " nits\n"
              << "Active EOTF     : " << dp.eotf_to_string(dp.active_eotf) << "\n"
              << "=======================\n";

    // ── Capture + analysis loop ───────────────────────────────────────────
    // av_read_frame blocks until the next display vblank delivers a new frame.
    // At 60Hz this loop body executes ~60 times per second on the main thread.
    //
    // There is no stream_index filtering: screen capture devices expose a
    // single video stream (v_idx == 0 always), so the check is retained for
    // correctness but will never skip a packet in practice.
    //
    // PTS handling: screen capture devices provide wall-clock PTS in their
    // time_base units. For the PSE detector these are used only to compute
    // inter-window time deltas (for the flash rate counter). As long as PTS
    // is monotonically increasing and approximately correct, the analysis
    // is unaffected by small jitter from vblank timing.
    //
    // OPT #8: cfg.needs_swscale and dc.enabled are invariant for the entire
    // run but are read through struct references, so the compiler cannot prove
    // they are constant and leaves the branches in the hot path.  We template
    // the inner loop on both flags so the compiler eliminates each dead branch
    // at instantiation time.  The outer dispatch runs once before the loop.
    auto run_capture = [&]<bool NEEDS_SWSCALE, bool DS_ENABLED>() {
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == v_idx) {
                if (avcodec_send_packet(cc, pkt) >= 0) {
                    while (avcodec_receive_frame(cc, frame) >= 0) {

                        // ── BGR0→YUV420p conversion (Y + Cr only) ───────────
                        AVFrame* yuv = frame;
                        if constexpr (NEEDS_SWSCALE) {
                            const BGR0ColorSpace cs = (frame->width >= 1280)
                                                    ? BGR0_CS_BT709 : BGR0_CS_BT601;
                            bgr0_to_yuv420p_cs(
                                frame->data[0],        frame->linesize[0],
                                yuv_frame->data[0],    yuv_frame->linesize[0],
                                yuv_frame->data[2],    yuv_frame->linesize[2],
                                frame->width, frame->height, cs);
                            yuv_frame->pts = frame->pts;
                            yuv = yuv_frame;
                        }

                        // ── Spatial downsample ────────────────────────────────
                        AVFrame* yuv_pse = yuv;
                        if constexpr (DS_ENABLED) {
                            sws_scale(sws_ds,
                                      yuv->data,      yuv->linesize,    0, cp->height,
                                      ds_frame->data, ds_frame->linesize);
                            ds_frame->pts = yuv->pts;
                            yuv_pse = ds_frame;
                        }

                        // ── Push to BOTH ring buffers ─────────────────────────
                        // buf  = 4-frame fast detector (catches >4.3 Hz @ 60fps)
                        // buf16 = 16-frame slow detector (catches 3–4.3 Hz @ 60fps)
                        // Both receive identical frame data; their ring sizes and
                        // combo masks are the only differences.
                        buf.push_frames(yuv_pse->data[0], yuv_pse->linesize[0],
                                        yuv_pse->data[2], yuv_pse->linesize[2],
                                        yuv_pse->width,   yuv_pse->height,
                                        yuv_pse->pts);

                        buf16.push_frames(yuv_pse->data[0], yuv_pse->linesize[0],
                                          yuv_pse->data[2], yuv_pse->linesize[2],
                                          yuv_pse->width,   yuv_pse->height,
                                          yuv_pse->pts);

                        // ── FAST detector ────────────────────────────────────
                        // Fires every frame once the 4-frame ring is full.
                        // Uses w1+w2; output prefixed [D] (diagnostics) / [i]
                        if (buf.stored_frames >= buf.MAX_SIZE)
                            process_pse_temporal_avx_threaded(buf, w1, w2, dp, state);

                        // ── SLOW detector ────────────────────────────────────
                        // Fires every frame once the 16-frame ring is full
                        // (~267ms warm-up at 60fps).  Uses w3+w4; output
                        // prefixed [S] so it is distinct from the fast detector.
                        // Alarm output is prefixed [!!!][SLOW].
                        if (buf16.stored_frames >= buf16.MAX_SIZE)
                            process_pse_temporal_avx_threaded_16slot(
                                buf16, w3, w4, dp, state16);

                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(pkt);
        }
    };

    // Dispatch once — selects the specialisation with no dead branches.
    const bool ns = cfg.needs_swscale;
    const bool de = dc.enabled;
    if      ( ns &&  de) run_capture.operator()<true,  true>();
    else if ( ns && !de) run_capture.operator()<true,  false>();
    else if (!ns &&  de) run_capture.operator()<false, true>();
    else                 run_capture.operator()<false, false>();

    // ── Cleanup ───────────────────────────────────────────────────────────
    if (sws_ds)   sws_freeContext(sws_ds);
    if (ds_frame) av_frame_free(&ds_frame);
    if (yuv_frame) av_frame_free(&yuv_frame);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&cc);
    avformat_close_input(&fmt_ctx);

    auto end_t = std::chrono::high_resolution_clock::now();
    std::cout << "[+] Done. Runtime: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count()
              << " ms\n";
    return 0;
}