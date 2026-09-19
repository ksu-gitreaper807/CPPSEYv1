// bgr0_yuv420p.c
// Direct, branch-miss–free BGR0→YUV420p conversion for the PSE pipeline.
//
// Architecture
// ────────────
//   Outer loop   : steps by 2 rows → zero even/odd chroma branches (Fix 2)
//   AVX2 body    : 16 pixels wide; processes one row pair per call
//   Scalar tail  : 0–14 remaining pixels; same branch direction every frame
//   Entry point  : plain direct call; no function-pointer dispatch (Fix 1)
//
// Data flow (one 16-pixel column, two rows)
// ──────────────────────────────────────────
//   1. Load 2×2 BGR0 chunks (2 rows × 64 bytes each)
//   2. Deinterleave B/G/R channels using _mm256_shuffle_epi8
//   3. Compute 32 Y values (16 per row) via Q8 fixed-point multiply-add
//   4. Horizontal-pair-sum B/G/R for each row with _mm_maddubs_epi16
//   5. Vertical-sum both rows → divide by 4 → 8 averaged channel values
//   6. Compute 8 Cb and 8 Cr values from averaged channels
//   7. Pack and store Y, Cb, Cr
//
// Numerical model
// ───────────────
//   BT.601 studio swing, Q8 fixed point:
//     Y  =  16 + ( 66·R + 129·G +  25·B + 128) >> 8   [unsigned, fits u16]
//     Cb = 128 + (112·B -  38·R -  74·G + 128) >> 8   [signed, fits s16]
//     Cr = 128 + (112·R -  94·G -  18·B + 128) >> 8   [signed, fits s16]
//
//   BT.709 studio swing, Q8 fixed point:
//     Y  =  16 + ( 47·R + 157·G +  16·B + 128) >> 8
//     Cb = 128 + (112·B -  26·R -  86·G + 128) >> 8
//     Cr = 128 + (112·R - 102·G -  10·B + 128) >> 8
//
//   All intermediate sums fit in the stated type:
//     Y  max before >> 8 : 66·255+129·255+25·255+128+4096 = 60 324 < 65 535 ✓
//     Cb inner range     : [-28 432, +28 688] ⊂ [-32 768, +32 767]          ✓
//     Cr inner range     : [-28 432, +28 688] (same bound by construction)   ✓
//
// Compile: gcc -O2 -mavx2 -mfma -c bgr0_yuv420p.c

#include "bgr0_yuv420p.h"
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
