#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libswscale.
// sws_scale() performs an area (box) average YUV420p resize — close enough
// to SWS_BILINEAR for the synthetic test content used by the harness.
#pragma once
#include <libavutil/pixfmt.h>

typedef struct SwsContext {
    int sw, sh, dw, dh;
} SwsContext;

#define SWS_FAST_BILINEAR 2
#define SWS_BILINEAR      2
#define SWS_BICUBIC       4
#define SWS_AREA          5

SwsContext *sws_getContext(int srcW, int srcH, AVPixelFormat srcFormat,
                           int dstW, int dstH, AVPixelFormat dstFormat,
                           int flags, void *filterIn, void *filterOut,
                           const double *param);
void        sws_freeContext(SwsContext *ctx);
int         sws_scale(SwsContext *ctx,
                      const uint8_t *const srcSlice[], const int srcStride[],
                      int srcSliceY, int srcSliceH,
                      uint8_t *const dst[], const int dstStride[]);

#endif
}