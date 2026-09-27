#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavutil pixel format definitions.
#pragma once
#include <libavutil/avutil.h>

typedef enum AVPixelFormat {
    AV_PIX_FMT_NONE = -1,
    AV_PIX_FMT_YUV420P = 0,
    AV_PIX_FMT_BGR0    = 11,
} AVPixelFormat;

const char *av_get_pix_fmt_name(AVPixelFormat fmt);

#endif
}