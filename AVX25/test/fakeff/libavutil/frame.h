#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavutil AVFrame.
// Only the fields touched by the detector are modelled.
#pragma once
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>

#define AV_NUM_DATA_POINTERS 8

typedef struct AVFrame {
    uint8_t *data[AV_NUM_DATA_POINTERS];
    int     linesize[AV_NUM_DATA_POINTERS];
    int     width;
    int     height;
    int64_t pts;
    int     format;
} AVFrame;

AVFrame *av_frame_alloc(void);
void     av_frame_free(AVFrame **frame);
void     av_frame_unref(AVFrame *frame);
int      av_frame_get_buffer(AVFrame *frame, int align);

#endif
}