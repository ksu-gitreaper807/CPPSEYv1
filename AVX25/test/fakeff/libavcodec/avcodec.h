#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavcodec.
#pragma once
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/colorspace.h>
#include <libavutil/frame.h>
#include <stdint.h>

typedef enum AVMediaType {
    AVMEDIA_TYPE_UNKNOWN = -1,
    AVMEDIA_TYPE_VIDEO   = 0,
    AVMEDIA_TYPE_AUDIO   = 1,
} AVMediaType;

typedef struct AVCodecParameters {
    int  codec_id;
    int  width;
    int  height;
    int  format;      // AVPixelFormat
} AVCodecParameters;

typedef struct AVCodec {
    const char *name;
} AVCodec;

struct AVCodecContext {
    // fake-FFmpeg internal state (see fakeff_impl.cpp)
    int  pending_frame;
    bool frame_delivered;
};

typedef struct AVPacket {
    int     stream_index;
    void   *data;      // fake: points at a slot holding the frame index
    int64_t size;
} AVPacket;

const AVCodec *avcodec_find_decoder(int codec_id);
AVCodecContext *avcodec_alloc_context3(const AVCodec *codec);
int avcodec_parameters_to_context(AVCodecContext *ctx, const AVCodecParameters *par);
int avcodec_open2(AVCodecContext *ctx, const AVCodec *codec, void *options);
void avcodec_free_context(AVCodecContext **ctx);
int avcodec_send_packet(AVCodecContext *ctx, const AVPacket *pkt);
int avcodec_receive_frame(AVCodecContext *ctx, AVFrame *frame);

AVPacket *av_packet_alloc(void);
void      av_packet_free(AVPacket **pkt);
void      av_packet_unref(AVPacket *pkt);

#endif
}