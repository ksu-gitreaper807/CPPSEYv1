#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavformat.
#pragma once
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>

typedef struct AVStream {
    AVCodecParameters *codecpar;
    Rational           time_base;
    Rational           avg_frame_rate;
    Rational           r_frame_rate;
} AVStream;

typedef struct AVFormatContext {
    int       nb_streams;
    AVStream **streams;
} AVFormatContext;

const char *avformat_name_for_test(void); // not in real FFmpeg; harmless

int avformat_open_input(AVFormatContext **ps, const char *url, const void *fmt, AVDictionary **options);
int avformat_find_stream_info(AVFormatContext *s, AVDictionary **options);
void avformat_close_input(AVFormatContext **ps);
int  av_find_best_stream(AVFormatContext *ic, AVMediaType type,
                         int wanted, int related, void *decoder_ret, int flags);
int  av_read_frame(AVFormatContext *s, AVPacket *pkt);

#endif
}