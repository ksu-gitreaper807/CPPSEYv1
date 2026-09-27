#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavutil core types.
// Used ONLY by the synthetic test harness (see test/README.md) so the
// unmodified detector source can be compiled and exercised without a real
// FFmpeg installation.  NOT part of the production build.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define AV_ERROR_MAX_STRING_SIZE 128

#define AVERROR(e) (-(e))
#define AVERROR_EOF      (-53)
#define AVERROR_EAGAIN   (-11)
#define AVERROR_INVALIDDATA (-1134136151)

typedef struct Rational {
    int num;
    int den;
} Rational;

static inline double av_q2d(Rational a)
{
    return (a.den != 0) ? (double)a.num / (double)a.den : 0.0;
}

int av_strerror(int errnum, char *errbuf, size_t errbuf_size);

#endif
}