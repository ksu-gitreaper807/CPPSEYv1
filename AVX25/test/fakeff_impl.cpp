// fakeff_impl.cpp — minimal API-compatible FFmpeg/X11 implementation for the
// PSE detector synthetic test harness.
//
// The "capture device" is replaced by a procedural frame generator
// (pse_synth_frame in scenarios.cpp).  The rest of the detector — BGR0→YUV420p
// conversion, spatial downsample, dual PSE detectors, rate logic, alarm
// output — runs UNMODIFIED.
//
// This file is test infrastructure only; it is never part of the production
// build (which links against real FFmpeg).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/colorspace.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>

// ── Scenario generator interface (defined in scenarios.cpp) ────────────────
extern "C" void pse_synth_frame(int idx, uint8_t *bgr0, int w, int h);

// ============================================================================
// Shared fake "device" state
// ============================================================================
struct FakeFormatCtx {
    int   width = 1920, height = 1080;
    int   fps_num = 60, fps_den = 1;
    int64_t frame_index = 0;
    int64_t max_frames = -1;
    AVCodecParameters cpar;
    AVStream          stream;
};
static FakeFormatCtx *g_fmt = nullptr;
FakeFormatCtx *g_fmt_holder() { return g_fmt; }

// ============================================================================
// avutil
// ============================================================================
int av_strerror(int errnum, char *errbuf, size_t errbuf_size)
{
    if (errbuf && errbuf_size) {
        snprintf(errbuf, errbuf_size, "fakeff error %d", errnum);
    }
    return -1;
}

const char *av_get_pix_fmt_name(AVPixelFormat fmt)
{
    switch (fmt) {
        case AV_PIX_FMT_YUV420P: return "yuv420p";
        case AV_PIX_FMT_BGR0:    return "bgr0";
        default:                 return "unknown";
    }
}

// ============================================================================
// AVDictionary (tiny fixed store; open_input reads the keys it needs)
// ============================================================================
struct FakeDict {
    char key[8][64];
    char value[8][256];
    int  count = 0;
};
static FakeDict g_dict;

int av_dict_set(AVDictionary **pm, const char *key, const char *value, int /*flags*/)
{
    (void)pm;
    if (g_dict.count >= 8) return -1;
    int i = g_dict.count++;
    snprintf(g_dict.key[i],   sizeof(g_dict.key[i]),   "%s", key);
    snprintf(g_dict.value[i], sizeof(g_dict.value[i]), "%s", value ? value : "");
    return 0;
}

void av_dict_free(AVDictionary **m)
{
    (void)m;
    g_dict.count = 0;
}

static const char *dict_get(const char *key)
{
    for (int i = 0; i < g_dict.count; ++i)
        if (strcmp(g_dict.key[i], key) == 0) return g_dict.value[i];
    return nullptr;
}

// ============================================================================
// AVFrame
// ============================================================================
AVFrame *av_frame_alloc(void) { return new AVFrame(); }
void     av_frame_free(AVFrame **frame)
{
    if (frame && *frame) {
        for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) delete[] (*frame)->data[i];
        delete *frame;
        *frame = nullptr;
    }
}
void av_frame_unref(AVFrame *frame)
{
    if (!frame) return;
    for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
        delete[] frame->data[i];
        frame->data[i] = nullptr;
        frame->linesize[i] = 0;
    }
}

static int round_up(int x, int align) { return (x + align - 1) / align * align; }

int av_frame_get_buffer(AVFrame *frame, int align)
{
    align = std::max(align, 32);
    int w = frame->width, h = frame->height;
    int y_size = round_up(w * h, align);
    int uv_size = round_up((w / 2) * (h / 2), align);
    frame->linesize[0] = w;
    frame->linesize[1] = w / 2;
    frame->linesize[2] = w / 2;
    frame->data[0] = new uint8_t[y_size];
    frame->data[1] = new uint8_t[uv_size];
    frame->data[2] = new uint8_t[uv_size];
    if (frame->format == AV_PIX_FMT_BGR0) {
        int bgr_size = round_up(w * h * 4, align);
        delete[] frame->data[0];
        frame->data[0] = new uint8_t[bgr_size];
        frame->linesize[0] = w * 4;
        delete[] frame->data[1]; delete[] frame->data[2];
        frame->data[1] = nullptr; frame->data[2] = nullptr;
    }
    return 0;
}

// ============================================================================
// avcodec
// ============================================================================
static AVCodec g_fake_codec = { "fake-rawvideo" };

const AVCodec *avcodec_find_decoder(int codec_id) { (void)codec_id; return &g_fake_codec; }

AVCodecContext *avcodec_alloc_context3(const AVCodec *codec)
{
    (void)codec;
    auto *cc = new AVCodecContext();
    cc->pending_frame = -1;
    cc->frame_delivered = false;
    return cc;
}
int avcodec_parameters_to_context(AVCodecContext *ctx, const AVCodecParameters *par)
{
    (void)ctx; (void)par;
    return 0;
}
int avcodec_open2(AVCodecContext *ctx, const AVCodec *codec, void *options)
{
    (void)ctx; (void)codec; (void)options;
    return 0;
}
void avcodec_free_context(AVCodecContext **ctx)
{
    if (ctx && *ctx) { delete *ctx; *ctx = nullptr; }
}

int avcodec_send_packet(AVCodecContext *ctx, const AVPacket *pkt)
{
    if (!ctx || !pkt || !pkt->data) return AVERROR(EAGAIN);
    ctx->pending_frame = *static_cast<int*>(pkt->data);
    ctx->frame_delivered = false;
    return 0;
}

int avcodec_receive_frame(AVCodecContext *ctx, AVFrame *frame)
{
    if (!ctx || ctx->pending_frame < 0 || ctx->frame_delivered)
        return AVERROR(EAGAIN);
    ctx->frame_delivered = true;

    FakeFormatCtx *fmt = g_fmt_holder();
    int w = fmt->width, h = fmt->height;
    frame->width = w;
    frame->height = h;
    frame->format = AV_PIX_FMT_BGR0;
    frame->linesize[0] = w * 4;
    delete[] frame->data[0];
    frame->data[0] = new uint8_t[round_up((size_t)w * h * 4, 32)];
    pse_synth_frame(ctx->pending_frame, frame->data[0], w, h);
    frame->pts = ctx->pending_frame;
    ctx->pending_frame = -1;
    return 0;
}

AVPacket *av_packet_alloc(void) { return new AVPacket(); }
void av_packet_free(AVPacket **pkt)
{
    if (pkt && *pkt) { delete *pkt; *pkt = nullptr; }
}
void av_packet_unref(AVPacket *pkt)
{
    if (pkt) { pkt->data = nullptr; pkt->size = 0; }
}

// ============================================================================
// avformat — the fake "screen capture device"
// ============================================================================
const AVInputFormat *av_find_input_format(const char *name)
{
    static AVInputFormat fmt;
    fmt.name = name ? name : "";
    return &fmt;
}
int avdevice_register_all(void) { return 0; }

static int parse_fps(const char *s, int &num, int &den)
{
    if (!s || !*s) return -1;
    char *end = nullptr;
    double v = strtod(s, &end);
    if (end && *end == '/') {
        double d = strtod(end + 1, nullptr);
        if (d <= 0) return -1;
        num = (int)lround(v);
        den = (int)lround(d);
        if (num <= 0 || den <= 0) return -1;
    } else {
        if (v <= 0) return -1;
        num = (int)lround(v * 1000);
        den = 1000;
    }
    return 0;
}

int avformat_open_input(AVFormatContext **ps, const char *url, const void *fmt, AVDictionary **options)
{
    (void)url; (void)fmt; (void)options;

    FakeFormatCtx *f = new FakeFormatCtx();

    // video_size — PSE_SYNTH_SIZE (harness override) wins over the X11-queried
    // value (which falls back to 1920x1080 headless).
    const char *size = getenv("PSE_SYNTH_SIZE");
    if (!size) size = dict_get("video_size");
    if (size) {
        int w = 0, h = 0;
        if (sscanf(size, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
            f->width = w & ~1;
            f->height = h & ~1;
        }
    }

    // framerate override (argv[2] in the real detector).
    const char *fr = dict_get("framerate");
    if (fr) parse_fps(fr, f->fps_num, f->fps_den);

    f->max_frames = atoll(getenv("PSE_SYNTH_FRAMES") ? getenv("PSE_SYNTH_FRAMES") : "600");

    f->cpar.codec_id = 0;
    f->cpar.width    = f->width;
    f->cpar.height   = f->height;
    f->cpar.format   = AV_PIX_FMT_BGR0;
    f->stream.codecpar      = &f->cpar;
    f->stream.time_base     = { f->fps_den, f->fps_num };
    std::fprintf(stderr, "[fakeff] fps_num=%d fps_den=%d tb={%d,%d} size=%dx%d\n",
                 f->fps_num, f->fps_den, f->fps_den, f->fps_num, f->width, f->height);
    f->stream.avg_frame_rate= { f->fps_num, f->fps_den };
    f->stream.r_frame_rate  = { f->fps_num, f->fps_den };

    delete g_fmt;
    g_fmt = f;
    AVFormatContext *ctx = new AVFormatContext();
    ctx->nb_streams = 1;
    ctx->streams = new AVStream*[1];
    ctx->streams[0] = &f->stream;
    *ps = ctx;
    return 0;
}

int avformat_find_stream_info(AVFormatContext *s, AVDictionary **options)
{
    (void)s; (void)options;
    return 0;
}

void avformat_close_input(AVFormatContext **ps)
{
    if (ps && *ps) {
        delete[] (*ps)->streams;
        delete *ps;
        *ps = nullptr;
    }
}

int av_find_best_stream(AVFormatContext *ic, AVMediaType type, int wanted,
                        int related, void *decoder_ret, int flags)
{
    (void)ic; (void)type; (void)wanted; (void)related; (void)decoder_ret; (void)flags;
    return 0;
}

int av_read_frame(AVFormatContext *s, AVPacket *pkt)
{
    if (!g_fmt) return AVERROR_EOF;
    if (g_fmt->max_frames >= 0 && g_fmt->frame_index >= g_fmt->max_frames)
        return AVERROR_EOF;
    pkt->stream_index = 0;
    // Reuse a static slot: the real detector consumes the packet before the
    // next av_read_frame call, so one slot is safe.
    static int frame_slot;
    frame_slot = (int)g_fmt->frame_index++;
    pkt->data = &frame_slot;
    pkt->size = 4;
    return 0;
}

// ============================================================================
// libswscale — area (box) average YUV420p resize
// ============================================================================
SwsContext *sws_getContext(int srcW, int srcH, AVPixelFormat srcFormat,
                           int dstW, int dstH, AVPixelFormat dstFormat,
                           int flags, void *fi, void *fo, const double *param)
{
    (void)srcFormat; (void)dstFormat; (void)flags; (void)fi; (void)fo; (void)param;
    auto *c = new SwsContext();
    c->sw = srcW; c->sh = srcH; c->dw = dstW; c->dh = dstH;
    return c;
}

void sws_freeContext(SwsContext *ctx) { delete ctx; }

static void box_plane(const uint8_t *src, int sstride, int sw, int sh,
                      uint8_t *dst, int dstride, int dw, int dh)
{
    for (int y = 0; y < dh; ++y) {
        int y0 = (y * sh) / dh;
        int y1 = ((y + 1) * sh) / dh;
        if (y1 <= y0) y1 = y0 + 1;
        const uint8_t *r0 = src + (size_t)y0 * sstride;
        const uint8_t *r1 = src + (size_t)(y1 - 1) * sstride;
        for (int x = 0; x < dw; ++x) {
            int x0 = (x * sw) / dw;
            int x1 = ((x + 1) * sw) / dw;
            if (x1 <= x0) x1 = x0 + 1;
            // For the 1.5x analysis downscale the region is at most 2x2, so
            // the average is a corner sum (duplicates counted once).
            int n = (y1 - y0) * (x1 - x0);
            int acc = r0[x0] + r1[x0] + r0[x1 - 1] + r1[x1 - 1];
            if (y1 - y0 == 1) acc -= r0[x0] + r1[x1 - 1];
            if (x1 - x0 == 1) acc -= r0[x0] + r1[x0];
            if ((y1 - y0) == 1 && (x1 - x0) == 1) acc += r0[x0];
            (void)n;
            dst[y * dstride + x] = (uint8_t)(acc / ((y1 - y0) * (x1 - x0)));
        }
    }
}

int sws_scale(SwsContext *c,
              const uint8_t *const srcSlice[], const int srcStride[],
              int srcSliceY, int srcSliceH,
              uint8_t *const dst[], const int dstStride[])
{
    (void)srcSliceY; (void)srcSliceH;
    int sw2 = c->sw / 2, sh2 = c->sh / 2, dw2 = c->dw / 2, dh2 = c->dh / 2;
    box_plane(srcSlice[0], srcStride[0], c->sw, c->sh, dst[0], dstStride[0], c->dw, c->dh);
    box_plane(srcSlice[1], srcStride[1], sw2, sh2, dst[1], dstStride[1], dw2, dh2);
    box_plane(srcSlice[2], srcStride[2], sw2, sh2, dst[2], dstStride[2], dw2, dh2);
    return c->dh;
}

// ============================================================================
// X11 — no display available; detector falls back to its safe defaults
// ============================================================================
Display *XOpenDisplay(const char *display) { (void)display; return nullptr; }
void XCloseDisplay(Display *dpy) { (void)dpy; }
int  DefaultScreen(Display *dpy) { (void)dpy; return 0; }
Window RootWindow(Display *dpy, int s) { (void)dpy; (void)s; return 0; }
Window DefaultRootWindow(Display *dpy) { (void)dpy; return 0; }
int XDisplayWidth(Display *dpy, int s) { (void)dpy; (void)s; return 1920; }
int XDisplayHeight(Display *dpy, int s) { (void)dpy; (void)s; return 1080; }
void XFree(void *p) { (void)p; }
Atom XInternAtom(Display *d, const char *n, int o) { (void)d; (void)n; (void)o; return 1; }
void XFlush(Display *d) { (void)d; }

int XRRQueryVersion(Display *d, int *maj, int *min) { (void)d; (void)maj; (void)min; return 0; }
XRRScreenResources *XRRGetScreenResources(Display *d, Window w) { (void)d; (void)w; return nullptr; }
void XRRFreeScreenResources(XRRScreenResources *r) { (void)r; }
XRROutputInfo *XRRGetOutputInfo(Display *d, XRRScreenResources *r, RROutput o)
{ (void)d; (void)r; (void)o; return nullptr; }
void XRRFreeOutputInfo(XRROutputInfo *o) { (void)o; }
int XRRGetOutputProperty(Display *d, RROutput o, Atom p, long off, long len,
                         int del, int nz, Atom t, Atom *at, int *f,
                         unsigned long *ni, unsigned long *ba, unsigned char **prop)
{
    (void)d; (void)o; (void)p; (void)off; (void)len; (void)del; (void)nz;
    (void)t; (void)at; (void)f; (void)ni; (void)ba; (void)prop;
    return 0;
}
XRRCrtcInfo *XRRGetCrtcInfo(Display *d, XRRScreenResources *r, RRCrtc c)
{ (void)d; (void)r; (void)c; return nullptr; }
void XRRFreeCrtcInfo(XRRCrtcInfo *c) { (void)c; }
XRRCrtcGamma *XRRGetCrtcGamma(Display *d, RRCrtc c) { (void)d; (void)c; return nullptr; }
void XRRSetCrtcGamma(Display *d, RRCrtc c, XRRCrtcGamma *g) { (void)d; (void)c; (void)g; }
XRRCrtcGamma *XRRAllocGamma(int size) { (void)size; return nullptr; }
void XRRFreeGamma(XRRCrtcGamma *g) { (void)g; }
