#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for X11/extensions/Xrandr.h (harness only).
#pragma once
#include <X11/Xlib.h>

typedef unsigned int RROutput;
typedef unsigned int RRCrtc;

typedef struct _XRROutputInfo {
    int      connection;
    int      noutput;
} XRROutputInfo;

typedef struct _XRRScreenResources {
    int          noutput;
    RROutput    *outputs;
    int          ncrtc;
    RRCrtc      *crtcs;
} XRRScreenResources;

typedef struct _XRRCrtcInfo {
    int noutput;
} XRRCrtcInfo;

typedef struct _XRRCrtcGamma {
    int      size;
    unsigned short *red;
    unsigned short *green;
    unsigned short *blue;
} XRRCrtcGamma;

#define RR_Connected 1

int             XRRQueryVersion(Display *dpy, int *major, int *minor);
XRRScreenResources *XRRGetScreenResources(Display *dpy, Window root);
void            XRRFreeScreenResources(XRRScreenResources *res);
XRROutputInfo  *XRRGetOutputInfo(Display *dpy, XRRScreenResources *res, RROutput output);
void            XRRFreeOutputInfo(XRROutputInfo *oi);
// NOTE: the detector source calls this with 13 arguments (no res parameter);
// the stub matches that call.  It is never executed in the harness (no X11).
int             XRRGetOutputProperty(Display *dpy, RROutput output, Atom property,
                                     long offset, long len, int do_delete, int nonzero,
                                     Atom type, Atom *actual_type_return,
                                     int *format_return, unsigned long *nitems_return,
                                     unsigned long *bytes_after_return, unsigned char **prop_return);
XRRCrtcInfo    *XRRGetCrtcInfo(Display *dpy, XRRScreenResources *res, RRCrtc crtc);
void            XRRFreeCrtcInfo(XRRCrtcInfo *ci);
XRRCrtcGamma   *XRRGetCrtcGamma(Display *dpy, RRCrtc crtc);
void            XRRSetCrtcGamma(Display *dpy, RRCrtc crtc, XRRCrtcGamma *gamma);
XRRCrtcGamma   *XRRAllocGamma(int size);
void            XRRFreeGamma(XRRCrtcGamma *gamma);

#endif
}