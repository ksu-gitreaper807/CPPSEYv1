#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for X11/Xlib.h (test harness only).
// XOpenDisplay() returns NULL so the detector takes its no-display fallback
// paths (SDR default profile, protection disabled) exactly as it would on a
// headless machine.
#pragma once

typedef struct _XDisplay Display;
typedef unsigned long Window;
typedef unsigned long Atom;

#define Success 1
#define Failure 0
#define True    1
#define False   0
#define None    0L
#define AnyPropertyType 0L

Display *XOpenDisplay(const char *display);
void     XCloseDisplay(Display *dpy);
int      DefaultScreen(Display *dpy);
Window   RootWindow(Display *dpy, int screen);
Window   DefaultRootWindow(Display *dpy);
int      XDisplayWidth(Display *dpy, int screen);
int      XDisplayHeight(Display *dpy, int screen);
void     XFree(void *ptr);
Atom     XInternAtom(Display *dpy, const char *name, int only_if_exists);
void     XFlush(Display *dpy);

#endif
}