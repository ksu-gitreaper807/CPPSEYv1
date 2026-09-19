// xflash.c — full-screen X11 flash test for the PSE detector (x11grab path)
//
// Why this exists: under Wayland (e.g. KDE Plasma Wayland) x11grab can only
// see the X11 root window = the XWayland framebuffer.  Native Wayland apps
// are invisible to it.  This tool creates a FULL-SCREEN X11 window and
// flashes it black/white at a given rate, so you can verify the detector
// end-to-end on a Wayland machine:
//
//   gcc -O2 -o xflash xflash.c -lX11
//   ./xflash [hz=6] [seconds=10]
//
// Run it in one terminal and  ./detector :0.0 60  in another.
//
// Expected on a healthy capture path:
//   6 Hz  -> flood of [i] lines with Y_conc~100%, [D] lines with
//            diry=RISE/FALL flips, then [!!!] PSE EPILEPTIC FLASH ALARM
//            within ~0.5 s
//   3 Hz  -> [!!!][SLOW] PSE 3Hz FLASH ALARM within ~1 s
//
// If NO [i] lines appear at all, x11grab is not seeing the X11 root
// content (check with:  xwd -root -display :0.0 -out /tmp/x.xwd ).

#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    double hz       = (argc > 1) ? atof(argv[1]) : 6.0;
    double duration = (argc > 2) ? atof(argv[2]) : 10.0;
    if (hz <= 0.0 || hz >= 29.0) {
        fprintf(stderr, "usage: %s [hz=6] [seconds=10]  (hz in (0,29))\n", argv[0]);
        return 1;
    }

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "XOpenDisplay failed — no X/XWayland display available\n");
        return 1;
    }
    int  scr  = DefaultScreen(dpy);
    int  W    = DisplayWidth(dpy, scr);
    int  H    = DisplayHeight(dpy, scr);
    GC   gc   = DefaultGC(dpy, scr);

    XSetWindowAttributes attr;
    attr.background_pixel  = BlackPixel(dpy, scr);
    attr.override_redirect = True;
    Window w = XCreateWindow(dpy, RootWindow(dpy, scr), 0, 0, W, H, 0, 0,
                             InputOutput, CopyFromParent,
                             CWBackPixel, &attr);

    unsigned long black = BlackPixel(dpy, scr);
    unsigned long white = WhitePixel(dpy, scr);

    XStoreName(dpy, w, "xflash");
    XMapWindow(dpy, w);
    XRaiseWindow(dpy, w);
    XFlush(dpy);
    fprintf(stderr, "[xflash] %dx%d  %g Hz  %.0f s\n", W, H, hz, duration);
    fprintf(stderr, "[xflash] window id: 0x%lx\n", (unsigned long)w);
    fprintf(stderr, "[xflash] capture it with:  ./detector 0x%lx 60\n",
            (unsigned long)w);

    struct timespec ts0, ts;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = (double)ts0.tv_sec + ts0.tv_nsec * 1e-9;
    const double period = 1.0 / hz;
    int prev_on = -1;

    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double t = (double)ts.tv_sec + ts.tv_nsec * 1e-9 - t0;
        if (t >= duration) break;
        int on = ((int)(t / period)) % 2 == 0;
        if (on != prev_on) {
            XSetForeground(dpy, gc, on ? white : black);
            XFillRectangle(dpy, w, gc, 0, 0, W, H);
            XFlush(dpy);
            prev_on = on;
        }
        struct timespec sl = { 0, 10 * 1000 * 1000 };  /* 10 ms slice */
        nanosleep(&sl, NULL);
    }

    XDestroyWindow(dpy, w);
    XCloseDisplay(dpy);
    fprintf(stderr, "[xflash] done\n");
    return 0;
}
