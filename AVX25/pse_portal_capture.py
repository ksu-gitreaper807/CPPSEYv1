#!/usr/bin/env python3
"""Generic xdg ScreenCast/PipeWire capture for the detector's ordinary pipe.

stderr: READY <width>x<height>\n after the first valid negotiated sample;
        ERROR <message>\n on failure (exit status 1).
stdout: struct.pack('<4sIQ', b'PSE1', width * height * 4, pts_ns), then
        tightly packed BGRx rows, repeated until EOS, closure, or a signal.

Dimensions are even, physical sample dimensions, never portal logical size.
PTS comes from Gst.Buffer.pts, falling back to time.monotonic_ns() only when
absent. Repeated/backward timestamps advance by one ns to stay strictly ordered.
Requires PyGObject and Gio, Gst, GstApp, GstVideo introspection, GStreamer base
plugins and pipewiresrc, plus an xdg ScreenCast portal backend.
--test-source [--frames N] uses live 320x240 videotestsrc without a display.

Importing this module does not load GI, initialize GStreamer or open a portal.
"""

import argparse
import os
import secrets
import select
import signal
import struct
import sys
import threading
import time


FRAME_HEADER = struct.Struct('<4sIQ')
CLOCK_TIME_NONE = (1 << 64) - 1
PORTAL_NAME = 'org.freedesktop.portal.Desktop'
PORTAL_PATH = '/org/freedesktop/portal/desktop'
REQUEST_IFACE = 'org.freedesktop.portal.Request'
SESSION_IFACE = 'org.freedesktop.portal.Session'
Gio = GLib = Gst = GstApp = GstVideo = None


class CaptureError(RuntimeError):
    pass


class CaptureStopped(Exception):
    pass


def eprint(message):
    # Protocol diagnostics remain single lines even for GLib exceptions.
    print(' '.join(str(message).splitlines()), file=sys.stderr, flush=True)


def load_dependencies(test_source=False):
    global Gio, GLib, Gst, GstApp, GstVideo
    try:
        import gi
        for namespace, version in (('Gio', '2.0'), ('Gst', '1.0'),
                                   ('GstApp', '1.0'), ('GstVideo', '1.0')):
            gi.require_version(namespace, version)
        from gi.repository import Gio, GLib, Gst, GstApp, GstVideo
        Gst.init(None)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise CaptureError(f'dependencies: Python GI/GStreamer unavailable: {exc}') from exc
    required = ['videorate', 'videoconvert', 'queue', 'appsink',
                'videotestsrc' if test_source else 'pipewiresrc']
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        raise CaptureError('dependencies: missing GStreamer elements: ' + ', '.join(missing))


def validate_geometry(width, height):
    if width <= 0 or height <= 0:
        raise CaptureError(f'invalid sample dimensions {width}x{height}')
    if width % 2 or height % 2:
        raise CaptureError(f'odd sample dimensions {width}x{height}; even BGRx dimensions required')
    if width * height * 4 > 0xffffffff:
        raise CaptureError('sample exceeds the protocol uint32 payload limit')


def pack_bgrx_rows(data, width, height, offset, stride):
    """Remove allocation padding; support signed strides and nonzero offsets."""
    validate_geometry(width, height)
    row_bytes = width * 4
    if abs(stride) < row_bytes:
        raise CaptureError(f'invalid BGRx stride {stride} for width {width}')
    view = memoryview(data).cast('B')
    last_row = offset + (height - 1) * stride
    if min(offset, last_row) < 0 or max(offset, last_row) + row_bytes > len(view):
        raise CaptureError('BGRx buffer is shorter than its offset/stride layout')
    if stride == row_bytes:
        return bytes(view[offset:offset + row_bytes * height])
    return b''.join(view[offset + y * stride:offset + y * stride + row_bytes]
                    for y in range(height))


def normalize_timestamp(pts, previous=None):
    timestamp = time.monotonic_ns() if pts is None or pts == CLOCK_TIME_NONE else int(pts)
    if previous is not None:
        timestamp = max(timestamp, previous + 1)
    if not 0 <= timestamp < CLOCK_TIME_NONE:
        raise CaptureError('timestamp is outside the valid uint64 nanosecond range')
    return timestamp


def frame_header(payload_length, pts_ns):
    if not 0 < payload_length <= 0xffffffff or not 0 <= pts_ns < CLOCK_TIME_NONE:
        raise CaptureError('invalid frame header length or timestamp')
    return FRAME_HEADER.pack(b'PSE1', payload_length, pts_ns)


def close_portal_object(bus, path, interface):
    if path is None:
        return
    try:
        bus.call_sync(PORTAL_NAME, path, interface, 'Close', None, None,
                      Gio.DBusCallFlags.NONE, 1000, None)
    except Exception:
        # Requests/sessions may already have disappeared after Response/Closed.
        pass


def portal_request(bus, portal, method, signature, arguments, options,
                   timeout_s=30, stop=None, session=None):
    """Subscribe before invoking; retain early replies even on legacy paths.

    https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.Request.html
    A broad subscription is deliberate: the returned handle can differ from the
    predicted token path, including when Response precedes the method reply.
    """
    token = 'pse' + secrets.token_hex(16)
    sender = bus.get_unique_name().lstrip(':').replace('.', '_')
    path = f'{PORTAL_PATH}/request/{sender}/{token}'
    options = dict(options, handle_token=GLib.Variant('s', token))
    responses = {}
    loop = GLib.MainLoop()
    deadline = time.monotonic() + timeout_s
    failure = None
    success = False
    subscription = timeout_id = None

    def on_response(_bus, _sender, object_path, _iface, _signal, params, _data):
        responses[object_path] = params.unpack()
        if object_path == path:
            loop.quit()

    def check_wait():
        nonlocal failure
        if stop is not None and stop.is_set():
            failure = CaptureStopped()
        elif session is not None and session.closed:
            failure = CaptureError(f'{method}: portal session closed')
        elif time.monotonic() >= deadline:
            failure = CaptureError(f'{method}: request timed out after {timeout_s:g}s')
        if failure is not None:
            loop.quit()
        return True  # Removed explicitly in finally, including on timeout.

    try:
        subscription = bus.signal_subscribe(
            PORTAL_NAME, REQUEST_IFACE, 'Response', None, None,
            Gio.DBusSignalFlags.NONE, on_response, None)
        timeout_id = GLib.timeout_add(max(1, min(50, int(timeout_s * 1000))), check_wait)
        check_wait()
        if failure is not None:
            raise failure
        reply = portal.call_sync(method, GLib.Variant(signature, (*arguments, options)),
                                 Gio.DBusCallFlags.NONE, 5000, None)
        returned_path = reply.unpack()[0]
        if not isinstance(returned_path, str) or not returned_path.startswith(PORTAL_PATH + '/request/'):
            raise CaptureError(f'{method}: invalid returned request handle {returned_path!r}')
        path = returned_path
        check_wait()
        if path not in responses and failure is None:
            loop.run()
        if failure is not None:
            raise failure
        code, data = responses[path]
        if code != 0:
            reason = 'cancelled' if code == 1 else 'failed'
            raise CaptureError(f'{method}: {reason} (response {code})')
        success = True
        return data
    finally:
        if timeout_id is not None:
            GLib.source_remove(timeout_id)
        if subscription is not None:
            bus.signal_unsubscribe(subscription)
        if not success:
            close_portal_object(bus, path, REQUEST_IFACE)


class PortalSession:
    """Own the session, Closed subscription and duplicated PipeWire FD."""

    def __init__(self, bus, stop):
        self.bus = bus
        self.stop = stop
        self.portal = None
        self.path = None
        self.subscription = None
        self.pw_fd = None
        self.node_id = None
        self.serial = None
        self.closed = False

    def _on_closed(self, _bus, _sender, path, _iface, _signal, _params, _data):
        if path == self.path:
            self.closed = True

    def open(self):
        try:
            self.portal = Gio.DBusProxy.new_sync(
                self.bus, Gio.DBusProxyFlags.NONE, None, PORTAL_NAME, PORTAL_PATH,
                'org.freedesktop.portal.ScreenCast', None)
            token = 'pse' + secrets.token_hex(16)
            sender = self.bus.get_unique_name().lstrip(':').replace('.', '_')
            # Keep the predicted handle for cleanup even if CreateSession fails.
            self.path = f'{PORTAL_PATH}/session/{sender}/{token}'
            self.subscription = self.bus.signal_subscribe(
                PORTAL_NAME, SESSION_IFACE, 'Closed', None, None,
                Gio.DBusSignalFlags.NONE, self._on_closed, None)

            def request(method, signature, args, options, timeout):
                return portal_request(self.bus, self.portal, method, signature,
                                      args, options, timeout, self.stop, self)

            data = request('CreateSession', '(a{sv})', (),
                           {'session_handle_token': GLib.Variant('s', token)}, 15)
            returned_path = data['session_handle']
            if not isinstance(returned_path, str) or not returned_path.startswith(PORTAL_PATH + '/session/'):
                raise CaptureError('CreateSession: invalid session handle')
            self.path = returned_path
            # Defaults are hidden cursor and no persistence, including old portals.
            request('SelectSources', '(oa{sv})', (self.path,),
                    {'types': GLib.Variant('u', 1),
                     'multiple': GLib.Variant('b', False)}, 30)
            data = request('Start', '(osa{sv})', (self.path, ''), {}, 180)
            streams = data.get('streams', [])
            if len(streams) != 1:
                raise CaptureError(f'Start: expected one stream, received {len(streams)}')
            self.node_id, properties = streams[0]
            self.node_id = int(self.node_id)
            self.serial = properties.get('pipewire-serial')
            # properties['size'] is compositor coordinates, not buffer pixels.
            reply, fd_list = self.portal.call_with_unix_fd_list_sync(
                'OpenPipeWireRemote', GLib.Variant('(oa{sv})', (self.path, {})),
                Gio.DBusCallFlags.NONE, 5000, None, None)
            if fd_list is None:
                raise CaptureError('OpenPipeWireRemote: no Unix file descriptors')
            self.pw_fd = fd_list.get(reply.unpack()[0])
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.pw_fd is not None:
                fd, self.pw_fd = self.pw_fd, None
                os.close(fd)
        finally:
            try:
                close_portal_object(self.bus, self.path, SESSION_IFACE)
            finally:
                self.path = None
                if self.subscription is not None:
                    self.bus.signal_unsubscribe(self.subscription)
                    self.subscription = None
                self.closed = True


def sample_frame(sample):
    caps, buffer = sample.get_caps(), sample.get_buffer()
    if caps is None or buffer is None:
        raise CaptureError('appsink sample lacks caps or buffer')
    info = GstVideo.VideoInfo.new_from_caps(caps)
    if info is None or info.finfo.format != GstVideo.VideoFormat.BGRX:
        raise CaptureError('appsink did not negotiate packed BGRx video')
    width, height = info.width, info.height
    validate_geometry(width, height)
    offset, stride = info.offset[0], info.stride[0]
    meta = GstVideo.buffer_get_video_meta(buffer)
    if meta is not None:
        if (meta.format != GstVideo.VideoFormat.BGRX or meta.n_planes != 1
                or meta.width != width or meta.height != height):
            raise CaptureError('BGRx video metadata disagrees with negotiated caps')
        offset, stride = meta.offset[0], meta.stride[0]
    ok, mapped = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise CaptureError('cannot map appsink video buffer for reading')
    try:
        payload = pack_bgrx_rows(mapped.data, width, height, offset, stride)
    finally:
        buffer.unmap(mapped)
    return width, height, payload, buffer.pts


def write_all(fd, data, pump):
    """Use a nonblocking ordinary pipe so signals/portal closure stay serviced."""
    remaining = memoryview(data)
    while remaining:
        pump()
        try:
            count = os.write(fd, remaining)
        except BlockingIOError:
            select.select([], [fd], [], 0.05)
            continue
        except InterruptedError:
            continue
        if count <= 0:
            raise CaptureError('stdout pipe made no write progress')
        remaining = remaining[count:]


def capture_pipeline(fps, test_source, frames, session):
    if test_source:
        limit = f' num-buffers={frames}' if frames is not None else ''
        source = (f'videotestsrc is-live=true{limit} '
                  f'! video/x-raw,width=320,height=240,framerate={fps}/1')
    else:
        # Preserve PipeWire's buffer timestamps instead of restamping on arrival.
        source = (f'pipewiresrc name=portal_source fd={session.pw_fd} path={session.node_id} '
                  'keepalive-time=1000 do-timestamp=false')
    pipeline = Gst.parse_launch(
        f'{source} ! videorate drop-only=true max-rate={fps} '
        '! videoconvert ! video/x-raw,format=BGRx '
        '! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream '
        '! appsink name=asink max-buffers=1 drop=true sync=false async=false wait-on-eos=false')
    if not test_source and session.serial is not None:
        source_element = pipeline.get_by_name('portal_source')
        # New portals provide a stable serial; older pipewiresrc uses node ID.
        if source_element.find_property('target-object'):
            source_element.set_property('target-object', str(int(session.serial)))
    return pipeline


def run_capture(fps, stop, test_source=False, frames=None, session=None):
    pipeline = capture_pipeline(fps, test_source, frames, session)
    output_fd = sys.stdout.fileno()
    was_blocking = os.get_blocking(output_fd)
    context = GLib.MainContext.default()
    gst_bus = pipeline.get_bus()
    eos = False
    geometry = None
    previous_pts = None

    def pump():
        nonlocal eos
        while context.pending():
            context.iteration(False)
        if stop.is_set():
            raise CaptureStopped()
        if session is not None and session.closed:
            raise CaptureError('portal session closed')
        while True:
            message = gst_bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is None:
                break
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise CaptureError(f'GStreamer: {error}; {debug or "no debug details"}')
            eos = True

    try:
        os.set_blocking(output_fd, False)
        sink = pipeline.get_by_name('asink')
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pump()  # Prefer the bus's concrete error to a generic state failure.
            raise CaptureError('GStreamer failed to enter PLAYING')
        startup_deadline = time.monotonic() + 10
        while True:
            pump()
            sample = sink.try_pull_sample(50 * Gst.MSECOND)
            pump()
            if sample is None:
                if eos or sink.is_eos():
                    if geometry is None:
                        raise CaptureError('GStreamer EOS before the first sample')
                    break
                if geometry is None and time.monotonic() >= startup_deadline:
                    raise CaptureError('GStreamer timed out waiting for first sample (10s)')
                continue
            width, height, payload, pts = sample_frame(sample)
            pts = normalize_timestamp(pts, previous_pts)
            if geometry is None:
                geometry = (width, height)
                eprint(f'READY {width}x{height}')
            elif geometry != (width, height):
                raise CaptureError(f'sample dimensions changed from {geometry[0]}x{geometry[1]} '
                                   f'to {width}x{height}')
            write_all(output_fd, frame_header(len(payload), pts), pump)
            write_all(output_fd, payload, pump)
            previous_pts = pts
    finally:
        try:
            pipeline.set_state(Gst.State.NULL)
        finally:
            os.set_blocking(output_fd, was_blocking)


class ProtocolArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise CaptureError('arguments: ' + message)


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('expected a positive integer') from None
    if not 1 <= number <= 2147483647:
        raise argparse.ArgumentTypeError('expected an integer in 1..2147483647')
    return number


def main(argv=None):
    session = None
    handlers = {}
    stop = threading.Event()
    try:
        parser = ProtocolArgumentParser(description=__doc__)
        parser.add_argument('--fps', type=positive_int, default=60)
        parser.add_argument('--test-source', action='store_true')
        parser.add_argument('--frames', type=positive_int)
        args = parser.parse_args(argv)
        if args.frames is not None and not args.test_source:
            raise CaptureError('arguments: --frames requires --test-source')
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda _sig, _frame: stop.set())
        load_dependencies(args.test_source)
        if not args.test_source:
            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            session = PortalSession(bus, stop)
            session.open()
        run_capture(args.fps, stop, args.test_source, args.frames, session)
        return 0
    except (CaptureStopped, BrokenPipeError):
        return 0
    except Exception as exc:
        eprint(f'ERROR {exc}')
        return 1
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
