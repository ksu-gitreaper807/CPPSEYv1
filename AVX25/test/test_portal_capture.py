"""Run with python3 -B -m unittest discover -s test -p test_portal_capture.py -v.

Pure protocol/layout tests do not require GI. GI-dependent tests skip explicitly
when unavailable; live videotestsrc tests use ordinary pipes and no display/bus.
"""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'pse_portal_capture.py'
SPEC = importlib.util.spec_from_file_location('portal_capture_under_test', SCRIPT)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class ProtocolTests(unittest.TestCase):
    def test_import_does_not_load_gi_or_capture(self):
        code = ("import importlib.util,sys; "
                "s=importlib.util.spec_from_file_location('capture',sys.argv[1]); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert 'gi' not in sys.modules; assert m.Gst is None")
        result = subprocess.run([sys.executable, '-S', '-B', '-c', code, str(SCRIPT)],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b'')
        self.assertEqual(result.stderr, b'')

    def test_exact_little_endian_header(self):
        timestamp = 0x0102030405060708
        header = capture.frame_header(320 * 240 * 4, timestamp)
        self.assertEqual(len(header), 16)
        self.assertEqual(header, struct.pack('<4sIQ', b'PSE1', 307200, timestamp))
        self.assertEqual(header[8:], b'\x08\x07\x06\x05\x04\x03\x02\x01')

    def test_invalid_header_ranges(self):
        for length, timestamp in ((0, 0), (1 << 32, 0), (4, -1),
                                  (4, capture.CLOCK_TIME_NONE)):
            with self.subTest(length=length, timestamp=timestamp):
                with self.assertRaises(capture.CaptureError):
                    capture.frame_header(length, timestamp)

    def test_tightly_packed_rows(self):
        self.assertEqual(capture.pack_bgrx_rows(bytes(range(16)), 2, 2, 0, 8),
                         bytes(range(16)))

    def test_padding_and_offset_removed(self):
        data = b'HEAD' + b'abcdefgh' + b'PAD!' + b'ijklmnop' + b'TAIL'
        self.assertEqual(capture.pack_bgrx_rows(data, 2, 2, 4, 12), b'abcdefghijklmnop')

    def test_negative_stride(self):
        self.assertEqual(capture.pack_bgrx_rows(b'ijklmnopPAD!abcdefgh', 2, 2, 12, -12),
                         b'abcdefghijklmnop')

    def test_bad_layouts_rejected(self):
        for data, offset, stride in ((bytes(15), 0, 8), (bytes(32), 0, 7),
                                      (bytes(32), -1, 8), (bytes(32), 0, -8),
                                      (bytes(32), 24, 8)):
            with self.subTest(offset=offset, stride=stride):
                with self.assertRaises(capture.CaptureError):
                    capture.pack_bgrx_rows(data, 2, 2, offset, stride)

    def test_odd_empty_and_oversized_geometry_rejected(self):
        for width, height in ((3, 2), (2, 3), (0, 2), (-2, 2), (65536, 65536)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(capture.CaptureError):
                    capture.validate_geometry(width, height)

    def test_preserves_actual_pts_and_gaps(self):
        previous = None
        actual = []
        for pts in (0, 11_000_001, 92_345_678, 5_000_000_000):
            previous = capture.normalize_timestamp(pts, previous)
            actual.append(previous)
        self.assertEqual(actual, [0, 11_000_001, 92_345_678, 5_000_000_000])

    def test_timestamp_fallback_and_strict_order(self):
        with mock.patch.object(capture.time, 'monotonic_ns', return_value=1000) as clock:
            self.assertEqual(capture.normalize_timestamp(0), 0)
            clock.assert_not_called()
            self.assertEqual(capture.normalize_timestamp(None), 1000)
            self.assertEqual(capture.normalize_timestamp(capture.CLOCK_TIME_NONE), 1000)
            self.assertEqual(capture.normalize_timestamp(900, 1000), 1001)
            self.assertEqual(capture.normalize_timestamp(1000, 1000), 1001)
            self.assertEqual(capture.normalize_timestamp(None, 2000), 2001)
        with self.assertRaises(capture.CaptureError):
            capture.normalize_timestamp(0, capture.CLOCK_TIME_NONE - 1)

    def test_partial_and_blocked_pipe_writes(self):
        pump = mock.Mock()
        with mock.patch.object(capture.os, 'write', side_effect=[BlockingIOError(), 2, 3]) as write:
            with mock.patch.object(capture.select, 'select') as wait:
                capture.write_all(123, b'12345', pump)
        self.assertEqual(bytes(write.call_args_list[-1].args[1]), b'345')
        wait.assert_called_once_with([], [123], [], 0.05)
        self.assertEqual(pump.call_count, 3)

    def test_arguments_report_structured_error(self):
        for args in (['--fps', '0'], ['--fps', 'nan'], ['--frames', '2'],
                     ['--test-source', '--frames', '-1'], ['--unknown']):
            with self.subTest(args=args):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(capture.main(args), 1)
                self.assertTrue(stderr.getvalue().startswith('ERROR arguments:'))
                self.assertEqual(len(stderr.getvalue().splitlines()), 1)

    def test_missing_python_gi_reports_structured_error(self):
        result = subprocess.run([sys.executable, '-S', '-B', str(SCRIPT),
                                 '--test-source', '--frames', '1'],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b'')
        self.assertTrue(result.stderr.startswith(b'ERROR dependencies:'), result.stderr)
        self.assertNotIn(b'Traceback', result.stderr)


class WithGStreamer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            capture.load_dependencies(test_source=True)
        except capture.CaptureError as exc:
            raise unittest.SkipTest(str(exc))


class FakeBus:
    def __init__(self):
        self.callbacks = {}
        self.unsubscribed = []
        self.closed = []
        self.next_id = 1

    def get_unique_name(self):
        return ':1.42'

    def signal_subscribe(self, _name, interface, _signal, path, _arg, _flags, callback, _data):
        ident = self.next_id
        self.next_id += 1
        self.callbacks[ident] = (interface, path, callback)
        return ident

    def signal_unsubscribe(self, ident):
        self.callbacks.pop(ident)
        self.unsubscribed.append(ident)

    def call_sync(self, _name, path, interface, method, *_args):
        self.closed.append((path, interface, method))

    def respond(self, path, code, data):
        params = capture.GLib.Variant('(ua{sv})', (code, data))
        for interface, subscribed_path, callback in list(self.callbacks.values()):
            if interface == capture.REQUEST_IFACE and subscribed_path in (None, path):
                callback(self, capture.PORTAL_NAME, path, interface, 'Response', params, None)


class PortalRequestTests(WithGStreamer):
    def setUp(self):
        self.bus = FakeBus()
        self.portal = mock.Mock()
        self.path = capture.PORTAL_PATH + '/request/legacy/returned'
        self.sources = []
        real_timeout_add = capture.GLib.timeout_add

        def record_timeout(*args):
            ident = real_timeout_add(*args)
            self.sources.append(ident)
            return ident

        self.addCleanup(mock.patch.stopall)
        mock.patch.object(capture.GLib, 'timeout_add', side_effect=record_timeout).start()

    def tearDown(self):
        self.assertEqual(self.bus.callbacks, {})
        context = capture.GLib.MainContext.default()
        for ident in self.sources:
            self.assertIsNone(context.find_source_by_id(ident), 'request timer leaked')

    def request(self, **kwargs):
        return capture.portal_request(self.bus, self.portal, 'Start', '(osa{sv})',
                                      ('/session', ''), {}, timeout_s=0.03, **kwargs)

    def reply(self, code=0, data=None, delayed=False):
        def call(*_args):
            self.assertTrue(self.bus.callbacks, 'method invoked before subscription')
            self.bus.respond(capture.PORTAL_PATH + '/request/unrelated', 2, {})

            def respond():
                self.bus.respond(self.path, code, data or {})
                return False

            if delayed:
                capture.GLib.idle_add(respond)
            else:
                respond()  # Response before method return, including changed path.
            return capture.GLib.Variant('(o)', (self.path,))
        self.portal.call_sync.side_effect = call

    def test_early_response_on_returned_path(self):
        self.reply(data={'value': capture.GLib.Variant('u', 42)})
        self.assertEqual(self.request(), {'value': 42})
        self.assertEqual(self.bus.closed, [])

    def test_delayed_response(self):
        self.reply(delayed=True)
        self.assertEqual(self.request(), {})

    def test_cancelled_request_closed(self):
        self.reply(code=1)
        with self.assertRaisesRegex(capture.CaptureError, 'cancelled'):
            self.request()
        self.assertIn((self.path, capture.REQUEST_IFACE, 'Close'), self.bus.closed)

    def test_timeout_closed_and_sources_removed(self):
        self.portal.call_sync.return_value = capture.GLib.Variant('(o)', (self.path,))
        with self.assertRaisesRegex(capture.CaptureError, 'timed out'):
            self.request()
        self.assertIn((self.path, capture.REQUEST_IFACE, 'Close'), self.bus.closed)

    def test_method_failure_cleans_subscription_timer_and_request(self):
        self.portal.call_sync.side_effect = RuntimeError('DBus failure')
        with self.assertRaisesRegex(RuntimeError, 'DBus failure'):
            self.request()
        self.assertEqual(self.bus.closed[0][1:], (capture.REQUEST_IFACE, 'Close'))

    def test_signal_cancels_request(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(capture.CaptureStopped):
            self.request(stop=stop)
        self.portal.call_sync.assert_not_called()
        self.assertEqual(self.bus.closed[0][1:], (capture.REQUEST_IFACE, 'Close'))

    def test_session_closure_aborts_pending_request(self):
        self.portal.call_sync.return_value = capture.GLib.Variant('(o)', (self.path,))
        session = mock.Mock(closed=False)
        capture.GLib.idle_add(lambda: (setattr(session, 'closed', True), False)[1])
        with self.assertRaisesRegex(capture.CaptureError, 'session closed'):
            self.request(session=session)

    def test_session_stage_failure_closes_session(self):
        path = capture.PORTAL_PATH + '/session/1_42/test'
        session = capture.PortalSession(self.bus, threading.Event())
        with mock.patch.object(capture.Gio.DBusProxy, 'new_sync', return_value=self.portal):
            with mock.patch.object(capture, 'portal_request', side_effect=[
                    {'session_handle': path}, capture.CaptureError('SelectSources failed')]):
                with self.assertRaisesRegex(capture.CaptureError, 'SelectSources'):
                    session.open()
        self.assertIn((path, capture.SESSION_IFACE, 'Close'), self.bus.closed)
        self.assertTrue(session.closed)

    def test_session_owns_fd_and_ignores_portal_logical_size(self):
        path = capture.PORTAL_PATH + '/session/1_42/test'
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, write_fd)
        session = capture.PortalSession(self.bus, threading.Event())
        self.addCleanup(session.close)
        self.portal.call_with_unix_fd_list_sync.return_value = (
            capture.GLib.Variant('(h)', (0,)), mock.Mock(get=mock.Mock(return_value=read_fd)))
        with mock.patch.object(capture.Gio.DBusProxy, 'new_sync', return_value=self.portal):
            with mock.patch.object(capture, 'portal_request', side_effect=[
                    {'session_handle': path}, {},
                    {'streams': [(7, {'size': (111, 77)})]}]):
                session.open()
        self.assertEqual(session.node_id, 7)
        self.assertEqual(session.pw_fd, read_fd)
        os.fstat(read_fd)
        session.close()
        session.close()  # Idempotent: do not close an FD twice.
        with self.assertRaises(OSError):
            os.fstat(read_fd)
        self.assertEqual(self.bus.closed.count((path, capture.SESSION_IFACE, 'Close')), 1)


class SampleTests(WithGStreamer):
    def sample(self, data, width=2, height=2, offset=None, stride=None, fmt='BGRx'):
        caps = capture.Gst.Caps.from_string(
            f'video/x-raw,format={fmt},width={width},height={height},framerate=30/1')
        buffer = capture.Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        buffer.pts = 123456789
        if offset is not None:
            capture.GstVideo.buffer_add_video_meta_full(
                buffer, capture.GstVideo.VideoFrameFlags.NONE,
                capture.GstVideo.VideoFormat.BGRX, width, height, 1,
                [offset, 0, 0, 0], [stride, 0, 0, 0])
        return capture.Gst.Sample.new(buffer, caps, None, None)

    def test_caps_layout_and_pts(self):
        self.assertEqual(capture.sample_frame(self.sample(b'abcdefghijklmnop')),
                         (2, 2, b'abcdefghijklmnop', 123456789))

    def test_video_meta_overrides_caps_stride(self):
        sample = self.sample(b'HEADabcdefghPAD!ijklmnop', offset=4, stride=12)
        self.assertEqual(capture.sample_frame(sample),
                         (2, 2, b'abcdefghijklmnop', 123456789))

    def test_real_sample_short_buffer_and_odd_geometry_fail(self):
        for sample in (self.sample(bytes(15)), self.sample(bytes(24), width=3),
                       self.sample(bytes(16), fmt='RGBx')):
            with self.assertRaises(capture.CaptureError):
                capture.sample_frame(sample)


class TestSourceIntegrationTests(WithGStreamer):
    def decode(self, data):
        frames = []
        position = 0
        while position < len(data):
            self.assertGreaterEqual(len(data) - position, 16, 'truncated header')
            magic, size, pts = struct.unpack_from('<4sIQ', data, position)
            self.assertEqual(magic, b'PSE1')
            self.assertEqual(size, 320 * 240 * 4)
            position += 16
            payload = data[position:position + size]
            self.assertEqual(len(payload), size, 'truncated payload')
            frames.append((pts, payload))
            position += size
        return frames

    def test_live_source_finite_protocol_and_requested_fps(self):
        for fps in (12, 30):
            with self.subTest(fps=fps):
                result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--test-source',
                                         '--frames', '4', '--fps', str(fps)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, b'READY 320x240\n')
                frames = self.decode(result.stdout)
                # Both queue and appsink intentionally drop old buffers under
                # backpressure; four source buffers need not all reach stdout.
                self.assertGreaterEqual(len(frames), 2)
                self.assertLessEqual(len(frames), 4)
                timestamps = [pts for pts, _ in frames]
                self.assertGreater(timestamps[0], 0, 'live source PTS must not be frame index zero')
                for first, second in zip(timestamps, timestamps[1:]):
                    periods = (second - first) * fps / 1_000_000_000
                    self.assertGreaterEqual(round(periods), 1)
                    self.assertAlmostEqual(periods, round(periods), delta=1e-6)
                self.assertGreater(len(set(frames[0][1])), 1, 'expected actual test pixels')

    def test_single_frame_does_not_wait_for_async_done(self):
        result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--test-source',
                                 '--frames', '1', '--fps', '1'],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b'READY 320x240\n')
        self.assertEqual(len(self.decode(result.stdout)), 1)

    def test_full_ordinary_pipe_remains_interruptible(self):
        with subprocess.Popen([sys.executable, '-B', str(SCRIPT), '--test-source'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                readable, _, _ = select.select([process.stderr], [], [], 5)
                self.assertTrue(readable, 'helper did not announce readiness')
                self.assertEqual(process.stderr.readline(), b'READY 320x240\n')
                # Leave stdout unread: a 307200-byte payload fills an ordinary pipe.
                process.terminate()
                self.assertEqual(process.wait(timeout=3), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3)

    def test_closed_parent_pipe_exits_cleanly(self):
        with subprocess.Popen([sys.executable, '-B', str(SCRIPT), '--test-source'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                process.stdout.close()
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertNotIn(b'ERROR', process.stderr.read())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3)

    def test_videorate_drops_without_rewriting_pts(self):
        pipeline = capture.capture_pipeline(20, True, 12, None)
        elements = []
        iterator = pipeline.iterate_elements()
        while True:
            result, element = iterator.next()
            if result == capture.Gst.IteratorResult.DONE:
                break
            self.assertEqual(result, capture.Gst.IteratorResult.OK)
            elements.append(element)
        rate = next(e for e in elements if e.get_factory().get_name() == 'videorate')
        input_caps = next(e for e in elements if e.get_factory().get_name() == 'capsfilter'
                          and e.get_property('caps').get_structure(0).has_field('framerate'))
        input_caps.set_property('caps', capture.Gst.Caps.from_string(
            'video/x-raw,width=320,height=240,framerate=60/1'))
        sink = pipeline.get_by_name('asink')
        input_pts = []

        def record(_pad, info):
            input_pts.append(info.get_buffer().pts)
            return capture.Gst.PadProbeReturn.OK

        rate.get_static_pad('sink').add_probe(capture.Gst.PadProbeType.BUFFER, record)
        output_pts = []
        try:
            pipeline.set_state(capture.Gst.State.PLAYING)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                sample = sink.try_pull_sample(100 * capture.Gst.MSECOND)
                if sample is not None:
                    output_pts.append(sample.get_buffer().pts)
                elif sink.is_eos():
                    break
            else:
                self.fail('rate-limited pipeline did not reach EOS')
            self.assertEqual(len(input_pts), 12)
            self.assertGreater(len(output_pts), 1)
            self.assertLess(len(output_pts), len(input_pts))
            self.assertTrue(set(output_pts).issubset(input_pts), 'source PTS were rewritten')
            self.assertEqual(rate.get_property('duplicate'), 0)
            self.assertEqual(rate.get_property('max-rate'), 20)
            self.assertTrue(rate.get_property('drop-only'))
            self.assertEqual(sink.get_property('max-buffers'), 1)
            self.assertTrue(sink.get_property('drop'))
            self.assertFalse(sink.get_property('async'))
        finally:
            pipeline.set_state(capture.Gst.State.NULL)


if __name__ == '__main__':
    unittest.main()
