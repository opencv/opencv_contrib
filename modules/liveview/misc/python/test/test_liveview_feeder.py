import importlib.util
import os
import sys
import threading
import time
import types
import unittest

import numpy as np


MODULE_PATH = os.path.join(os.path.dirname(__file__),
                           "..", "package", "liveview", "__init__.py")


class FakeServer:
    def __init__(self, host, port, enable_webrtc):
        self.host = host
        self.port = 41000 if port == 0 else port
        self.enable_webrtc = enable_webrtc
        self.running = False
        self.published = []
        self.viewers = {}
        self.ever_viewer = False
        self.last_connected = 0
        self.last_disconnected = 0
        self.last_activity = 0
        self.lock = threading.Lock()

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def url(self):
        return "http://{}:{}/".format(self.host, self.port)

    def publish(self, name, frame):
        with self.lock:
            self.published.append((name, np.array(frame).copy()))

    def viewerCount(self, name=""):
        with self.lock:
            if not name:
                return sum(self.viewers.values())
            return self.viewers.get(name, 0)

    def hasViewers(self):
        return self.viewerCount() > 0

    def hasEverHadViewer(self):
        with self.lock:
            return self.ever_viewer

    def lastViewerConnectedTick(self):
        with self.lock:
            return self.last_connected

    def lastViewerDisconnectedTick(self):
        with self.lock:
            return self.last_disconnected

    def lastViewerActivityTick(self):
        with self.lock:
            return self.last_activity

    def connect_viewer(self, name):
        with self.lock:
            self.viewers[name] = self.viewers.get(name, 0) + 1
            self.ever_viewer = True
            self.last_connected += 1
            self.last_activity += 1

    def disconnect_viewer(self, name):
        with self.lock:
            if self.viewers.get(name, 0) > 1:
                self.viewers[name] -= 1
            else:
                self.viewers.pop(name, None)
            self.last_disconnected += 1
            self.last_activity += 1


class FakeLiveView:
    def __init__(self):
        self.servers = []
        self.fail_webrtc = False

    def createServer(self, host="127.0.0.1", port=0, enableWebRTC=False):
        if enableWebRTC and self.fail_webrtc:
            raise RuntimeError("webrtc unavailable")
        server = FakeServer(host, port, enableWebRTC)
        self.servers.append(server)
        return server


class FakeCapture:
    instances = []

    def __init__(self, device):
        self.device = device
        self.released = False
        self.frames = 0
        self.instances.append(self)

    def isOpened(self):
        return True

    def read(self):
        self.frames += 1
        return True, np.full((3, 4, 3), self.frames, np.uint8)

    def release(self):
        self.released = True


def load_liveview_module():
    fake_cv = types.SimpleNamespace()
    fake_cv.liveview = FakeLiveView()
    FakeCapture.instances = []
    fake_cv.VideoCapture = FakeCapture

    previous_cv2 = sys.modules.get("cv2")
    previous_liveview = sys.modules.get("cv2.liveview")
    sys.modules["cv2"] = fake_cv

    name = "cv2.liveview_test_support"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2
        if previous_liveview is None:
            sys.modules.pop("cv2.liveview", None)
        else:
            sys.modules["cv2.liveview"] = previous_liveview
    module.cv = fake_cv
    return module, fake_cv


class LiveViewFeederTests(unittest.TestCase):
    def test_show_publishes_single_frame_and_closes(self):
        lv, fake_cv = load_liveview_module()
        counter = {"value": 0}

        def feed():
            counter["value"] += 1
            return np.full((2, 3, 3), counter["value"], np.uint8)

        session = lv.show(feed, name="overlay", mode="mjpeg", show=False,
                          wait=True, fps=60, close_when_idle=False)
        self.assertEqual("ready", session.state)
        self.assertIn("overlay", session.channels)
        self.assertEqual("frames:overlay", session.source)
        self.assertTrue(session.id)
        self.assertGreaterEqual(len(fake_cv.liveview.servers[0].published), 1)
        self.assertIn("server", session.urls)
        self.assertIn("/stream/overlay.mjpeg", session.mjpeg_url)
        self.assertIn("/frame/overlay.jpg", session.snapshot_url)
        session.close()
        self.assertEqual("closed", session.state)
        self.assertFalse(fake_cv.liveview.servers[0].running)

    def test_show_publishes_multi_channel_dict(self):
        lv, fake_cv = load_liveview_module()

        def feed():
            return {
                "raw": np.zeros((2, 3, 3), np.uint8),
                "mask": np.ones((2, 3), np.uint8),
            }

        session = lv.show(feed, name="raw", mode="mjpeg", show=False,
                          wait=True, close_when_idle=False)
        session.close()
        published = [name for name, _ in fake_cv.liveview.servers[0].published]
        self.assertIn("raw", published)
        self.assertIn("mask", published)
        self.assertEqual(["mask", "raw"], session.channels)

    def test_none_keeps_session_alive_until_frame_arrives(self):
        lv, _ = load_liveview_module()
        counter = {"value": 0}

        def feed():
            counter["value"] += 1
            if counter["value"] < 3:
                return None
            return np.zeros((2, 3, 3), np.uint8)

        session = lv.show(feed, name="delayed", mode="mjpeg", show=False,
                          wait=True, fps=100, timeout=2, close_when_idle=False)
        self.assertEqual("ready", session.state)
        self.assertGreaterEqual(counter["value"], 3)
        session.close()

    def test_feeder_exception_fails_session(self):
        lv, _ = load_liveview_module()

        def feed():
            raise RuntimeError("pipeline failed")

        with self.assertRaises(lv.LiveViewError):
            lv.show(feed, name="bad", mode="mjpeg", show=False, wait=True,
                    timeout=1)

    def test_replace_closes_existing_session_for_same_source(self):
        lv, fake_cv = load_liveview_module()

        first = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                        name="same", mode="mjpeg", show=False, wait=True,
                        close_when_idle=False)
        second = lv.show(lambda: np.ones((2, 3, 3), np.uint8),
                         name="same", mode="mjpeg", show=False, wait=True,
                         close_when_idle=False)
        self.assertEqual("closed", first.state)
        self.assertEqual("ready", second.state)
        self.assertEqual(2, len(fake_cv.liveview.servers))
        second.close()

    def test_auto_falls_back_to_mjpeg_when_webrtc_server_fails(self):
        lv, fake_cv = load_liveview_module()
        fake_cv.liveview.fail_webrtc = True

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="auto", mode="auto", show=False, wait=True,
                          close_when_idle=False)
        self.assertEqual("mjpeg", session.transport)
        self.assertFalse(fake_cv.liveview.servers[0].enable_webrtc)
        session.close()

    def test_session_generates_notebook_html(self):
        lv, _ = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="html", mode="mjpeg", show=False, wait=True,
                          public_host="192.168.50.253", close_when_idle=False)
        body = session.html()
        self.assertIn("LiveView html", body)
        self.assertIn("<img", body)
        self.assertIn("http://192.168.50.253:41000/stream/html.mjpeg", body)
        self.assertEqual(body, session._repr_html_())
        session.close()

    def test_webrtc_html_uses_iframe(self):
        lv, _ = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="rtc", mode="webrtc", show=False, wait=True,
                          close_when_idle=False)
        body = session.html()
        self.assertIn("<iframe", body)
        self.assertIn("/webrtc/rtc", body)
        session.close()

    def test_close_all_closes_registered_sessions(self):
        lv, fake_cv = load_liveview_module()

        first = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                        name="a", mode="mjpeg", show=False, wait=True,
                        close_when_idle=False)
        second = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                         name="b", mode="mjpeg", show=False, wait=True,
                         close_when_idle=False)
        self.assertEqual(2, len(lv.sessions()))
        lv.close_all()
        self.assertEqual("closed", first.state)
        self.assertEqual("closed", second.state)
        self.assertEqual({}, lv.sessions())
        self.assertFalse(fake_cv.liveview.servers[0].running)
        self.assertFalse(fake_cv.liveview.servers[1].running)

    def test_camera_is_wrapper_over_show_and_releases_capture(self):
        lv, _ = load_liveview_module()

        session = lv.camera(0, mode="mjpeg", show=False, wait=True,
                            close_when_idle=False)
        self.assertEqual(1, len(FakeCapture.instances))
        capture = FakeCapture.instances[0]
        self.assertFalse(capture.released)
        session.close()
        self.assertTrue(capture.released)

    def test_show_does_not_release_capture_from_closure(self):
        lv, _ = load_liveview_module()
        capture = FakeCapture(0)

        def feed():
            ok, frame = capture.read()
            return frame if ok else None

        session = lv.show(feed, name="closure", mode="mjpeg", show=False,
                          wait=True, close_when_idle=False)
        self.assertFalse(capture.released)
        session.close()
        self.assertFalse(capture.released)
        capture.release()

    def test_show_does_not_release_capture_from_referenced_global(self):
        lv, _ = load_liveview_module()
        global GLOBAL_TEST_CAPTURE
        GLOBAL_TEST_CAPTURE = FakeCapture(0)

        def feed():
            ok, frame = GLOBAL_TEST_CAPTURE.read()
            return frame if ok else None

        session = lv.show(feed, name="global", mode="mjpeg", show=False,
                          wait=True, close_when_idle=False)
        self.assertFalse(GLOBAL_TEST_CAPTURE.released)
        session.close()
        self.assertFalse(GLOBAL_TEST_CAPTURE.released)
        GLOBAL_TEST_CAPTURE.release()
        del GLOBAL_TEST_CAPTURE

    def test_replace_disowns_previous_feeder_but_not_capture(self):
        lv, _ = load_liveview_module()
        first_capture = FakeCapture(0)
        second_capture = FakeCapture(0)

        def first_feed():
            ok, frame = first_capture.read()
            return frame if ok else None

        def second_feed():
            ok, frame = second_capture.read()
            return frame if ok else None

        first = lv.show(first_feed, name="replace_capture", mode="mjpeg",
                        show=False, wait=True, close_when_idle=False)
        self.assertFalse(first_capture.released)
        second = lv.show(second_feed, name="replace_capture", mode="mjpeg",
                         show=False, wait=True, close_when_idle=False)
        self.assertEqual("closed", first.state)
        self.assertFalse(first_capture.released)
        self.assertFalse(second_capture.released)
        second.close()
        self.assertFalse(second_capture.released)
        first_capture.release()
        second_capture.release()

    def test_session_closes_when_no_viewer_connects(self):
        lv, _ = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="unseen", mode="mjpeg", show=False, wait=True,
                          close_when_idle=True, connect_timeout=0.05)
        time.sleep(0.25)
        self.assertEqual("closed", session.state)
        self.assertEqual({}, lv.sessions())

    def test_session_closes_after_viewer_disconnect_idle_timeout(self):
        lv, fake_cv = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="idle", mode="mjpeg", show=False, wait=True,
                          close_when_idle=True, idle_timeout=0.05,
                          connect_timeout=2.0)
        server = fake_cv.liveview.servers[0]
        server.connect_viewer("idle")
        time.sleep(0.1)
        self.assertNotEqual("closed", session.state)
        server.disconnect_viewer("idle")
        time.sleep(0.25)
        self.assertEqual("closed", session.state)
        self.assertFalse(server.running)

    def test_reconnect_cancels_idle_close(self):
        lv, fake_cv = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="reconnect", mode="mjpeg", show=False, wait=True,
                          close_when_idle=True, idle_timeout=0.2,
                          connect_timeout=2.0)
        server = fake_cv.liveview.servers[0]
        server.connect_viewer("reconnect")
        server.disconnect_viewer("reconnect")
        time.sleep(0.05)
        server.connect_viewer("reconnect")
        time.sleep(0.25)
        self.assertNotEqual("closed", session.state)
        session.close()

    def test_camera_idle_close_releases_capture(self):
        lv, fake_cv = load_liveview_module()

        session = lv.camera(0, mode="mjpeg", show=False, wait=True,
                            close_when_idle=True, idle_timeout=0.05,
                            connect_timeout=2.0)
        capture = FakeCapture.instances[0]
        server = fake_cv.liveview.servers[0]
        server.connect_viewer("camera")
        server.disconnect_viewer("camera")
        time.sleep(0.25)
        self.assertEqual("closed", session.state)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
