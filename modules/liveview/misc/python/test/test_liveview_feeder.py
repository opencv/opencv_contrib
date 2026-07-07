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
                          wait=True, fps=60)
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

        session = lv.show(feed, name="raw", mode="mjpeg", show=False, wait=True)
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
                          wait=True, fps=100, timeout=2)
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
                        name="same", mode="mjpeg", show=False, wait=True)
        second = lv.show(lambda: np.ones((2, 3, 3), np.uint8),
                         name="same", mode="mjpeg", show=False, wait=True)
        self.assertEqual("closed", first.state)
        self.assertEqual("ready", second.state)
        self.assertEqual(2, len(fake_cv.liveview.servers))
        second.close()

    def test_auto_falls_back_to_mjpeg_when_webrtc_server_fails(self):
        lv, fake_cv = load_liveview_module()
        fake_cv.liveview.fail_webrtc = True

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="auto", mode="auto", show=False, wait=True)
        self.assertEqual("mjpeg", session.transport)
        self.assertFalse(fake_cv.liveview.servers[0].enable_webrtc)
        session.close()

    def test_session_generates_notebook_html(self):
        lv, _ = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="html", mode="mjpeg", show=False, wait=True,
                          public_host="192.168.50.253")
        body = session.html()
        self.assertIn("LiveView html", body)
        self.assertIn("<img", body)
        self.assertIn("http://192.168.50.253:41000/stream/html.mjpeg", body)
        self.assertEqual(body, session._repr_html_())
        session.close()

    def test_webrtc_html_uses_iframe(self):
        lv, _ = load_liveview_module()

        session = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                          name="rtc", mode="webrtc", show=False, wait=True)
        body = session.html()
        self.assertIn("<iframe", body)
        self.assertIn("/webrtc/rtc", body)
        session.close()

    def test_close_all_closes_registered_sessions(self):
        lv, fake_cv = load_liveview_module()

        first = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                        name="a", mode="mjpeg", show=False, wait=True)
        second = lv.show(lambda: np.zeros((2, 3, 3), np.uint8),
                         name="b", mode="mjpeg", show=False, wait=True)
        self.assertEqual(2, len(lv.sessions()))
        lv.close_all()
        self.assertEqual("closed", first.state)
        self.assertEqual("closed", second.state)
        self.assertEqual({}, lv.sessions())
        self.assertFalse(fake_cv.liveview.servers[0].running)
        self.assertFalse(fake_cv.liveview.servers[1].running)

    def test_camera_is_wrapper_over_show_and_releases_capture(self):
        lv, _ = load_liveview_module()

        session = lv.camera(0, mode="mjpeg", show=False, wait=True)
        self.assertEqual(1, len(FakeCapture.instances))
        capture = FakeCapture.instances[0]
        self.assertFalse(capture.released)
        session.close()
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
