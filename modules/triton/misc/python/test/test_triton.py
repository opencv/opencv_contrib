#!/usr/bin/env python
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import cv2 as cv
import numpy as np
from unittest import mock

from tests_common import NewOpenCVTests, unittest

# Two independent things are checked here, and they are deliberately NOT in the
# same test:
#
#   1. The binding exists and is callable, returns the requested shape, dtype and
#      channel count, and raises on unsupported inputs. This holds on ANY
#      CUDA build with a usable Triton backend; no particular CPU build is required.
#      HIP's Python GpuMat refusal path is checked separately below.
#   2. The numbers match CPU cv::resize of the PINNED reference configuration
#      (OpenCV 5.0.0 generic resize.cpp path, SSE3 baseline, no IPP, no custom
#      HAL). cv::resize of an arbitrary build is that reference only if the
#      build is configured so.
#
# Binding checks run independently of resize's CPU reference configuration.
# Numerical checks use the reference appropriate to each operation.

_F32_ATOL = {cv.INTER_NEAREST: 0.0, cv.INTER_LINEAR: 4.948e-07, cv.INTER_CUBIC: 1.401e-06, cv.INTER_AREA: 8.419e-07}


def _is_reference_build(info=None, version=None, optimized=None):
    """Recognize the generic resize reference, including runtime optimization state."""
    info = cv.getBuildInformation() if info is None else info
    version = cv.__version__ if version is None else version
    optimized = cv.useOptimized() if optimized is None else optimized
    fields = {}
    for line in info.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key not in fields:
            fields[key] = value.strip()
    return (version == "5.0.0" and optimized is True
            and fields.get("Baseline") == "SSE SSE2 SSE3"
            and not fields.get("Dispatched code generation", "")
            and fields.get("Custom HAL") == "NO"
            and fields.get("Intel IPP", "NO") in ("NO", "")
            and fields.get("Algorithm Hint") == "ALGO_HINT_ACCURATE")


class _TritonBase(NewOpenCVTests):
    def setUp(self):
        super(_TritonBase, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")
        if cv.triton.backendName() == "hip-jit":
            self.skipTest("CUDA GpuMat numerical tests do not run on the HIP device-buffer backend")
        if not cv.triton.haveBackend():
            self.skipTest("No Triton backend available")


class triton_hip_binding_test(NewOpenCVTests):
    """HIP queries work, and CUDA-only Python adapters raise instead of aborting."""

    def setUp(self):
        super(triton_hip_binding_test, self).setUp()
        if not hasattr(cv, "triton") or cv.triton.backendName() != "hip-jit":
            self.skipTest("requires the HIP Triton module")

    def test_gpumat_adapters_raise_safely(self):
        gpu = cv.cuda_GpuMat()
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]], np.float64)
        for fn, args in ((cv.triton.resize, (gpu, (32, 24))),
                         (cv.triton.cvtColor, (gpu, cv.COLOR_BGR2GRAY)),
                         (cv.triton.warpAffine, (gpu, matrix, (32, 24)))):
            for kwargs in ({}, {"stream": None}):
                with self.assertRaises(cv.error) as raised:
                    fn(*args, **kwargs)
                self.assertEqual(raised.exception.code, -213)  # StsNotImplemented


class triton_binding_test(_TritonBase):
    """The binding works. No reference configuration required."""

    def test_module_is_exposed(self):
        self.assertTrue(hasattr(cv, "triton"))
        self.assertIsInstance(cv.triton.backendName(), str)
        self.assertNotEqual(cv.triton.backendName(), "")

    def test_returns_requested_shape_dtype_and_channels(self):
        rng = np.random.RandomState(0)
        for channels in (1, 3, 4):
            shape = (60, 80) if channels == 1 else (60, 80, channels)
            # Both supported depths: a float32 upload/download path
            # that silently returned uint8 (or the wrong channel count) would
            # not have been visible from a uint8-only check.
            for img in (rng.randint(0, 256, shape).astype(np.uint8),
                        (rng.rand(*shape) * 4 - 2).astype(np.float32)):
                for dsize in ((40, 30), (120, 90)):
                    got = cv.triton.resize(cv.cuda_GpuMat(img), dsize,
                                           interpolation=cv.INTER_LINEAR).download()
                    expect = ((dsize[1], dsize[0]) if channels == 1
                              else (dsize[1], dsize[0], channels))
                    self.assertEqual(got.shape, expect)
                    self.assertEqual(got.dtype, img.dtype)

    def test_rejected_inputs_raise(self):
        u8 = np.zeros((32, 32, 3), np.uint8)
        for interp in (cv.INTER_LANCZOS4,):
            with self.assertRaises(cv.error):
                cv.triton.resize(cv.cuda_GpuMat(u8), (16, 16), interpolation=interp)


class triton_numeric_test(_TritonBase):
    """The numbers match the pinned CPU reference. Skips on other builds."""

    def setUp(self):
        super(triton_numeric_test, self).setUp()
        if not _is_reference_build():
            self.skipTest("cv.resize of this build is not the reference configuration; "
                          "the C++ suite compares against the portable oracle instead")

    def _check(self, np_mat, dsize, interp):
        src = cv.cuda_GpuMat(np_mat)
        got = cv.triton.resize(src, dsize, interpolation=interp).download()
        ref = cv.resize(np_mat, dsize, interpolation=interp)
        self.assertEqual(got.shape, ref.shape)
        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        self.assertLessEqual(float(np.nanmax(d)), 0.0 if np_mat.dtype == np.uint8 else _F32_ATOL[interp])

    def test_resize_matches_cpu(self):
        rng = np.random.RandomState(0)
        for channels in (1, 3, 4):
            shape = (120, 160) if channels == 1 else (120, 160, channels)
            u8 = rng.randint(0, 256, shape).astype(np.uint8)
            f32 = (rng.rand(*shape) * 4 - 2).astype(np.float32)
            for dsize, modes in (((97, 71), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA)),
                                 ((320, 240), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA)),
                                 ((80, 60), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA)),
                                 ((160, 120), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA))):
                for interp in modes:
                    self._check(u8, dsize, interp)
                    self._check(f32, dsize, interp)


# cvtColor compares with this build's own CPU entry, not the resize portable
# oracle. Integer conversions and float copies compare exactly; finite float
# arithmetic uses the documented per-pixel input-channel scale bound.
def _cvt_codes(names, scn, dcn):
    """Group related conversion names without duplicating their channel metadata."""
    return [(getattr(cv, "COLOR_" + name), scn, dcn) for name in names.split()]


_CVT_F32_COPY = (
    _cvt_codes("BGR2BGRA BGR2RGBA", 3, 4) +
    _cvt_codes("BGRA2BGR RGBA2BGR", 4, 3) +
    _cvt_codes("BGR2RGB", 3, 3) + _cvt_codes("BGRA2RGBA", 4, 4) +
    _cvt_codes("GRAY2BGR", 1, 3) + _cvt_codes("GRAY2BGRA", 1, 4))
_CVT_F32_ARITHMETIC = (
    _cvt_codes("BGR2GRAY RGB2GRAY", 3, 1) +
    _cvt_codes("BGRA2GRAY RGBA2GRAY", 4, 1) +
    _cvt_codes("BGR2YCrCb RGB2YCrCb YCrCb2BGR YCrCb2RGB "
               "BGR2YUV RGB2YUV YUV2BGR YUV2RGB", 3, 3))
_CVT_BAYER = (_cvt_codes("BayerBG2BGR BayerGB2BGR BayerRG2BGR BayerGR2BGR", 1, 3) +
              _cvt_codes("BayerBG2BGRA", 1, 4) + _cvt_codes("BayerBG2GRAY", 1, 1))
_CVT_422_COLOR = _cvt_codes("YUV2BGR_UYVY YUV2RGB_UYVY YUV2BGR_YUY2 "
                             "YUV2RGB_YUY2 YUV2BGR_YVYU YUV2RGB_YVYU", 2, 3)
_CVT_422_GRAY = _cvt_codes("YUV2GRAY_UYVY YUV2GRAY_YUY2", 2, 1)
_CVT_420 = sum((_cvt_codes("YUV2BGR_" + layout + " YUV2RGB_" + layout, 1, 3) +
                 _cvt_codes("YUV2BGRA_" + layout + " YUV2RGBA_" + layout, 1, 4)
                 for layout in ("NV12", "NV21")), [])
_CVT_PACKED = []
for _packed in ("BGR565", "BGR555"):
    for _color, _cn in (("BGR", 3), ("RGB", 3), ("BGRA", 4), ("RGBA", 4), ("GRAY", 1)):
        _CVT_PACKED += _cvt_codes(_color + "2" + _packed, _cn, 2)
        _CVT_PACKED += _cvt_codes(_packed + "2" + _color, 2, _cn)
_CVT_CIE = _cvt_codes("BGR2XYZ RGB2XYZ XYZ2BGR XYZ2RGB "
                       "BGR2Lab RGB2Lab Lab2BGR Lab2RGB "
                       "BGR2Luv RGB2Luv Luv2BGR Luv2RGB "
                       "LBGR2Lab LRGB2Lab Lab2LBGR Lab2LRGB Luv2LBGR Luv2LRGB", 3, 3)
_CVT_CODES = (_CVT_F32_COPY + _CVT_F32_ARITHMETIC +
              _cvt_codes("BGR2HSV RGB2HSV BGR2HSV_FULL RGB2HSV_FULL", 3, 3) +
              _CVT_422_COLOR + _CVT_422_GRAY + _CVT_420 +
              _CVT_PACKED + _CVT_BAYER + _CVT_CIE)
# 16U supports Bayer, gray expansion, and the gray/YCrCb/YUV integer families;
# the six 3/4-channel bit-copy codes remain unsupported at this depth.
_CVT_U16_CODES = _CVT_BAYER + _CVT_F32_COPY[-2:] + _CVT_F32_ARITHMETIC


def _cvt_input(rng, dtype, h, w, scn):
    shape = (h, w) if scn == 1 else (h, w, scn)
    return rng.randint(0, 256 if dtype == np.uint8 else 65536, shape).astype(dtype)


def _cvt_sizes(code):
    if code in {case[0] for case in _CVT_420}:
        # Logical output H/W: source is 3H/2 by W.
        return ((2, 2), (6, 14), (62, 98))
    if code in {case[0] for case in _CVT_422_COLOR}:
        return ((1, 2), (7, 14), (61, 98))
    return ((1, 1), (7, 13), (61, 97))


def _cvt_f32_allow(src, ref):
    scale = np.max(np.abs(src.astype(np.float64)), axis=2) if src.ndim == 3 else np.abs(src.astype(np.float64))
    if ref.ndim == 3:
        scale = scale[:, :, None]
    return (8.0 * 2.0 ** -24) * np.maximum(np.maximum(1.0, np.abs(ref.astype(np.float64))), scale)



class triton_reference_fixture_test(NewOpenCVTests):
    """CPU-only checks for the shape/depth fixtures used by the CUDA tests."""

    def test_resize_reference_build_detection(self):
        fields = {"Baseline": "SSE SSE2 SSE3", "Custom HAL": "NO",
                  "Algorithm Hint": "ALGO_HINT_ACCURATE"}

        def build_info(changes=None):
            values = dict(fields)
            values.update(changes or {})
            return "\n".join("%s: %s" % item for item in values.items())

        self.assertTrue(_is_reference_build(build_info(), "5.0.0", True))
        self.assertTrue(_is_reference_build(build_info({"Dispatched code generation": ""}),
                                            "5.0.0", True))
        for changes in ({"Baseline": "SSE SSE2 SSE3 SSE4_1"},
                        {"Dispatched code generation": "SSE4_1 AVX2"},
                        {"Intel IPP": "2022.0"}, {"Custom HAL": "custom"},
                        {"Algorithm Hint": "ALGO_HINT_APPROX"}):
            self.assertFalse(_is_reference_build(build_info(changes), "5.0.0", True),
                             msg=str(changes))
        for version in ("4.13.0", "5.0.1", "5.1.0", "5.0.0-dev"):
            self.assertFalse(_is_reference_build(build_info(), version, True))
        self.assertFalse(_is_reference_build(build_info(), "5.0.0", False))
        self.assertFalse(_is_reference_build("", "5.0.0", True))

    def test_warpaffine_numeric_does_not_use_resize_build_gate(self):
        def reject_resize_gate(*args, **kwargs):
            raise AssertionError("warpAffine must use the linked reference")

        with mock.patch.object(_TritonBase, "setUp") as setup:
            with mock.patch.dict(globals(), {"_is_reference_build": reject_resize_gate}):
                case = triton_warpaffine_numeric_test("test_matches_cpu")
                case.setUp()
            setup.assert_called_once_with()

    def test_cvtcolor_fixture_coverage_and_shapes(self):
        rng = np.random.RandomState(29)
        for cases, dtype, expected in ((_CVT_CODES, np.uint8, 84),
                                        (_CVT_U16_CODES, np.uint16, 20),
                                        (_CVT_F32_COPY + _CVT_F32_ARITHMETIC, np.float32, 20)):
            self.assertEqual(expected, len(cases))
            self.assertEqual(expected, len({code for code, _, _ in cases}))
            for code, scn, dcn in cases:
                for h, w in _cvt_sizes(code):
                    source_h = h * 3 // 2 if code in {c[0] for c in _CVT_420} else h
                    if dtype == np.float32:
                        shape = (h, w) if scn == 1 else (h, w, scn)
                        img = rng.uniform(-2, 2, shape).astype(dtype)
                    else:
                        img = _cvt_input(rng, dtype, source_h, w, scn)
                    ref = cv.cvtColor(img, code)
                    self.assertEqual((h, w, dcn) if dcn > 1 else (h, w), ref.shape)
                    self.assertEqual(dtype, ref.dtype)

    def test_cpu_bayer_single_pixel_axes(self):
        for dtype in (np.uint8, np.uint16):
            for code, scn, dcn in _CVT_BAYER:
                for h, w in ((1, 1), (1, 7), (7, 1)):
                    ref = cv.cvtColor(np.zeros((h, w), dtype), code)
                    self.assertEqual((h, w, dcn) if dcn > 1 else (h, w), ref.shape)
                    self.assertEqual(dtype, ref.dtype)

    def test_cpu_transparent_reference_preserves_background(self):
        matrix = np.array([[1., 0., 100.], [0., 1., 100.]], np.float64)
        for dtype in (np.uint8, np.uint16, np.float32):
            for cn in (1, 3, 4):
                shape = (7, 9) if cn == 1 else (7, 9, cn)
                img, initial = np.zeros(shape, dtype), np.full(shape, 17, dtype)
                for direction in (0, cv.WARP_INVERSE_MAP):
                    for interpolation in (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA):
                        ref = _wa_cpu_reference(img, matrix, (9, 7), interpolation | direction,
                                                cv.BORDER_TRANSPARENT, initial=initial)
                        np.testing.assert_array_equal(ref, initial)



def _cvt_separate4_model(src, code):
    # NumPy ufuncs materialize binary32 results separately; no multiply-add fusion.
    gray = {cv.COLOR_BGR2GRAY, cv.COLOR_RGB2GRAY, cv.COLOR_BGRA2GRAY, cv.COLOR_RGBA2GRAY}
    rgb = {cv.COLOR_RGB2GRAY, cv.COLOR_RGBA2GRAY, cv.COLOR_RGB2YCrCb,
           cv.COLOR_RGB2YUV, cv.COLOR_YCrCb2RGB, cv.COLOR_YUV2RGB}
    inverse = {cv.COLOR_YCrCb2BGR, cv.COLOR_YCrCb2RGB, cv.COLOR_YUV2BGR, cv.COLOR_YUV2RGB}
    uv = code in {cv.COLOR_BGR2YUV, cv.COLOR_RGB2YUV, cv.COLOR_YUV2BGR, cv.COLOR_YUV2RGB}
    blue = 2 if code in rgb else 0
    f = np.float32
    if code in inverse:
        y = src[..., 0]
        cr = src[..., 2 if uv else 1] - f(.5)
        cb = src[..., 1 if uv else 2] - f(.5)
        b = y + cb * f(2.032 if uv else 1.773)
        g = (y + cb * f(-.395 if uv else -.344)) + cr * f(-.581 if uv else -.714)
        r = y + cr * f(1.140 if uv else 1.403)
        return np.stack((b,g,r) if blue == 0 else (r,g,b), axis=2)
    a = src[..., 0] * f(.114 if blue == 0 else .299)
    b = src[..., 1] * f(.587)
    c = src[..., 2] * f(.299 if blue == 0 else .114)
    y = (a + b) + c
    if code in gray:
        return y
    end = (src.shape[1] // 4) * 4
    y[:, :end] = (a + (b + c))[:, :end]
    cr = (src[..., blue ^ 2] - y) * f(.877 if uv else .713) + f(.5)
    cb = (src[..., blue] - y) * f(.492 if uv else .564) + f(.5)
    return np.stack((y,cb,cr) if uv else (y,cr,cb), axis=2)


def _cvt_cpu_separate4():
    def state():
        return (cv.useOptimized(), cv.ipp.useIPP(), cv.ipp.useIPP_NotExact())
    before = state()
    for code, scn, _ in _CVT_F32_ARITHMETIC:
        for width in (1,3,4,5,7,8,9,15,16,17,31,33,65):
            for roi in (False, True):
                rng = np.random.RandomState(193 + code * 31 + width)
                parent = rng.uniform(-2, 2, (5, width+7, scn)).astype(np.float32)
                src = parent[1:4, 2:width+2]
                if not roi:
                    src = src.copy()
                src[:, 1::3] *= np.float32(32767.5)
                ref = cv.cvtColor(src, code)
                model = _cvt_separate4_model(src, code)
                if ref.shape != model.shape or ref.dtype != model.dtype or not np.isfinite(ref).all() or not np.array_equal(ref, model):
                    return False
    return before == state()


class triton_cvtcolor_test(_TritonBase):
    """CUDA cvtColor bindings against the linked CPU reference, per supported depth."""

    def _check_integer_matrix(self, cases, dtype):
        rng = np.random.RandomState(3)
        for code, scn, dcn in cases:
            for h, w in _cvt_sizes(code):
                with self.subTest(code=code, dtype=dtype, size=(h, w)):
                    source_h = h * 3 // 2 if code in {c[0] for c in _CVT_420} else h
                    img = _cvt_input(rng, dtype, source_h, w, scn)
                    got = cv.triton.cvtColor(cv.cuda_GpuMat(img), code).download()
                    ref = cv.cvtColor(img, code)
                    self.assertEqual((h, w, dcn) if dcn > 1 else (h, w), got.shape)
                    self.assertEqual(got.dtype, dtype)
                    np.testing.assert_array_equal(got, ref, err_msg="code " + str(code))

    def test_all_uint8_codes_exactly(self):
        self.assertEqual(84, len({code for code, _, _ in _CVT_CODES}))
        self._check_integer_matrix(_CVT_CODES, np.uint8)

    def test_all_uint16_codes_exactly(self):
        self.assertEqual(20, len({code for code, _, _ in _CVT_U16_CODES}))
        self._check_integer_matrix(_CVT_U16_CODES, np.uint16)

    def test_bayer_single_pixel_axes(self):
        rng = np.random.RandomState(31)
        for dtype in (np.uint8, np.uint16):
            for code, scn, dcn in _CVT_BAYER:
                for h, w in ((1, 1), (1, 7), (7, 1)):
                    img = _cvt_input(rng, dtype, h, w, scn)
                    got = cv.triton.cvtColor(cv.cuda_GpuMat(img), code).download()
                    np.testing.assert_array_equal(got, cv.cvtColor(img, code))
                    self.assertEqual((h, w, dcn) if dcn > 1 else (h, w), got.shape)

    def test_float_bound_does_not_borrow_another_pixels_scale(self):
        src = np.array([[[1.0, 1.0, 1.0], [65535.0, 65535.0, 65535.0]]], np.float32)
        ref = np.array([[1.0, 65535.0]], np.float32)
        wrong = ref.astype(np.float64)
        wrong[0, 0] += 1e-3
        self.assertGreater(abs(wrong[0, 0] - ref[0, 0]), _cvt_f32_allow(src, ref)[0, 0])
        # A whole-image bound would incorrectly accept this small-pixel error.
        self.assertLess(abs(wrong[0, 0] - ref[0, 0]), (8.0 * 2.0 ** -24) * 65535.0)

    def test_all_float32_codes_with_per_pixel_scale(self):
        rng = np.random.RandomState(19)
        self.assertEqual(20, len({code for code, _, _ in _CVT_F32_COPY + _CVT_F32_ARITHMETIC}))
        for cases, exact in ((_CVT_F32_COPY, True), (_CVT_F32_ARITHMETIC, False)):
            for code, scn, dcn in cases:
                for w in (1, 7, 16, 17, 257):
                    shape = (3, w) if scn == 1 else (3, w, scn)
                    img = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
                    img[:, ::3] *= np.float32(32767.5)
                    stream = cv.cuda_Stream()
                    src_gpu = cv.cuda_GpuMat(img)
                    got_gpu = cv.triton.cvtColor(src_gpu, code, stream=stream)
                    stream.waitForCompletion()
                    got = got_gpu.download()
                    ref = cv.cvtColor(img, code)
                    self.assertEqual(ref.shape, got.shape)
                    self.assertEqual(np.float32, got.dtype)
                    self.assertTrue(np.isfinite(got).all())
                    if exact:
                        np.testing.assert_array_equal(got, ref)
                    else:
                        self.assertTrue(np.all(np.abs(got.astype(np.float64) - ref) <= _cvt_f32_allow(img, ref)),
                                        msg="float32 code " + str(code))

    def test_float_mixed_scale_absolute_cap(self):
        previous = cv.useOptimized()
        try:
            for optimized in (False, True):
                cv.setUseOptimized(optimized)
                if not _cvt_cpu_separate4():
                    self.skipTest("unidentified linked CPU float profile; absolute cap not qualified")
                for code, scn, _ in _CVT_F32_ARITHMETIC:
                    for width in (1,3,4,5,7,8,9,15,16,17,31,33,65):
                        rng = np.random.RandomState(1921 + width)
                        img = rng.uniform(-2, 2, (5,width,scn)).astype(np.float32)
                        img[:, 1::3] *= np.float32(65535)
                        got = cv.triton.cvtColor(cv.cuda_GpuMat(img), code).download()
                        ref = cv.cvtColor(img, code)
                        error = np.abs(got.astype(np.float64) - ref)
                        self.assertTrue(np.isfinite(error).all())
                        self.assertTrue(np.all(error <= _cvt_f32_allow(img, ref)))
                        self.assertLessEqual(float(error.max()), 1e-4)
        finally:
            cv.setUseOptimized(previous)

    def test_malformed_video_and_legal_odd_extract(self):
        for code in (cv.COLOR_YUV2BGR_UYVY, cv.COLOR_YUV2RGB_UYVY,
                     cv.COLOR_YUV2BGR_YUY2, cv.COLOR_YUV2RGB_YUY2,
                     cv.COLOR_YUV2BGR_YVYU, cv.COLOR_YUV2RGB_YVYU):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(np.zeros((1, 3, 2), np.uint8)), code)
        for code, _, _ in _CVT_420:
            for shape in ((24, 31), (23, 32)):
                with self.assertRaises(cv.error):
                    cv.triton.cvtColor(cv.cuda_GpuMat(np.zeros(shape, np.uint8)), code)
        img = np.arange(6, dtype=np.uint8).reshape(1, 3, 2)
        for code, _, _ in _CVT_422_GRAY:
            got = cv.triton.cvtColor(cv.cuda_GpuMat(img), code).download()
            np.testing.assert_array_equal(got, cv.cvtColor(img, code))

    def test_rejected_codes_raise(self):
        img = np.zeros((32, 32, 3), np.uint8)
        # Reverse HSV and HLS are unsupported conversion families, even at uint8.
        for code in (cv.COLOR_HSV2BGR, cv.COLOR_BGR2HLS):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(img), code)
        # CV_32F is served only for supported conversion families: the bit-copy codes and
        # the three arithmetic families (colour->gray, YCrCb, YUV). Everything else at 32F is
        # refused. Supported arithmetic uses the documented per-pixel floating-point bound.
        f32 = np.random.RandomState(5).uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
        for code in (cv.COLOR_BGR2XYZ, cv.COLOR_BGR2HSV, cv.COLOR_BGR2Lab):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(f32), code)
        # ...and the arithmetic ones really are served, within the documented bound rather than
        # exactly: the CPU has several 32F answers (fused body W>=16, fused tail W<16, unfused
        # build), so equality is not the criterion here.
        got = cv.triton.cvtColor(cv.cuda_GpuMat(f32), cv.COLOR_BGR2GRAY).download()
        want = cv.cvtColor(f32, cv.COLOR_BGR2GRAY)
        self.assertEqual(got.dtype, np.float32)
        k = 8.0 * 2.0 ** -24
        allow = k * np.maximum(np.maximum(1.0, np.abs(want.astype(np.float64))),
                               np.abs(f32.astype(np.float64)).max(axis=2))
        self.assertLessEqual(float((np.abs(got.astype(np.float64) -
                                           want.astype(np.float64)) / allow).max()), 1.0)
        # cvtColor's YUV extraction wrapper is 8U-only even though extractChannel
        # alone accepts 32F. Preserve the CPU entry's refusal rather than extending it.
        f32_extract = np.ones((3, 7, 2), np.float32)
        for code in (cv.COLOR_YUV2GRAY_UYVY, cv.COLOR_YUV2GRAY_YUY2):
            with self.assertRaises(cv.error):
                cv.cvtColor(f32_extract, code)
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(f32_extract), code)
        # A source whose channel count does not match the code.
        with self.assertRaises(cv.error):
            cv.triton.cvtColor(cv.cuda_GpuMat(np.zeros((32, 32), np.uint8)),
                               cv.COLOR_BGR2GRAY)


# warpAffine: bindings are checked on any usable build;
# numeric tests use the selected linked CPU reference. NN/LINEAR/AREA use
# the padded vector path; CUBIC uses the natural-width call. Fused arithmetic
# is calibrated from CPU outputs, not hardware feature flags. The unchanged
# per-element criteria are uint8 exact, uint16 <= 1 level, and float32
# min(5e-5 * max(abs(ref), 1.0), 1e-4).
_WA_F32_REL = 5.0e-5
_WA_F32_FLOOR = 1.0

_WA_GEOMS = [
    ("identity",  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ("subpixel",  [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
    ("rot_scale", [[0.9, -0.43589, 12.0], [0.43589, 0.9, -7.0]]),
    ("shear",     [[1.0, 0.35, 0.0], [0.0, 1.0, 0.0]]),
    ("negative",  [[-1.0, 0.0, 60.0], [0.0, -1.0, 40.0]]),
]


def _wa_within(got, ref):
    """The per-element error bound. Returns (elements outside, max normalised error)."""
    if ref.dtype in (np.uint8, np.uint16):
        d = np.abs(got.astype(np.int64) - ref.astype(np.int64))
        return int((d > (0 if ref.dtype == np.uint8 else 1)).sum()), float(d.max(initial=0))
    a, b = got.astype(np.float64), ref.astype(np.float64)
    d = np.abs(a - b)
    allowed = np.minimum(_WA_F32_REL * np.maximum(np.abs(b), _WA_F32_FLOOR), 1e-4)
    return int((~(d <= allowed)).sum()), float(np.max(d / allowed, initial=0.0))



def _wa_cpu_reference(img, matrix, dsize, flags, border, border_value=0, initial=None):
    """Selected CPU path, preserving the same initialized destination region."""
    interp = flags & cv.INTER_MAX
    width = dsize[0] if interp == cv.INTER_CUBIC else ((dsize[0] + 255) // 128) * 128
    shape = (dsize[1], width) + img.shape[2:]
    wide = np.zeros(shape, img.dtype)
    if initial is not None:
        wide[:, :dsize[0]] = initial
    cv.warpAffine(img, matrix, (width, dsize[1]), dst=wide, flags=flags,
                  borderMode=border, borderValue=border_value)
    return wide[:, :dsize[0]].copy()


class triton_warpaffine_binding_test(_TritonBase):
    """The binding works -- shape, dtype, channels, matrix types, output reuse, refusals."""

    def test_returns_requested_shape_dtype_and_channels(self):
        rng = np.random.RandomState(0)
        M = np.array(_WA_GEOMS[1][1], np.float64)
        for channels in (1, 3, 4):
            shape = (37, 53) if channels == 1 else (37, 53, channels)
            for img in (rng.randint(0, 256, shape).astype(np.uint8),
                        rng.randint(0, 65536, shape).astype(np.uint16),
                        (rng.rand(*shape) * 4 - 2).astype(np.float32)):
                for dsize in ((53, 37), (80, 60)):
                    got = cv.triton.warpAffine(cv.cuda_GpuMat(img), M, dsize).download()
                    expect = ((dsize[1], dsize[0]) if channels == 1
                              else (dsize[1], dsize[0], channels))
                    self.assertEqual(got.shape, expect)
                    self.assertEqual(got.dtype, img.dtype)

    def test_matrix_accepts_float32_and_float64(self):
        # Both matrix depths are accepted and converted to double before sampling.
        # This fixture compares their outputs; it does not promise that arbitrary
        # coefficients rounded to float32 preserve float64 sampling boundaries.
        rng = np.random.RandomState(1)
        img = rng.randint(0, 256, (37, 53, 3)).astype(np.uint8)
        g = cv.cuda_GpuMat(img)
        ref = None
        for dt in (np.float32, np.float64):
            M = np.array(_WA_GEOMS[2][1], dt)
            got = cv.triton.warpAffine(g, M, (53, 37)).download()
            if ref is None:
                ref = got
            else:
                self.assertEqual(int(np.abs(got.astype(np.int16) -
                                            ref.astype(np.int16)).max()), 0)

    def test_border_value_is_a_scalar_per_channel(self):
        # A C3 border of (10, 20, 30) fills three channels; a single 77 fills channel 0 only,
        # exactly as cv::warpAffine expands Scalar(v) to (v, 0, 0, 0).
        img = np.zeros((9, 7, 3), np.uint8)
        M = np.array([[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0]])    # everything outside
        for bv, want in (((10, 20, 30), (10, 20, 30)), (77, (77, 0, 0))):
            got = cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (7, 9),
                                       borderMode=cv.BORDER_CONSTANT, borderValue=bv).download()
            self.assertEqual(tuple(int(v) for v in got[0, 0]), want)

    def test_float32_border_keeps_fractional_precision(self):
        img = np.zeros((9, 7), np.float32)
        M = np.array([[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0]])
        for bv in (0.5, -1.75, 3.25):
            got = cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (7, 9),
                                       borderMode=cv.BORDER_CONSTANT,
                                       borderValue=bv).download()
            self.assertAlmostEqual(float(got[0, 0]), bv, places=6)

    def test_destination_is_reused_when_supplied(self):
        rng = np.random.RandomState(2)
        img = rng.randint(0, 256, (37, 53, 3)).astype(np.uint8)
        M = np.array(_WA_GEOMS[2][1], np.float64)
        dst = cv.cuda_GpuMat((37, 53), cv.CV_8UC3)
        cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (53, 37), dst=dst)
        first = dst.download()
        cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (53, 37), dst=dst)
        self.assertEqual(int(np.abs(first.astype(np.int16) -
                                    dst.download().astype(np.int16)).max()), 0)

    def test_area_alias_all_depths_channels_and_directions(self):
        rng = np.random.RandomState(23)
        matrix = np.array(_WA_GEOMS[2][1], np.float64)
        for dtype in (np.uint8, np.uint16, np.float32):
            for cn in (1, 3, 4):
                shape = (19, 27) if cn == 1 else (19, 27, cn)
                img = (rng.uniform(-2, 2, shape).astype(dtype) if dtype == np.float32
                       else _cvt_input(rng, dtype, 19, 27, cn))
                gpu = cv.cuda_GpuMat(img)
                for direction in (0, cv.WARP_INVERSE_MAP):
                    area = cv.triton.warpAffine(gpu, matrix, (27, 19),
                                               flags=cv.INTER_AREA | direction,
                                               borderMode=cv.BORDER_REFLECT_101).download()
                    linear = cv.triton.warpAffine(gpu, matrix, (27, 19),
                                                 flags=cv.INTER_LINEAR | direction,
                                                 borderMode=cv.BORDER_REFLECT_101).download()
                    np.testing.assert_array_equal(area, linear)

    def test_cubic_transparent_is_refused_without_changing_destination(self):
        matrix = np.eye(3, dtype=np.float64)[:2]
        for dtype in (np.uint8, np.uint16, np.float32):
            for cn in (1, 3, 4):
                shape = (7, 9) if cn == 1 else (7, 9, cn)
                img = np.zeros(shape, dtype)
                initial = np.full(shape, 17, dtype)
                dst = cv.cuda_GpuMat(initial)
                with self.assertRaises(cv.error):
                    cv.triton.warpAffine(cv.cuda_GpuMat(img), matrix, (9, 7), dst=dst,
                                         flags=cv.INTER_CUBIC, borderMode=cv.BORDER_TRANSPARENT)
                np.testing.assert_array_equal(dst.download(), initial)

    def test_rejected_inputs_raise(self):
        u8 = np.zeros((32, 32, 3), np.uint8)
        M = np.eye(3, dtype=np.float64)[:2]
        # AREA follows the CPU rewrite to LINEAR; LANCZOS4 is unsupported.
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(u8), M, (32, 32), flags=cv.INTER_LANCZOS4)
        # AREA is served AND equals LINEAR, checked rather than merely not raising.
        area = cv.triton.warpAffine(cv.cuda_GpuMat(u8), M, (32, 32),
                                    flags=cv.INTER_AREA).download()
        lin = cv.triton.warpAffine(cv.cuda_GpuMat(u8), M, (32, 32),
                                   flags=cv.INTER_LINEAR).download()
        self.assertEqual(int(np.abs(area.astype(np.int16) - lin.astype(np.int16)).max()), 0)
        # CV_16U is served; CV_16S is not.
        got = cv.triton.warpAffine(cv.cuda_GpuMat(np.zeros((32, 32, 3), np.uint16)),
                                   M, (32, 32)).download()
        self.assertEqual(got.dtype, np.uint16)
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(np.zeros((32, 32, 3), np.int16)), M, (32, 32))
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(np.zeros((32, 32, 2), np.uint8)), M, (32, 32))
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(u8), np.zeros((3, 3), np.float64), (32, 32))


class triton_warpaffine_numeric_test(_TritonBase):
    """Compare with the linked CPU paths selected by warpAffine."""

    def _check(self, img, M, dsize, flags, border, bv=0):
        g = cv.cuda_GpuMat(img)
        got = cv.triton.warpAffine(g, M, dsize, flags=flags, borderMode=border,
                                   borderValue=bv).download()
        ref = _wa_cpu_reference(img, M, dsize, flags, border, bv)
        self.assertEqual(got.shape, ref.shape)
        nbad, worst = _wa_within(got, ref)
        self.assertEqual(nbad, 0, msg="%d elements outside the error bound, worst %.3f"
                                     % (nbad, worst))

    def test_matches_cpu(self):
        rng = np.random.RandomState(4)
        for name, m in _WA_GEOMS:
            M = np.array(m, np.float64)
            for channels in (1, 3, 4):
                shape = (37, 53) if channels == 1 else (37, 53, channels)
                for img in (rng.randint(0, 256, shape).astype(np.uint8),
                            rng.randint(0, 65536, shape).astype(np.uint16),
                            (rng.rand(*shape) * 4 - 2).astype(np.float32)):
                    for flags, border in ((cv.INTER_LINEAR, cv.BORDER_REPLICATE),
                                          (cv.INTER_NEAREST, cv.BORDER_CONSTANT),
                                          (cv.INTER_LINEAR, cv.BORDER_REFLECT_101),
                                          (cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                                           cv.BORDER_CONSTANT),
                                          (cv.INTER_CUBIC, cv.BORDER_CONSTANT),
                                          (cv.INTER_CUBIC, cv.BORDER_REFLECT_101)):
                        self._check(img, M, (53, 37), flags, border, 7)

    def test_transparent_initialized_destination(self):
        rng = np.random.RandomState(37)
        # Partial, rotated, and wholly outside maps exercise the old-dst blend,
        # sampling locality and the no-write rule independently.
        matrices = ([[1., 0., .5], [0., 1., -.5]], _WA_GEOMS[2][1],
                    [[1., 0., 100.], [0., 1., 100.]])
        for dtype in (np.uint8, np.uint16, np.float32):
            for cn in (1, 3, 4):
                shape = (19, 27) if cn == 1 else (19, 27, cn)
                img = (rng.uniform(-2, 2, shape).astype(dtype) if dtype == np.float32
                       else _cvt_input(rng, dtype, 19, 27, cn))
                initial = np.full(shape, 17, dtype)
                for geometry, values in enumerate(matrices):
                    matrix = np.array(values, np.float64)
                    for direction in (0, cv.WARP_INVERSE_MAP):
                        for interpolation in (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA):
                            flags = interpolation | direction
                            dst = cv.cuda_GpuMat(initial)
                            got = cv.triton.warpAffine(cv.cuda_GpuMat(img), matrix, (27, 19),
                                                       dst=dst, flags=flags,
                                                       borderMode=cv.BORDER_TRANSPARENT).download()
                            ref = _wa_cpu_reference(img, matrix, (27, 19), flags,
                                                    cv.BORDER_TRANSPARENT, initial=initial)
                            self.assertEqual(img.shape, got.shape)
                            self.assertEqual(img.dtype, got.dtype)
                            nbad, worst = _wa_within(got, ref)
                            self.assertEqual(0, nbad, msg="transparent %s C%d flags=%d geometry=%d worst=%g" %
                                             (dtype, cn, flags, geometry, worst))
                            if geometry == 2:
                                np.testing.assert_array_equal(got, initial)

    def test_matches_cpu_larger_destination(self):
        rng = np.random.RandomState(5)
        img = rng.randint(0, 256, (37, 53, 3)).astype(np.uint8)
        for name, m in _WA_GEOMS:
            self._check(img, np.array(m, np.float64), (120, 90),
                        cv.INTER_LINEAR, cv.BORDER_CONSTANT, 3)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
