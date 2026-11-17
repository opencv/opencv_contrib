// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#ifndef OPENCV_TRITON_TEST_HIP_OWNERSHIP_HPP
#define OPENCV_TRITON_TEST_HIP_OWNERSHIP_HPP
// Single-visible-device checks. Cross-device allocation/stream rejection is
// intentionally a separate two-device probe, not a skipped single-card pass.
static void ownershipChecks(hipStream_t stream)
{
    const unsigned char fill = 0xA5;
    const cv::Mat host = randomImage({16, 16}, CV_8UC3, 81001);
    DevImg source, target;
    alloc(source, 16, 16, CV_8UC3, 2, 0);
    alloc(target, 16, 16, CV_8UC3, 3, fill);
    upload(source, host);
    const double identity[6] = {1, 0, 0, 0, 1, 0};
    auto call = [&](int operation, const tr::DeviceImage &src, const tr::DeviceImage &dst)
    {
        const auto handle = reinterpret_cast<std::uintptr_t>(stream);
        if (operation == 0)
            tr::resize(src, dst, {16, 16}, cv::INTER_NEAREST, 0, handle);
        else if (operation == 1)
            tr::cvtColor(src, dst, cv::COLOR_BGR2RGB, 0, handle);
        else
            tr::warpAffine(src, dst, identity, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE, cv::Scalar(), 0, handle);
    };
    for (int operation = 0; operation < 3; ++operation)
        for (int boundary = 0; boundary < 4; ++boundary)
        {
            tr::DeviceImage src = source.view(), dst = target.view();
            // Ordinary cv::Mat memory is host memory, never a valid device image.
            // Preserve its bytes too when it is supplied as the output.
            cv::Mat hostOutput(16, 16, CV_8UC3, cv::Scalar::all(fill));
            if (boundary == 0)
            {
                src.ptr = reinterpret_cast<std::uintptr_t>(host.data);
                src.step = host.step;
            }
            if (boundary == 1)
            {
                dst.ptr = reinterpret_cast<std::uintptr_t>(hostOutput.data);
                dst.step = hostOutput.step;
            }
            // A real device pointer with an invented row pitch must not escape
            // the actual allocation even though the packed row itself fits.
            if (boundary == 2)
                src.step = static_cast<std::size_t>(1) << 20;
            if (boundary == 3)
                dst.step = static_cast<std::size_t>(1) << 20;
            bool refused = false;
            const auto before = cv::triton::detail::runtimeStats();
            try
            {
                call(operation, src, dst);
            }
            catch (const cv::Exception &error)
            {
                refused = error.code == cv::Error::StsBadArg;
            }
            HIPOK(hipStreamSynchronize(stream));
            const auto after = cv::triton::detail::runtimeStats();
            const cv::Mat parent = downloadParent(target);
            bool untouched = true;
            for (std::size_t i = 0; i < parent.total(); ++i)
                untouched &= parent.data[i] == fill;
            const bool hostUntouched =
                cv::norm(hostOutput,
                         cv::Mat(hostOutput.size(), hostOutput.type(), cv::Scalar::all(fill)),
                         cv::NORM_INF) == 0;
            report(
                refused && untouched && hostUntouched && before.launches == after.launches &&
                    before.copies == after.copies,
                cv::format(
                    "device ownership op=%d boundary=%d rejects host pointer/overspan before writes or submission",
                    operation, boundary));
        }
}
#endif // OPENCV_TRITON_TEST_HIP_OWNERSHIP_HPP
