#include "opencv2/hed.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"

namespace cv {
namespace hed {

class HEDDetectorImpl : public HEDDetector {
public:
    dnn::Net net;

    HEDDetectorImpl(const std::string& modelPath, const std::string& protoPath) {
        net = dnn::readNet(modelPath, protoPath);
    }

    cv::Mat detectEdges(cv::InputArray _image) override {
        cv::Mat image = _image.getMat();
        cv::Mat blob = dnn::blobFromImage(
            image, 1.0,
            cv::Size(image.cols, image.rows),
            cv::Scalar(104.00698793, 116.66876762, 122.67891434),
            false, false
        );
        net.setInput(blob);
        cv::Mat result = net.forward("sigmoid-fuse");

        // result shape is [1,1,H,W] — squeeze to [H,W]
        cv::Mat edge(result.size[2], result.size[3], CV_32F, result.ptr<float>());
        cv::Mat output;
        edge.copyTo(output);
        return output;
    }
};

cv::Ptr<HEDDetector> HEDDetector::create(
    const std::string& modelPath,
    const std::string& protoPath)
{
    return cv::makePtr<HEDDetectorImpl>(modelPath, protoPath);
}

} // hed
} // cv