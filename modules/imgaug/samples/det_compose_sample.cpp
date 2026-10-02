#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgaug.hpp>
#include <vector>

using namespace cv;


static void drawBoundingBoxes(Mat& img, std::vector<Rect>& bboxes){
    for(cv::Rect bbox: bboxes){
        cv::Point tl {bbox.x, bbox.y};
        cv::Point br {bbox.x + bbox.width, bbox.y + bbox.height};
        cv::rectangle(img, tl, br, cv::Scalar(0, 255, 0), 2);
    }
}


int main(){
    Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
    Mat dst;

    std::vector<Rect> bboxes{
            Rect{112, 40, 249, 343},
            Rect{61, 273, 113, 228}
    };

    std::vector<int> labels {1, 2};

    Mat ori_src;
    src.copyTo(ori_src);
    drawBoundingBoxes(ori_src, bboxes);

    imgaug::det::RandomRotation randomRotation(Vec2d(-30, 30));
    imgaug::det::RandomFlip randomFlip(1);
    imgaug::det::Resize resize(Size(224, 224));

    std::vector<Ptr<imgaug::det::Transform> > transforms {&randomRotation, &randomFlip, &resize};
    imgaug::det::Compose aug(transforms);

    aug.call(src, dst, bboxes, labels);

    drawBoundingBoxes(dst, bboxes);

    imshow("src", ori_src);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}