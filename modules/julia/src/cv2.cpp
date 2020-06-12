// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2020 by Archit Rungta


#include "jlcxx/array.hpp"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"
#include "jlcxx/tuple.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace std;
using namespace jlcxx;

#include "jlcv2.hpp"

namespace jlcxx
{
template <typename T>
struct IsSmartPointerType<cv::Ptr<T>> : std::true_type
{
};
template <typename T>
struct ConstructorPointerType<cv::Ptr<T>>
{
    typedef T *type;
};

template <typename T, int Val>
struct BuildParameterList<cv::Vec<T, Val>>
{
    typedef ParameterList<T, std::integral_constant<int, Val>> type;
};
template <>
struct SuperType<cv::Feature2D>
{
    typedef cv::Algorithm type;
};
template <>
struct SuperType<cv::SimpleBlobDetector>
{
    typedef cv::Feature2D type;
};
}

// Needed to prevent documentation warning

namespace cv { namespace julia {
void initJulia(int, char**) {}
}}



JLCXX_MODULE cv_wrap(jlcxx::Module &mod)
{
    mod.map_type<RotatedRect>("RotatedRect");
    mod.map_type<TermCriteria>("TermCriteria");
    mod.map_type<Range>("Range");

    mod.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("CxxVec")
        .apply<Vec4f, Vec6f, Vec3d, Vec2d>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            typedef typename get_template_type_vec<WrappedT>::type T;
            wrapped.template constructor<const T *>();
        });

    mod.add_type<Mat>("CxxMat").constructor<int, const int *, int, void *, const size_t *>();

    mod.method("jlopencv_core_get_sizet", []() { return sizeof(size_t); });
    jlcxx::add_smart_pointer<cv::Ptr>(mod, "cv_Ptr");
    mod.method("jlopencv_core_Mat_mutable_data", [](Mat m) {
        return make_tuple(m.data, m.type(), m.channels(), m.size[1], m.size[0], m.step[1], m.step[0]);
    });

    mod.add_type<Parametric<TypeVar<1>>>("CxxScalar")
        .apply<Scalar_<int>, Scalar_<float>, Scalar_<double>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            typedef typename get_template_type<WrappedT>::type T;
            wrapped.template constructor<T, T, T, T>();
        });

    using namespace cv;
    mod.add_type<cv::Algorithm>("Algorithm");
    mod.add_type<cv::KeyPoint>("KeyPoint");
    mod.add_type<cv::Feature2D>("Feature2D", jlcxx::julia_base_type<cv::Algorithm>());
    mod.add_type<cv::SimpleBlobDetector>("SimpleBlobDetector", jlcxx::julia_base_type<cv::Feature2D>());
    mod.add_type<cv::SimpleBlobDetector::Params>("SimpleBlobDetector_Params");
    mod.add_type<cv::CascadeClassifier>("CascadeClassifier");
    mod.add_type<cv::VideoCapture>("VideoCapture");

    mod.method("jlopencv_KeyPoint_set_pt", [](cv::KeyPoint &cobj, const Point2f &v) { cobj.pt = v; });
    mod.method("jlopencv_KeyPoint_set_size", [](cv::KeyPoint &cobj, const float &v) { cobj.size = v; });
    mod.method("jlopencv_KeyPoint_set_angle", [](cv::KeyPoint &cobj, const float &v) { cobj.angle = v; });
    mod.method("jlopencv_KeyPoint_set_response", [](cv::KeyPoint &cobj, const float &v) { cobj.response = v; });
    mod.method("jlopencv_KeyPoint_set_octave", [](cv::KeyPoint &cobj, const int &v) { cobj.octave = v; });
    mod.method("jlopencv_KeyPoint_set_class_id", [](cv::KeyPoint &cobj, const int &v) { cobj.class_id = v; });

    mod.method("jlopencv_KeyPoint_get_pt", [](const cv::KeyPoint &cobj) { return cobj.pt; });
    mod.method("jlopencv_KeyPoint_get_size", [](const cv::KeyPoint &cobj) { return cobj.size; });
    mod.method("jlopencv_KeyPoint_get_angle", [](const cv::KeyPoint &cobj) { return cobj.angle; });
    mod.method("jlopencv_KeyPoint_get_response", [](const cv::KeyPoint &cobj) { return cobj.response; });
    mod.method("jlopencv_KeyPoint_get_octave", [](const cv::KeyPoint &cobj) { return cobj.octave; });
    mod.method("jlopencv_KeyPoint_get_class_id", [](const cv::KeyPoint &cobj) { return cobj.class_id; });

    mod.method("jlopencv_cv_cv_CascadeClassifier_cv_CascadeClassifier_CascadeClassifier", [](string &filename) { return jlcxx::create<cv::CascadeClassifier>(filename); });
    mod.method("jlopencv_cv_cv_CascadeClassifier_cv_CascadeClassifier_detectMultiScale", [](cv::CascadeClassifier &cobj, Mat &image, double &scaleFactor, int &minNeighbors, int &flags, Size &minSize, Size &maxSize) {vector<Rect> objects; cobj.detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize);  return objects; });
    mod.method("jlopencv_cv_cv_CascadeClassifier_cv_CascadeClassifier_empty", [](cv::CascadeClassifier &cobj) { auto retval = cobj.empty();  return retval; });

    mod.method("jlopencv_cv_cv_VideoCapture_cv_VideoCapture_VideoCapture", [](string &filename, int &apiPreference) { return jlcxx::create<cv::VideoCapture>(filename, apiPreference); });
    mod.method("jlopencv_cv_cv_VideoCapture_cv_VideoCapture_VideoCapture", [](int &index, int &apiPreference) { return jlcxx::create<cv::VideoCapture>(index, apiPreference); });
    mod.method("jlopencv_cv_cv_VideoCapture_cv_VideoCapture_read", [](cv::VideoCapture &cobj, Mat &image) { auto retval = cobj.read(image);  return make_tuple(move(retval),move(image)); });
    mod.method("jlopencv_cv_cv_VideoCapture_cv_VideoCapture_release", [](cv::VideoCapture &cobj) { cobj.release();  ; });

    mod.method("jlopencv_cv_cv_Feature2D_cv_Feature2D_detect", [](cv::Ptr<cv::Feature2D> &cobj, Mat &image, Mat &mask) {vector<KeyPoint> keypoints; cobj->detect(image, keypoints, mask);  return keypoints; });

    mod.method("jlopencv_cv_cv_SimpleBlobDetector_create", [](SimpleBlobDetector_Params &parameters) { auto retval = cv::SimpleBlobDetector::create(parameters); return retval; });

    mod.method("jlopencv_cv_cv_imread", [](string &filename, int &flags) { auto retval = cv::imread(filename, flags); return retval; });
    mod.method("jlopencv_cv_cv_imshow", [](string &winname, Mat &mat) { cv::imshow(winname, mat); ; });
    mod.method("jlopencv_cv_cv_namedWindow", [](string &winname, int &flags) { cv::namedWindow(winname, flags); ; });
    mod.method("jlopencv_cv_cv_waitKey", [](int &delay) { auto retval = cv::waitKey(delay); return retval; });
    mod.method("jlopencv_cv_cv_rectangle", [](Mat &img, Point &pt1, Point &pt2, Scalar &color, int &thickness, int &lineType, int &shift) { cv::rectangle(img, pt1, pt2, color, thickness, lineType, shift); return img; });
    mod.method("jlopencv_cv_cv_cvtColor", [](Mat &src, int &code, Mat &dst, int &dstCn) { cv::cvtColor(src, dst, code, dstCn); return dst; });
    mod.method("jlopencv_cv_cv_equalizeHist", [](Mat &src, Mat &dst) { cv::equalizeHist(src, dst); return dst; });
    mod.method("jlopencv_cv_cv_destroyAllWindows", []() { cv::destroyAllWindows(); ; });

    mod.set_const("IMREAD_COLOR", (int)cv::IMREAD_COLOR);
    mod.set_const("IMREAD_GRAYSCALE", (int)cv::IMREAD_GRAYSCALE);

    mod.set_const("WINDOW_AUTOSIZE", (int)cv::WINDOW_AUTOSIZE);
    mod.set_const("WINDOW_FREERATIO", (int)cv::WINDOW_FREERATIO);
    mod.set_const("WINDOW_FULLSCREEN", (int)cv::WINDOW_FULLSCREEN);
    mod.set_const("WINDOW_GUI_EXPANDED", (int)cv::WINDOW_GUI_EXPANDED);
    mod.set_const("WINDOW_GUI_NORMAL", (int)cv::WINDOW_GUI_NORMAL);
    mod.set_const("WINDOW_KEEPRATIO", (int)cv::WINDOW_KEEPRATIO);
    mod.set_const("WINDOW_NORMAL", (int)cv::WINDOW_NORMAL);
    mod.set_const("WINDOW_OPENGL", (int)cv::WINDOW_OPENGL);

    mod.set_const("LINE_4", (int)cv::LINE_4);
    mod.set_const("LINE_8", (int)cv::LINE_8);
    mod.set_const("LINE_AA", (int)cv::LINE_AA);

    mod.set_const("CASCADE_DO_CANNY_PRUNING", (int)cv::CASCADE_DO_CANNY_PRUNING);
    mod.set_const("CASCADE_DO_ROUGH_SEARCH", (int)cv::CASCADE_DO_ROUGH_SEARCH);
    mod.set_const("CASCADE_FIND_BIGGEST_OBJECT", (int)cv::CASCADE_FIND_BIGGEST_OBJECT);
    mod.set_const("CASCADE_SCALE_IMAGE", (int)cv::CASCADE_SCALE_IMAGE);

    mod.set_const("COLOR_BGR2BGR555", (int)cv::COLOR_BGR2BGR555);
    mod.set_const("COLOR_BGR2BGR565", (int)cv::COLOR_BGR2BGR565);
    mod.set_const("COLOR_BGR2BGRA", (int)cv::COLOR_BGR2BGRA);
    mod.set_const("COLOR_BGR2GRAY", (int)cv::COLOR_BGR2GRAY);
    mod.set_const("COLOR_BGR2HLS", (int)cv::COLOR_BGR2HLS);
    mod.set_const("COLOR_BGR2HLS_FULL", (int)cv::COLOR_BGR2HLS_FULL);
    mod.set_const("COLOR_BGR2HSV", (int)cv::COLOR_BGR2HSV);
    mod.set_const("COLOR_BGR2HSV_FULL", (int)cv::COLOR_BGR2HSV_FULL);
    mod.set_const("COLOR_BGR2Lab", (int)cv::COLOR_BGR2Lab);
    mod.set_const("COLOR_BGR2LAB", (int)cv::COLOR_BGR2Lab);
    mod.set_const("COLOR_BGR2Luv", (int)cv::COLOR_BGR2Luv);
    mod.set_const("COLOR_BGR2LUV", (int)cv::COLOR_BGR2Luv);
    mod.set_const("COLOR_BGR2RGB", (int)cv::COLOR_BGR2RGB);
    mod.set_const("COLOR_BGR2RGBA", (int)cv::COLOR_BGR2RGBA);
    mod.set_const("COLOR_BGR2XYZ", (int)cv::COLOR_BGR2XYZ);
    mod.set_const("COLOR_BGR2YCrCb", (int)cv::COLOR_BGR2YCrCb);
    mod.set_const("COLOR_BGR2YCR_CB", (int)cv::COLOR_BGR2YCrCb);
    mod.set_const("COLOR_BGR2YUV", (int)cv::COLOR_BGR2YUV);
    mod.set_const("COLOR_BGR2YUV_I420", (int)cv::COLOR_BGR2YUV_I420);
    mod.set_const("COLOR_BGR2YUV_IYUV", (int)cv::COLOR_BGR2YUV_IYUV);
    mod.set_const("COLOR_BGR2YUV_YV12", (int)cv::COLOR_BGR2YUV_YV12);
    mod.set_const("COLOR_BGR5552BGR", (int)cv::COLOR_BGR5552BGR);
    mod.set_const("COLOR_BGR5552BGRA", (int)cv::COLOR_BGR5552BGRA);
    mod.set_const("COLOR_BGR5552GRAY", (int)cv::COLOR_BGR5552GRAY);
    mod.set_const("COLOR_BGR5552RGB", (int)cv::COLOR_BGR5552RGB);
    mod.set_const("COLOR_BGR5552RGBA", (int)cv::COLOR_BGR5552RGBA);
    mod.set_const("COLOR_BGR5652BGR", (int)cv::COLOR_BGR5652BGR);
    mod.set_const("COLOR_BGR5652BGRA", (int)cv::COLOR_BGR5652BGRA);
    mod.set_const("COLOR_BGR5652GRAY", (int)cv::COLOR_BGR5652GRAY);
    mod.set_const("COLOR_BGR5652RGB", (int)cv::COLOR_BGR5652RGB);
    mod.set_const("COLOR_BGR5652RGBA", (int)cv::COLOR_BGR5652RGBA);
    mod.set_const("COLOR_BGRA2BGR", (int)cv::COLOR_BGRA2BGR);
    mod.set_const("COLOR_BGRA2BGR555", (int)cv::COLOR_BGRA2BGR555);
    mod.set_const("COLOR_BGRA2BGR565", (int)cv::COLOR_BGRA2BGR565);
    mod.set_const("COLOR_BGRA2GRAY", (int)cv::COLOR_BGRA2GRAY);
    mod.set_const("COLOR_BGRA2RGB", (int)cv::COLOR_BGRA2RGB);
    mod.set_const("COLOR_BGRA2RGBA", (int)cv::COLOR_BGRA2RGBA);
    mod.set_const("COLOR_BGRA2YUV_I420", (int)cv::COLOR_BGRA2YUV_I420);
    mod.set_const("COLOR_BGRA2YUV_IYUV", (int)cv::COLOR_BGRA2YUV_IYUV);
    mod.set_const("COLOR_BGRA2YUV_YV12", (int)cv::COLOR_BGRA2YUV_YV12);
    mod.set_const("COLOR_BayerBG2BGR", (int)cv::COLOR_BayerBG2BGR);
    mod.set_const("COLOR_BAYER_BG2BGR", (int)cv::COLOR_BayerBG2BGR);
    mod.set_const("COLOR_BayerBG2BGRA", (int)cv::COLOR_BayerBG2BGRA);
    mod.set_const("COLOR_BAYER_BG2BGRA", (int)cv::COLOR_BayerBG2BGRA);
    mod.set_const("COLOR_BayerBG2BGR_EA", (int)cv::COLOR_BayerBG2BGR_EA);
    mod.set_const("COLOR_BAYER_BG2BGR_EA", (int)cv::COLOR_BayerBG2BGR_EA);
    mod.set_const("COLOR_BayerBG2BGR_VNG", (int)cv::COLOR_BayerBG2BGR_VNG);
    mod.set_const("COLOR_BAYER_BG2BGR_VNG", (int)cv::COLOR_BayerBG2BGR_VNG);
    mod.set_const("COLOR_BayerBG2GRAY", (int)cv::COLOR_BayerBG2GRAY);
    mod.set_const("COLOR_BAYER_BG2GRAY", (int)cv::COLOR_BayerBG2GRAY);
    mod.set_const("COLOR_BayerBG2RGB", (int)cv::COLOR_BayerBG2RGB);
    mod.set_const("COLOR_BAYER_BG2RGB", (int)cv::COLOR_BayerBG2RGB);
    mod.set_const("COLOR_BayerBG2RGBA", (int)cv::COLOR_BayerBG2RGBA);
    mod.set_const("COLOR_BAYER_BG2RGBA", (int)cv::COLOR_BayerBG2RGBA);
    mod.set_const("COLOR_BayerBG2RGB_EA", (int)cv::COLOR_BayerBG2RGB_EA);
    mod.set_const("COLOR_BAYER_BG2RGB_EA", (int)cv::COLOR_BayerBG2RGB_EA);
    mod.set_const("COLOR_BayerBG2RGB_VNG", (int)cv::COLOR_BayerBG2RGB_VNG);
    mod.set_const("COLOR_BAYER_BG2RGB_VNG", (int)cv::COLOR_BayerBG2RGB_VNG);
    mod.set_const("COLOR_BayerGB2BGR", (int)cv::COLOR_BayerGB2BGR);
    mod.set_const("COLOR_BAYER_GB2BGR", (int)cv::COLOR_BayerGB2BGR);
    mod.set_const("COLOR_BayerGB2BGRA", (int)cv::COLOR_BayerGB2BGRA);
    mod.set_const("COLOR_BAYER_GB2BGRA", (int)cv::COLOR_BayerGB2BGRA);
    mod.set_const("COLOR_BayerGB2BGR_EA", (int)cv::COLOR_BayerGB2BGR_EA);
    mod.set_const("COLOR_BAYER_GB2BGR_EA", (int)cv::COLOR_BayerGB2BGR_EA);
    mod.set_const("COLOR_BayerGB2BGR_VNG", (int)cv::COLOR_BayerGB2BGR_VNG);
    mod.set_const("COLOR_BAYER_GB2BGR_VNG", (int)cv::COLOR_BayerGB2BGR_VNG);
    mod.set_const("COLOR_BayerGB2GRAY", (int)cv::COLOR_BayerGB2GRAY);
    mod.set_const("COLOR_BAYER_GB2GRAY", (int)cv::COLOR_BayerGB2GRAY);
    mod.set_const("COLOR_BayerGB2RGB", (int)cv::COLOR_BayerGB2RGB);
    mod.set_const("COLOR_BAYER_GB2RGB", (int)cv::COLOR_BayerGB2RGB);
    mod.set_const("COLOR_BayerGB2RGBA", (int)cv::COLOR_BayerGB2RGBA);
    mod.set_const("COLOR_BAYER_GB2RGBA", (int)cv::COLOR_BayerGB2RGBA);
    mod.set_const("COLOR_BayerGB2RGB_EA", (int)cv::COLOR_BayerGB2RGB_EA);
    mod.set_const("COLOR_BAYER_GB2RGB_EA", (int)cv::COLOR_BayerGB2RGB_EA);
    mod.set_const("COLOR_BayerGB2RGB_VNG", (int)cv::COLOR_BayerGB2RGB_VNG);
    mod.set_const("COLOR_BAYER_GB2RGB_VNG", (int)cv::COLOR_BayerGB2RGB_VNG);
    mod.set_const("COLOR_BayerGR2BGR", (int)cv::COLOR_BayerGR2BGR);
    mod.set_const("COLOR_BAYER_GR2BGR", (int)cv::COLOR_BayerGR2BGR);
    mod.set_const("COLOR_BayerGR2BGRA", (int)cv::COLOR_BayerGR2BGRA);
    mod.set_const("COLOR_BAYER_GR2BGRA", (int)cv::COLOR_BayerGR2BGRA);
    mod.set_const("COLOR_BayerGR2BGR_EA", (int)cv::COLOR_BayerGR2BGR_EA);
    mod.set_const("COLOR_BAYER_GR2BGR_EA", (int)cv::COLOR_BayerGR2BGR_EA);
    mod.set_const("COLOR_BayerGR2BGR_VNG", (int)cv::COLOR_BayerGR2BGR_VNG);
    mod.set_const("COLOR_BAYER_GR2BGR_VNG", (int)cv::COLOR_BayerGR2BGR_VNG);
    mod.set_const("COLOR_BayerGR2GRAY", (int)cv::COLOR_BayerGR2GRAY);
    mod.set_const("COLOR_BAYER_GR2GRAY", (int)cv::COLOR_BayerGR2GRAY);
    mod.set_const("COLOR_BayerGR2RGB", (int)cv::COLOR_BayerGR2RGB);
    mod.set_const("COLOR_BAYER_GR2RGB", (int)cv::COLOR_BayerGR2RGB);
    mod.set_const("COLOR_BayerGR2RGBA", (int)cv::COLOR_BayerGR2RGBA);
    mod.set_const("COLOR_BAYER_GR2RGBA", (int)cv::COLOR_BayerGR2RGBA);
    mod.set_const("COLOR_BayerGR2RGB_EA", (int)cv::COLOR_BayerGR2RGB_EA);
    mod.set_const("COLOR_BAYER_GR2RGB_EA", (int)cv::COLOR_BayerGR2RGB_EA);
    mod.set_const("COLOR_BayerGR2RGB_VNG", (int)cv::COLOR_BayerGR2RGB_VNG);
    mod.set_const("COLOR_BAYER_GR2RGB_VNG", (int)cv::COLOR_BayerGR2RGB_VNG);
    mod.set_const("COLOR_BayerRG2BGR", (int)cv::COLOR_BayerRG2BGR);
    mod.set_const("COLOR_BAYER_RG2BGR", (int)cv::COLOR_BayerRG2BGR);
    mod.set_const("COLOR_BayerRG2BGRA", (int)cv::COLOR_BayerRG2BGRA);
    mod.set_const("COLOR_BAYER_RG2BGRA", (int)cv::COLOR_BayerRG2BGRA);
    mod.set_const("COLOR_BayerRG2BGR_EA", (int)cv::COLOR_BayerRG2BGR_EA);
    mod.set_const("COLOR_BAYER_RG2BGR_EA", (int)cv::COLOR_BayerRG2BGR_EA);
    mod.set_const("COLOR_BayerRG2BGR_VNG", (int)cv::COLOR_BayerRG2BGR_VNG);
    mod.set_const("COLOR_BAYER_RG2BGR_VNG", (int)cv::COLOR_BayerRG2BGR_VNG);
    mod.set_const("COLOR_BayerRG2GRAY", (int)cv::COLOR_BayerRG2GRAY);
    mod.set_const("COLOR_BAYER_RG2GRAY", (int)cv::COLOR_BayerRG2GRAY);
    mod.set_const("COLOR_BayerRG2RGB", (int)cv::COLOR_BayerRG2RGB);
    mod.set_const("COLOR_BAYER_RG2RGB", (int)cv::COLOR_BayerRG2RGB);
    mod.set_const("COLOR_BayerRG2RGBA", (int)cv::COLOR_BayerRG2RGBA);
    mod.set_const("COLOR_BAYER_RG2RGBA", (int)cv::COLOR_BayerRG2RGBA);
    mod.set_const("COLOR_BayerRG2RGB_EA", (int)cv::COLOR_BayerRG2RGB_EA);
    mod.set_const("COLOR_BAYER_RG2RGB_EA", (int)cv::COLOR_BayerRG2RGB_EA);
    mod.set_const("COLOR_BayerRG2RGB_VNG", (int)cv::COLOR_BayerRG2RGB_VNG);
    mod.set_const("COLOR_BAYER_RG2RGB_VNG", (int)cv::COLOR_BayerRG2RGB_VNG);
    mod.set_const("COLOR_COLORCVT_MAX", (int)cv::COLOR_COLORCVT_MAX);
    mod.set_const("COLOR_GRAY2BGR", (int)cv::COLOR_GRAY2BGR);
    mod.set_const("COLOR_GRAY2BGR555", (int)cv::COLOR_GRAY2BGR555);
    mod.set_const("COLOR_GRAY2BGR565", (int)cv::COLOR_GRAY2BGR565);
    mod.set_const("COLOR_GRAY2BGRA", (int)cv::COLOR_GRAY2BGRA);
    mod.set_const("COLOR_GRAY2RGB", (int)cv::COLOR_GRAY2RGB);
    mod.set_const("COLOR_GRAY2RGBA", (int)cv::COLOR_GRAY2RGBA);
    mod.set_const("COLOR_HLS2BGR", (int)cv::COLOR_HLS2BGR);
    mod.set_const("COLOR_HLS2BGR_FULL", (int)cv::COLOR_HLS2BGR_FULL);
    mod.set_const("COLOR_HLS2RGB", (int)cv::COLOR_HLS2RGB);
    mod.set_const("COLOR_HLS2RGB_FULL", (int)cv::COLOR_HLS2RGB_FULL);
    mod.set_const("COLOR_HSV2BGR", (int)cv::COLOR_HSV2BGR);
    mod.set_const("COLOR_HSV2BGR_FULL", (int)cv::COLOR_HSV2BGR_FULL);
    mod.set_const("COLOR_HSV2RGB", (int)cv::COLOR_HSV2RGB);
    mod.set_const("COLOR_HSV2RGB_FULL", (int)cv::COLOR_HSV2RGB_FULL);
    mod.set_const("COLOR_LBGR2Lab", (int)cv::COLOR_LBGR2Lab);
    mod.set_const("COLOR_LBGR2LAB", (int)cv::COLOR_LBGR2Lab);
    mod.set_const("COLOR_LBGR2Luv", (int)cv::COLOR_LBGR2Luv);
    mod.set_const("COLOR_LBGR2LUV", (int)cv::COLOR_LBGR2Luv);
    mod.set_const("COLOR_LRGB2Lab", (int)cv::COLOR_LRGB2Lab);
    mod.set_const("COLOR_LRGB2LAB", (int)cv::COLOR_LRGB2Lab);
    mod.set_const("COLOR_LRGB2Luv", (int)cv::COLOR_LRGB2Luv);
    mod.set_const("COLOR_LRGB2LUV", (int)cv::COLOR_LRGB2Luv);
    mod.set_const("COLOR_Lab2BGR", (int)cv::COLOR_Lab2BGR);
    mod.set_const("COLOR_LAB2BGR", (int)cv::COLOR_Lab2BGR);
    mod.set_const("COLOR_Lab2LBGR", (int)cv::COLOR_Lab2LBGR);
    mod.set_const("COLOR_LAB2LBGR", (int)cv::COLOR_Lab2LBGR);
    mod.set_const("COLOR_Lab2LRGB", (int)cv::COLOR_Lab2LRGB);
    mod.set_const("COLOR_LAB2LRGB", (int)cv::COLOR_Lab2LRGB);
    mod.set_const("COLOR_Lab2RGB", (int)cv::COLOR_Lab2RGB);
    mod.set_const("COLOR_LAB2RGB", (int)cv::COLOR_Lab2RGB);
    mod.set_const("COLOR_Luv2BGR", (int)cv::COLOR_Luv2BGR);
    mod.set_const("COLOR_LUV2BGR", (int)cv::COLOR_Luv2BGR);
    mod.set_const("COLOR_Luv2LBGR", (int)cv::COLOR_Luv2LBGR);
    mod.set_const("COLOR_LUV2LBGR", (int)cv::COLOR_Luv2LBGR);
    mod.set_const("COLOR_Luv2LRGB", (int)cv::COLOR_Luv2LRGB);
    mod.set_const("COLOR_LUV2LRGB", (int)cv::COLOR_Luv2LRGB);
    mod.set_const("COLOR_Luv2RGB", (int)cv::COLOR_Luv2RGB);
    mod.set_const("COLOR_LUV2RGB", (int)cv::COLOR_Luv2RGB);
    mod.set_const("COLOR_RGB2BGR", (int)cv::COLOR_RGB2BGR);
    mod.set_const("COLOR_RGB2BGR555", (int)cv::COLOR_RGB2BGR555);
    mod.set_const("COLOR_RGB2BGR565", (int)cv::COLOR_RGB2BGR565);
    mod.set_const("COLOR_RGB2BGRA", (int)cv::COLOR_RGB2BGRA);
    mod.set_const("COLOR_RGB2GRAY", (int)cv::COLOR_RGB2GRAY);
    mod.set_const("COLOR_RGB2HLS", (int)cv::COLOR_RGB2HLS);
    mod.set_const("COLOR_RGB2HLS_FULL", (int)cv::COLOR_RGB2HLS_FULL);
    mod.set_const("COLOR_RGB2HSV", (int)cv::COLOR_RGB2HSV);
    mod.set_const("COLOR_RGB2HSV_FULL", (int)cv::COLOR_RGB2HSV_FULL);
    mod.set_const("COLOR_RGB2Lab", (int)cv::COLOR_RGB2Lab);
    mod.set_const("COLOR_RGB2LAB", (int)cv::COLOR_RGB2Lab);
    mod.set_const("COLOR_RGB2Luv", (int)cv::COLOR_RGB2Luv);
    mod.set_const("COLOR_RGB2LUV", (int)cv::COLOR_RGB2Luv);
    mod.set_const("COLOR_RGB2RGBA", (int)cv::COLOR_RGB2RGBA);
    mod.set_const("COLOR_RGB2XYZ", (int)cv::COLOR_RGB2XYZ);
    mod.set_const("COLOR_RGB2YCrCb", (int)cv::COLOR_RGB2YCrCb);
    mod.set_const("COLOR_RGB2YCR_CB", (int)cv::COLOR_RGB2YCrCb);
    mod.set_const("COLOR_RGB2YUV", (int)cv::COLOR_RGB2YUV);
    mod.set_const("COLOR_RGB2YUV_I420", (int)cv::COLOR_RGB2YUV_I420);
    mod.set_const("COLOR_RGB2YUV_IYUV", (int)cv::COLOR_RGB2YUV_IYUV);
    mod.set_const("COLOR_RGB2YUV_YV12", (int)cv::COLOR_RGB2YUV_YV12);
    mod.set_const("COLOR_RGBA2BGR", (int)cv::COLOR_RGBA2BGR);
    mod.set_const("COLOR_RGBA2BGR555", (int)cv::COLOR_RGBA2BGR555);
    mod.set_const("COLOR_RGBA2BGR565", (int)cv::COLOR_RGBA2BGR565);
    mod.set_const("COLOR_RGBA2BGRA", (int)cv::COLOR_RGBA2BGRA);
    mod.set_const("COLOR_RGBA2GRAY", (int)cv::COLOR_RGBA2GRAY);
    mod.set_const("COLOR_RGBA2RGB", (int)cv::COLOR_RGBA2RGB);
    mod.set_const("COLOR_RGBA2YUV_I420", (int)cv::COLOR_RGBA2YUV_I420);
    mod.set_const("COLOR_RGBA2YUV_IYUV", (int)cv::COLOR_RGBA2YUV_IYUV);
    mod.set_const("COLOR_RGBA2YUV_YV12", (int)cv::COLOR_RGBA2YUV_YV12);
    mod.set_const("COLOR_RGBA2mRGBA", (int)cv::COLOR_RGBA2mRGBA);
    mod.set_const("COLOR_RGBA2M_RGBA", (int)cv::COLOR_RGBA2mRGBA);
    mod.set_const("COLOR_XYZ2BGR", (int)cv::COLOR_XYZ2BGR);
    mod.set_const("COLOR_XYZ2RGB", (int)cv::COLOR_XYZ2RGB);
    mod.set_const("COLOR_YCrCb2BGR", (int)cv::COLOR_YCrCb2BGR);
    mod.set_const("COLOR_YCR_CB2BGR", (int)cv::COLOR_YCrCb2BGR);
    mod.set_const("COLOR_YCrCb2RGB", (int)cv::COLOR_YCrCb2RGB);
    mod.set_const("COLOR_YCR_CB2RGB", (int)cv::COLOR_YCrCb2RGB);
    mod.set_const("COLOR_YUV2BGR", (int)cv::COLOR_YUV2BGR);
    mod.set_const("COLOR_YUV2BGRA_I420", (int)cv::COLOR_YUV2BGRA_I420);
    mod.set_const("COLOR_YUV2BGRA_IYUV", (int)cv::COLOR_YUV2BGRA_IYUV);
    mod.set_const("COLOR_YUV2BGRA_NV12", (int)cv::COLOR_YUV2BGRA_NV12);
    mod.set_const("COLOR_YUV2BGRA_NV21", (int)cv::COLOR_YUV2BGRA_NV21);
    mod.set_const("COLOR_YUV2BGRA_UYNV", (int)cv::COLOR_YUV2BGRA_UYNV);
    mod.set_const("COLOR_YUV2BGRA_UYVY", (int)cv::COLOR_YUV2BGRA_UYVY);
    mod.set_const("COLOR_YUV2BGRA_Y422", (int)cv::COLOR_YUV2BGRA_Y422);
    mod.set_const("COLOR_YUV2BGRA_YUNV", (int)cv::COLOR_YUV2BGRA_YUNV);
    mod.set_const("COLOR_YUV2BGRA_YUY2", (int)cv::COLOR_YUV2BGRA_YUY2);
    mod.set_const("COLOR_YUV2BGRA_YUYV", (int)cv::COLOR_YUV2BGRA_YUYV);
    mod.set_const("COLOR_YUV2BGRA_YV12", (int)cv::COLOR_YUV2BGRA_YV12);
    mod.set_const("COLOR_YUV2BGRA_YVYU", (int)cv::COLOR_YUV2BGRA_YVYU);
    mod.set_const("COLOR_YUV2BGR_I420", (int)cv::COLOR_YUV2BGR_I420);
    mod.set_const("COLOR_YUV2BGR_IYUV", (int)cv::COLOR_YUV2BGR_IYUV);
    mod.set_const("COLOR_YUV2BGR_NV12", (int)cv::COLOR_YUV2BGR_NV12);
    mod.set_const("COLOR_YUV2BGR_NV21", (int)cv::COLOR_YUV2BGR_NV21);
    mod.set_const("COLOR_YUV2BGR_UYNV", (int)cv::COLOR_YUV2BGR_UYNV);
    mod.set_const("COLOR_YUV2BGR_UYVY", (int)cv::COLOR_YUV2BGR_UYVY);
    mod.set_const("COLOR_YUV2BGR_Y422", (int)cv::COLOR_YUV2BGR_Y422);
    mod.set_const("COLOR_YUV2BGR_YUNV", (int)cv::COLOR_YUV2BGR_YUNV);
    mod.set_const("COLOR_YUV2BGR_YUY2", (int)cv::COLOR_YUV2BGR_YUY2);
    mod.set_const("COLOR_YUV2BGR_YUYV", (int)cv::COLOR_YUV2BGR_YUYV);
    mod.set_const("COLOR_YUV2BGR_YV12", (int)cv::COLOR_YUV2BGR_YV12);
    mod.set_const("COLOR_YUV2BGR_YVYU", (int)cv::COLOR_YUV2BGR_YVYU);
    mod.set_const("COLOR_YUV2GRAY_420", (int)cv::COLOR_YUV2GRAY_420);
    mod.set_const("COLOR_YUV2GRAY_I420", (int)cv::COLOR_YUV2GRAY_I420);
    mod.set_const("COLOR_YUV2GRAY_IYUV", (int)cv::COLOR_YUV2GRAY_IYUV);
    mod.set_const("COLOR_YUV2GRAY_NV12", (int)cv::COLOR_YUV2GRAY_NV12);
    mod.set_const("COLOR_YUV2GRAY_NV21", (int)cv::COLOR_YUV2GRAY_NV21);
    mod.set_const("COLOR_YUV2GRAY_UYNV", (int)cv::COLOR_YUV2GRAY_UYNV);
    mod.set_const("COLOR_YUV2GRAY_UYVY", (int)cv::COLOR_YUV2GRAY_UYVY);
    mod.set_const("COLOR_YUV2GRAY_Y422", (int)cv::COLOR_YUV2GRAY_Y422);
    mod.set_const("COLOR_YUV2GRAY_YUNV", (int)cv::COLOR_YUV2GRAY_YUNV);
    mod.set_const("COLOR_YUV2GRAY_YUY2", (int)cv::COLOR_YUV2GRAY_YUY2);
    mod.set_const("COLOR_YUV2GRAY_YUYV", (int)cv::COLOR_YUV2GRAY_YUYV);
    mod.set_const("COLOR_YUV2GRAY_YV12", (int)cv::COLOR_YUV2GRAY_YV12);
    mod.set_const("COLOR_YUV2GRAY_YVYU", (int)cv::COLOR_YUV2GRAY_YVYU);
    mod.set_const("COLOR_YUV2RGB", (int)cv::COLOR_YUV2RGB);
    mod.set_const("COLOR_YUV2RGBA_I420", (int)cv::COLOR_YUV2RGBA_I420);
    mod.set_const("COLOR_YUV2RGBA_IYUV", (int)cv::COLOR_YUV2RGBA_IYUV);
    mod.set_const("COLOR_YUV2RGBA_NV12", (int)cv::COLOR_YUV2RGBA_NV12);
    mod.set_const("COLOR_YUV2RGBA_NV21", (int)cv::COLOR_YUV2RGBA_NV21);
    mod.set_const("COLOR_YUV2RGBA_UYNV", (int)cv::COLOR_YUV2RGBA_UYNV);
    mod.set_const("COLOR_YUV2RGBA_UYVY", (int)cv::COLOR_YUV2RGBA_UYVY);
    mod.set_const("COLOR_YUV2RGBA_Y422", (int)cv::COLOR_YUV2RGBA_Y422);
    mod.set_const("COLOR_YUV2RGBA_YUNV", (int)cv::COLOR_YUV2RGBA_YUNV);
    mod.set_const("COLOR_YUV2RGBA_YUY2", (int)cv::COLOR_YUV2RGBA_YUY2);
    mod.set_const("COLOR_YUV2RGBA_YUYV", (int)cv::COLOR_YUV2RGBA_YUYV);
    mod.set_const("COLOR_YUV2RGBA_YV12", (int)cv::COLOR_YUV2RGBA_YV12);
    mod.set_const("COLOR_YUV2RGBA_YVYU", (int)cv::COLOR_YUV2RGBA_YVYU);
    mod.set_const("COLOR_YUV2RGB_I420", (int)cv::COLOR_YUV2RGB_I420);
    mod.set_const("COLOR_YUV2RGB_IYUV", (int)cv::COLOR_YUV2RGB_IYUV);
    mod.set_const("COLOR_YUV2RGB_NV12", (int)cv::COLOR_YUV2RGB_NV12);
    mod.set_const("COLOR_YUV2RGB_NV21", (int)cv::COLOR_YUV2RGB_NV21);
    mod.set_const("COLOR_YUV2RGB_UYNV", (int)cv::COLOR_YUV2RGB_UYNV);
    mod.set_const("COLOR_YUV2RGB_UYVY", (int)cv::COLOR_YUV2RGB_UYVY);
    mod.set_const("COLOR_YUV2RGB_Y422", (int)cv::COLOR_YUV2RGB_Y422);
    mod.set_const("COLOR_YUV2RGB_YUNV", (int)cv::COLOR_YUV2RGB_YUNV);
    mod.set_const("COLOR_YUV2RGB_YUY2", (int)cv::COLOR_YUV2RGB_YUY2);
    mod.set_const("COLOR_YUV2RGB_YUYV", (int)cv::COLOR_YUV2RGB_YUYV);
    mod.set_const("COLOR_YUV2RGB_YV12", (int)cv::COLOR_YUV2RGB_YV12);
    mod.set_const("COLOR_YUV2RGB_YVYU", (int)cv::COLOR_YUV2RGB_YVYU);
    mod.set_const("COLOR_YUV420p2BGR", (int)cv::COLOR_YUV420p2BGR);
    mod.set_const("COLOR_YUV420P2BGR", (int)cv::COLOR_YUV420p2BGR);
    mod.set_const("COLOR_YUV420p2BGRA", (int)cv::COLOR_YUV420p2BGRA);
    mod.set_const("COLOR_YUV420P2BGRA", (int)cv::COLOR_YUV420p2BGRA);
    mod.set_const("COLOR_YUV420p2GRAY", (int)cv::COLOR_YUV420p2GRAY);
    mod.set_const("COLOR_YUV420P2GRAY", (int)cv::COLOR_YUV420p2GRAY);
    mod.set_const("COLOR_YUV420p2RGB", (int)cv::COLOR_YUV420p2RGB);
    mod.set_const("COLOR_YUV420P2RGB", (int)cv::COLOR_YUV420p2RGB);
    mod.set_const("COLOR_YUV420p2RGBA", (int)cv::COLOR_YUV420p2RGBA);
    mod.set_const("COLOR_YUV420P2RGBA", (int)cv::COLOR_YUV420p2RGBA);
    mod.set_const("COLOR_YUV420sp2BGR", (int)cv::COLOR_YUV420sp2BGR);
    mod.set_const("COLOR_YUV420SP2BGR", (int)cv::COLOR_YUV420sp2BGR);
    mod.set_const("COLOR_YUV420sp2BGRA", (int)cv::COLOR_YUV420sp2BGRA);
    mod.set_const("COLOR_YUV420SP2BGRA", (int)cv::COLOR_YUV420sp2BGRA);
    mod.set_const("COLOR_YUV420sp2GRAY", (int)cv::COLOR_YUV420sp2GRAY);
    mod.set_const("COLOR_YUV420SP2GRAY", (int)cv::COLOR_YUV420sp2GRAY);
    mod.set_const("COLOR_YUV420sp2RGB", (int)cv::COLOR_YUV420sp2RGB);
    mod.set_const("COLOR_YUV420SP2RGB", (int)cv::COLOR_YUV420sp2RGB);
    mod.set_const("COLOR_YUV420sp2RGBA", (int)cv::COLOR_YUV420sp2RGBA);
    mod.set_const("COLOR_YUV420SP2RGBA", (int)cv::COLOR_YUV420sp2RGBA);
    mod.set_const("COLOR_mRGBA2RGBA", (int)cv::COLOR_mRGBA2RGBA);
    mod.set_const("COLOR_M_RGBA2RGBA", (int)cv::COLOR_mRGBA2RGBA);

    mod.set_const("CAP_ANY", (int)cv::CAP_ANY);
}
