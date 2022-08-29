// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRANSFORMS_DET_HPP
#define OPENCV_TRANSFORMS_DET_HPP


namespace cv{
    namespace imgaug{
        namespace det{

            //! @addtogroup det
            //! @{

            class CV_EXPORTS_W Transform{
            public:
                CV_WRAP virtual void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& target) const = 0;
                CV_WRAP virtual ~Transform() = default;
            };

            class CV_EXPORTS_W Compose{
            public:
                CV_WRAP explicit Compose(std::vector<cv::Ptr<cv::imgaug::det::Transform> >& transforms);
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& target) const;

                std::vector<cv::Ptr<cv::imgaug::det::Transform> > transforms;
            };

            class CV_EXPORTS_W RandomFlip: cv::imgaug::det::Transform{
            public:
                CV_WRAP explicit RandomFlip(int flipCode=0, float p=0.5);
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& target) const;
                void flipBoundingBox(std::vector<cv::Rect>& target, const Size& size) const;

                int flipCode;
                float p;
            };

//        class CV_EXPORTS_W RandomCrop: cv::det::Transform{
//        public:
//            CV_WRAP explicit RandomCrop(const Size& sz, const Vec4i& padding=Vec4i() , bool pad_if_need=false, const Scalar& fill=Scalar(), int padding_mode=BORDER_CONSTANT);
//            CV_WRAP void call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target) const;
//
//            const Size sz;
//            Vec4i padding;
//            bool pad_if_need;
//            Scalar fill;
//            int padding_mode;
//        };

            class CV_EXPORTS_W Resize: cv::imgaug::det::Transform{
            public:
                CV_WRAP explicit Resize(const Size& size, int interpolation=INTER_NEAREST);
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& target) const;
                void resizeBoundingBox(std::vector<cv::Rect>& target, const Size& imgSize) const;

                const Size size;
                int interpolation;
            };

            class CV_EXPORTS_W Convert: cv::imgaug::det::Transform{
            public:
                CV_WRAP explicit Convert(int code);
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& target) const;

                int code;
            };
            //! @}
        }
    }
}

#endif //OPENCV_TRANSFORMS_DET_HPP
