#ifndef OPENCV_AUG_TRANSFORMS_HPP
#define OPENCV_AUG_TRANSFORMS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cv{
    namespace imgaug{
        /*
         * Base class for all augmentation classes
         */
        class CV_EXPORTS_W Transform{
        public:
            CV_WRAP virtual void call(InputArray src, OutputArray dst) const = 0;
            CV_WRAP virtual ~Transform() = default;
        };

        class CV_EXPORTS_W Compose{
        public:
            CV_WRAP explicit Compose(std::vector<Ptr<Transform> >& transforms);
            CV_WRAP void call(InputArray src, OutputArray dst) const;

            std::vector<Ptr<Transform> > transforms;
        };

        CV_EXPORTS_W void randomCrop(InputArray src, OutputArray dst, const Size& sz, const Vec4i& padding=Vec4i() , bool pad_if_need=false, int fill=0, int padding_mode=BORDER_CONSTANT);

//    CV_EXPORTS_W void randomCropV1(InputOutputArray src, const Size& sz, const Vec4i& padding=Vec4i() , bool pad_if_need=false, int fill=0, int padding_mode=BORDER_CONSTANT);

        CV_EXPORTS_W void randomFlip(InputArray src, OutputArray dst, int flipCode=0, double p=0.5);

        class CV_EXPORTS_W RandomCrop: public Transform{
        public:
            CV_WRAP explicit RandomCrop(const Size& sz, const Vec4i& padding=Vec4i(0,0,0,0), bool pad_if_need=false, int fill=0, int padding_mode=BORDER_CONSTANT);
            CV_WRAP ~RandomCrop() override = default;
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Size sz;
            Vec4i padding;
            bool pad_if_need;
            int fill;
            int padding_mode;
        };

        class CV_EXPORTS_W RandomFlip: public Transform{
        public:
            CV_WRAP explicit RandomFlip(int flipCode=0, double p=0.5);
            CV_WRAP ~RandomFlip() override = default;
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            int flipCode;
            double p;
        };

        /*
         * All interpolation types: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121ac97d8e4880d8b5d509e96825c7522deb
         */
        class CV_EXPORTS_W Resize: public Transform{
        public:
            CV_WRAP explicit Resize(const Size& sz, int interpolation=INTER_LINEAR);
            CV_WRAP ~Resize() override = default;
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Size sz;
            int interpolation;
        };

        CV_EXPORTS_W void centerCrop(InputArray src, OutputArray dst, const Size& size);

        class CV_EXPORTS_W CenterCrop : public Transform {
        public:
            CV_WRAP explicit CenterCrop(const Size& size);
            CV_WRAP ~CenterCrop() override = default;
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Size size;
        };

        class CV_EXPORTS_W Pad : public Transform{
        public:
            CV_WRAP explicit Pad(const Vec4i& padding, const Scalar& = Scalar(), int padding_mode = BORDER_CONSTANT);
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Vec4i padding;
            const Scalar fill;
            int padding_mode;
        };

        CV_EXPORTS void randomResizedCrop(InputArray src, OutputArray dst, const Size& size, const Vec2d& scale = Vec2d(0.08, 1.0), const Vec2d& ratio = Vec2d(3.0 / 4.0, 4.0 / 3.0), int interpolation = INTER_LINEAR);

        class CV_EXPORTS_W RandomResizedCrop : public Transform {
        public:
            CV_WRAP explicit RandomResizedCrop(const Size& size, const Vec2d& scale = Vec2d(0.08, 1.0), const Vec2d& ratio = Vec2d(3.0 / 4.0, 4.0 / 3.0), int interpolation = INTER_LINEAR);
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Size size;
            Vec2d scale;
            Vec2d ratio;
            int interpolation;
        };

        CV_EXPORTS void colorJitter(InputArray src, OutputArray dst, const Vec2d& brightness=Vec2d(), const Vec2d& contrast=Vec2d(), const Vec2d& saturation=Vec2d(), const Vec2d& hue=Vec2d());

        class CV_EXPORTS_W ColorJitter : public Transform {
        public:
            CV_WRAP explicit ColorJitter(const Vec2d& brightness=Vec2d(), const Vec2d& contrast=Vec2d(), const Vec2d& saturation=Vec2d(), const Vec2d& hue=Vec2d());
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Vec2d brightness;
            Vec2d contrast;
            Vec2d saturation;
            Vec2d hue;
        };

        CV_EXPORTS void randomRotation(InputArray src, OutputArray dst, const Vec2d& degrees, int interpolation=INTER_NEAREST, bool expand=false, const Point2f& center=Point2f(), int fill=0);

        class CV_EXPORTS_W RandomRotation : public Transform {
        public:
            CV_WRAP explicit RandomRotation(const Vec2d& degrees, int interpolation=INTER_NEAREST, bool expand=false, const Point2f& center=Point2f(), int fill=0);
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Vec2d degrees;
            int interpolation;
            bool expand;
            Point2f center;
            int fill;
        };

        class CV_EXPORTS_W GrayScale : public Transform {
        public:
            CV_WRAP explicit GrayScale(int num_channels=1);
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            int num_channels;
        };

        CV_EXPORTS void randomGrayScale(InputArray src, OutputArray dst, double p=0.1);

        class CV_EXPORTS_W RandomGrayScale : public Transform {
        public:
            CV_WRAP explicit RandomGrayScale(double p=0.1);
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            double p;
        };

        CV_EXPORTS void randomErasing(InputArray src, OutputArray dst, double p=0.5, const Vec2d& scale=Vec2d(0.02, 0.33), const Vec2d& ratio=Vec2d(0.3, 0.33), const Scalar& value=Scalar(0, 100, 100), bool inplace=false);

        class CV_EXPORTS_W RandomErasing : public Transform {
        public:
            CV_WRAP explicit RandomErasing(double p=0.5, const Vec2d& scale=Vec2d(0.02, 0.33), const Vec2d& ratio=Vec2d(0.3, 0.33), const Scalar& value=Scalar(0, 100, 100), bool inplace=false);
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            double p;
            Vec2d scale;
            Vec2d ratio;
            Scalar value;
            bool inplace;
        };

        // NOTE: After normalizing, data are represented as 32-bit float in range(0,1)
        class CV_EXPORTS_W Normalize : public Transform {
        public:
            CV_WRAP explicit Normalize(const Scalar& mean=Scalar(0,0,0,0), const Scalar& std=Scalar(1,1,1,1));
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Scalar mean;
            Scalar std;
        };

        CV_EXPORTS void gaussianBlur(InputArray src, OutputArray dst, const Size& kernel_size, const Vec2f& sigma=Vec2f(0.1, 2.0));

        class CV_EXPORTS_W GaussianBlurAug : public Transform {
        public:
            CV_WRAP explicit GaussianBlurAug(const Size& kernel_size, const Vec2f& sigma=Vec2f(0.1, 2.0));
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Size kernel_size;
            Vec2f sigma;
        };

        CV_EXPORTS void randomAffine(InputArray src, OutputArray dst, const Vec2f& degrees=Vec2f(0., 0.), const Vec2f& translations=Vec2f(0., 0.), const Vec2f& scales=Vec2f(1., 1.), const Vec4f& shears=Vec4f(0., 0., 0., 0.), int interpolation=INTER_NEAREST, const Scalar& fill=Scalar(), const Point2i& center=Point2i(-1, -1));

        class CV_EXPORTS_W RandomAffine: public Transform{
        public:
            CV_WRAP explicit RandomAffine(const Vec2f& degrees=Vec2f(0., 0.), const Vec2f& translations=Vec2f(0., 0.), const Vec2f& scales=Vec2f(1., 1.), const Vec4f& shears=Vec4f(0., 0., 0., 0.), int interpolation=INTER_NEAREST, const Scalar& fill=Scalar(), const Point2i& center=Point2i(-1, -1));
            CV_WRAP void call(InputArray src, OutputArray dst) const override;

            Vec2f degrees;
            Vec2f translations;
            Vec2f scales;
            Vec4f shears;
            int interpolation;
            Scalar fill;
            Point2i center;

        };
    }


}

#endif
