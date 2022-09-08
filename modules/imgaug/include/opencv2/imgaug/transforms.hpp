// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_AUG_TRANSFORMS_HPP
#define OPENCV_AUG_TRANSFORMS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>


namespace cv{
    //! Data augmentation module
    namespace imgaug{

    //! @addtogroup imgaug
    //! @{

    //! Base class for all data augmentation classes.
    class CV_EXPORTS_W Transform{
    public:
        CV_WRAP virtual void call(InputArray src, OutputArray dst) const = 0;
        CV_WRAP virtual ~Transform() = default;
    };

    //! Combine a series of data augmentation methods into one and apply them sequentially.
    class CV_EXPORTS_W Compose{
    public:
        /** @brief Initialize the Compose class by passing a series of data augmentation you want to apply.
         *
         * @param transforms Series of data augmentation methods. All data augmentation classes should inherited from cv::imgaug::Transform.
         */
        CV_WRAP explicit Compose(std::vector<Ptr<Transform> >& transforms);
        /** @brief Call composed data augmentation methods, apply them to the input image sequentially.
         *
         * @param src Source image.
         * @param dst Destination image.
         *
         * @note Some data augmentation methods only support images in certain formats.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const;

        //! vector of the pointers to the data augmentation instances.
        std::vector<Ptr<Transform> > transforms;
    };

    //! Crop the given image at a random location
    class CV_EXPORTS_W RandomCrop: public Transform{
    public:
        /** @brief Initialize the RandomCrop class.
         *
         * @param sz Size of the cropped image.
         * @param padding Padding on the borders of the source image. Four element tuple needs to be provided,
         * which is the padding for the top, bottom, left and right respectively. By default no padding is added.
         * @param pad_if_need When the cropped size is smaller than the source image (with padding), exception will raise.
         * Set this value to true to automatically pad the image to avoid this exception.
         * @param fill Fill value of the padded pixels. By default is 0.
         * @param padding_mode Type of padding. Default is #BORDER_CONSTANT, see #BorderTypes for details.
         */
        CV_WRAP explicit RandomCrop(const Size& sz, const Vec4i& padding=Vec4i(0,0,0,0), bool pad_if_need=false, int fill=0, int padding_mode=BORDER_CONSTANT);

        CV_WRAP ~RandomCrop() override = default;

        /** @brief Apply augmentation method on source image, this operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Size sz;
        Vec4i padding;
        bool pad_if_need;
        int fill;
        int padding_mode;
    };

    //! Flip the image randomly along specified axes.
    class CV_EXPORTS_W RandomFlip: public Transform{
    public:
        /** Initialize the RandomFlip class.
         *
         * @param flipCode flipCode to specify the axis along which image is flipped. Set
         * 0 for vertical axis, positive for horizontal axis, negative for both axes.
         * \f[\texttt{dst} _{ij} =
           \left\{
           \begin{array}{l l}
           \texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\
           \texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\
           \texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\
           \end{array}
           \right.\f]
         * @param p Probability to apply this method. p should be in range 0 to 1, larger p denotes higher probability.
         */
        CV_WRAP explicit RandomFlip(int flipCode=0, double p=0.5);

        CV_WRAP ~RandomFlip() override = default;

        /** @brief Apply augmentation method on source image, this operation is inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        int flipCode;
        double p;
    };

    //! Resize the image to specified size
    class CV_EXPORTS_W Resize: public Transform{
    public:
        /** @brief Initialize the Resize class.
         *
         * @param sz Size of the resized image.
         * @param interpolation Interpolation mode. Refer to #InterpolationFlags for more details.
         */
        CV_WRAP explicit Resize(const Size& sz, int interpolation=INTER_LINEAR);

        CV_WRAP ~Resize() override = default;

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Size sz;
        int interpolation;
    };

    //! Crop the given image at the center
    class CV_EXPORTS_W CenterCrop : public Transform {
    public:
        /** @brief Initialize the CenterCrop class.
         *
         * @param size Size of the cropped image.
         */
        CV_WRAP explicit CenterCrop(const Size& size);

        CV_WRAP ~CenterCrop() override = default;

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Size size;
    };

    //! Pad the given image on the borders.
    class CV_EXPORTS_W Pad : public Transform{
    public:
        /** Initialize the Pad class.
         *
         * @param padding Padding on the borders of the source image. Four-elements tuple needs to be provided,
         * which is the padding for the top, bottom, left and right respectively.
         * @param fill Fill value of the padded pixels. By default fill value is 0 for all channels.
         * @param padding_mode Type of padding. Default is #BORDER_CONSTANT, see #BorderTypes for details.
         */
        CV_WRAP explicit Pad(const Vec4i& padding, const Scalar& fill = Scalar(), int padding_mode = BORDER_CONSTANT);

        CV_WRAP ~Pad() override = default;

        /** @brief Apply augmentation method on source image. This operation is inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Vec4i padding;
        const Scalar fill;
        int padding_mode;
    };

    //! Crop a random portion of image and resize it to a given size.
    class CV_EXPORTS_W RandomResizedCrop : public Transform {
    public:
        /** @brief Initialize the RandomResizedCrop class.
         *
         * @param size Expected output size of the destination image.
         * @param scale Specify the the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
         * @param ratio lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
         * @param interpolation Interpolation mode. Refer to #InterpolationFlags for more details.
         */
        CV_WRAP explicit RandomResizedCrop(const Size& size, const Vec2d& scale = Vec2d(0.08, 1.0), const Vec2d& ratio = Vec2d(3.0 / 4.0, 4.0 / 3.0), int interpolation = INTER_LINEAR);

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Size size;
        Vec2d scale;
        Vec2d ratio;
        int interpolation;
    };

    //! Change the brightness, contrast, saturation and hue of the given image randomly. The activated functions are applied in random order.
    class CV_EXPORTS_W ColorJitter : public Transform {
    public:
        /** Initialize the ColorJitter class.
         *
         * @param brightness Specify the lower and upper bounds for the brightness factor.
         * Brightness factor is >= 0. When brightness factor is 1, the brightness of the augmented image will not be changed.
         * When brightness factor is larger, the augmented image is brighter.
         * By default this function is disabled.
         * You can also pass cv::Vec2d() to disable this function manually.
         * @param contrast Specify the lower and upper bounds for the contrast factor.
         * Contrast factor is >= 0. When contrast factor is 1, the contrast of the augmented image will not be changed.
         * When contrast factor is larger, the contrast of the destination image is larger.
         * By default this function is disabled. You can also pass cv::Vec2d() to disable this function manually.
         * @param saturation Specify the lower and upper bounds for the saturation factor.
         * Saturation factor is >= 0. When saturation factor is 1, the saturation of the augmented image will not be changed.
         * When saturation factor is larger, the saturation of the destination image is larger.
         * By default this function is disabled. You can also pass cv::Vec2d() to disable this function manually.
         * @param hue Specify the lower and upper bounds for the hue factor.
         * Hue factor should be in range of -1 to 1. When hue factor is 0, the hue of the augmented image will not be changed.
         * By default this function is disabled. You can also pass cv::Vec2d() to disable this function manually.
         */
        CV_WRAP explicit ColorJitter(const Vec2d& brightness=Vec2d(), const Vec2d& contrast=Vec2d(), const Vec2d& saturation=Vec2d(), const Vec2d& hue=Vec2d());

        /** Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Vec2d brightness;
        Vec2d contrast;
        Vec2d saturation;
        Vec2d hue;
    };

    //! Rotate the given image by a random degree.
    class CV_EXPORTS_W RandomRotation : public Transform {
    public:
        /** @brief Initialize the RandomRotation class.
         *
         * @param degrees Specify the lower and upper bounds for the rotation degree.
         * @param interpolation Interpolation mode. Refer to #InterpolationFlags for more details.
         * @param center Rotation center, origin is the left corner of the image. By default it is set to the center of the image.
         * @param fill Fill value for the area outside the rotated image. Default is 0 for all channels.
         */
        CV_WRAP explicit RandomRotation(const Vec2d& degrees, int interpolation=INTER_LINEAR, const Point2f& center=Point2f(), const Scalar& fill=Scalar());

        /** Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Vec2d degrees;
        int interpolation;
        Point2f center;
        Scalar fill;
    };

    //! Convert the image into grayscale image of specified channels.
    class CV_EXPORTS_W GrayScale : public Transform {
    public:
        /** @brief Initialize the GrayScale class.
         *
         * @param num_channels number of the channels of the destination image. All channels are same.
         */
        CV_WRAP explicit GrayScale(int num_channels=1);

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        int num_channels;
    };

    //! Convert the given image into grayscale given a certain probability.
    class CV_EXPORTS_W RandomGrayScale : public Transform {
    public:
        /** @brief Initialize the RandomGrayScale class.
         *
         * @param p Probability of turning a image into grayscale. p should be in range 0 to 1. A larger p means a higher probability.
         */
        CV_WRAP explicit RandomGrayScale(double p=0.1);

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        double p;
    };

    //! Randomly erase a area of the given image.
    class CV_EXPORTS_W RandomErasing : public Transform {
    public:
        /** Initialize the RandomErasing class.
         *
         * @param p Probability to apply the random erasing operation.
         * @param scale Range of proportion of erased area against input image.
         * @param ratio Range of aspect ratio of erased area.
         * @param value Fill value of the erased area.
         * @param inplace If true, erase the area on the source image.
         * If false, erase the area on the destination image, which will not affect the source image.
         */
        CV_WRAP explicit RandomErasing(double p=0.5, const Vec2d& scale=Vec2d(0.02, 0.33), const Vec2d& ratio=Vec2d(0.3, 0.33), const Scalar& value=Scalar(0, 100, 100), bool inplace=false);

        /** @brief Apply augmentation method on source image.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        double p;
        Vec2d scale;
        Vec2d ratio;
        Scalar value;
        bool inplace;
    };


    //! Normalize given image with mean and standard deviation.
    //! The destination image will be normalized into range 0 to 1 first,
    //! then the normalization operation will be applied to each channel of the image.
    class CV_EXPORTS_W Normalize : public Transform {
    public:
        /** @brief Initialize the Normalize class.
         *
         * @param mean Sequence of means for each channels.
         * @param std Sequence of standard deviations for each channels.
         *
         * @note The image read in OpenCV is of type BGR by default, you should provide the mean and std in order of [B,G,R] if the type of source image is BGR.
         */
        CV_WRAP explicit Normalize(const Scalar& mean=Scalar(0,0,0,0), const Scalar& std=Scalar(1,1,1,1));

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Scalar mean;
        Scalar std;
    };

    //! Blurs image with randomly chosen Gaussian blur.
    class CV_EXPORTS_W GaussianBlur : public Transform {
    public:
        /** @brief Initialize the GaussianBlur class.
         *
         * @param kernel_size Size of the gaussian kernel.
         * @param sigma Specify the lower and upper bounds of the standard deviation to be used for creating kernel to perform blurring.
         */
        CV_WRAP explicit GaussianBlur(const Size& kernel_size, const Vec2f& sigma=Vec2f(0.1, 2.0));

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Size kernel_size;
        Vec2f sigma;
    };

    //! Apply random affine transformation to the image.
    class CV_EXPORTS_W RandomAffine: public Transform{
    public:
        /** Initialize the RandomAffine class.
         *
         * @param degrees Range of rotation degrees to select from.
         * @param translations Tuple of maximum absolute fraction for horizontal and vertical translations. By default translation is 0 in both directions.
         * @param scales Scaling factor interval. The scale factor is sampled uniformly from the interval. By default scale factor is 1.
         * @param shears Range of degrees to select from. Degree along x axis shear_x is sampled from range [shears[0], shear[1]]. Degree along y axis shear_y is sampled from range [shears[2], shear[3]]. By default, shear_x and shear_y are all 0.
         * @param interpolation Interpolation mode. Refer to #InterpolationFlags for more details.
         * @param fill Fill value of the area outside the transformed image.
         * @param center Rotation center. Origin is the left corner of the image. By default it is set to the center of the image.
         */
        CV_WRAP explicit RandomAffine(const Vec2f& degrees=Vec2f(0., 0.), const Vec2f& translations=Vec2f(0., 0.), const Vec2f& scales=Vec2f(1., 1.), const Vec4f& shears=Vec4f(0., 0., 0., 0.), int interpolation=INTER_NEAREST, const Scalar& fill=Scalar(), const Point2i& center=Point2i(-1, -1));

        /** @brief Apply augmentation method on source image. This operation is not inplace.
         *
         * @param src Source image.
         * @param dst Destination image.
         */
        CV_WRAP void call(InputArray src, OutputArray dst) const override;

        Vec2f degrees;
        Vec2f translations;
        Vec2f scales;
        Vec4f shears;
        int interpolation;
        Scalar fill;
        Point2i center;

    };

    //! @cond IGNORED
    void grayScale(InputArray _src, OutputArray _dst, int num_channels);
    void randomCrop(InputArray src, OutputArray dst, const Size& sz, const Vec4i& padding=Vec4i() , bool pad_if_need=false, int fill=0, int padding_mode=BORDER_CONSTANT);CV_EXPORTS_W void randomFlip(InputArray src, OutputArray dst, int flipCode=0, double p=0.5);
    void centerCrop(InputArray src, OutputArray dst, const Size& size);
    void randomResizedCrop(InputArray src, OutputArray dst, const Size& size, const Vec2d& scale = Vec2d(0.08, 1.0), const Vec2d& ratio = Vec2d(3.0 / 4.0, 4.0 / 3.0), int interpolation = INTER_LINEAR);
    void colorJitter(InputArray src, OutputArray dst, const Vec2d& brightness=Vec2d(), const Vec2d& contrast=Vec2d(), const Vec2d& saturation=Vec2d(), const Vec2d& hue=Vec2d());
    void randomRotation(InputArray src, OutputArray dst, const Vec2d& degrees, int interpolation=INTER_LINEAR, const Point2f& center=Point2f(), const Scalar& fill=Scalar(0));
    void randomGrayScale(InputArray src, OutputArray dst, double p=0.1);
    void randomErasing(InputArray src, OutputArray dst, double p=0.5, const Vec2d& scale=Vec2d(0.02, 0.33), const Vec2d& ratio=Vec2d(0.3, 0.33), const Scalar& value=Scalar(0, 100, 100), bool inplace=false);
    void gaussianBlur(InputArray src, OutputArray dst, const Size& kernel_size, const Vec2f& sigma=Vec2f(0.1, 2.0));
    void randomAffine(InputArray src, OutputArray dst, const Vec2f& degrees=Vec2f(0., 0.), const Vec2f& translations=Vec2f(0., 0.), const Vec2f& scales=Vec2f(1., 1.), const Vec4f& shears=Vec4f(0., 0., 0., 0.), int interpolation=INTER_NEAREST, const Scalar& fill=Scalar(), const Point2i& center=Point2i(-1, -1));
    //! @endcond

    //! @}

    }
}

#endif
