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

            //! Base class for all data augmentation classes for detection task
            class CV_EXPORTS_W Transform{
            public:
                CV_WRAP virtual void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, CV_IN_OUT std::vector<int>& labels) const = 0;
                CV_WRAP virtual ~Transform() = default;
            };

            //! Combine data augmentation methods into one and apply them sequentially to source image and annotation
            //! All combined data augmentation class must inherited from cv::imgaug::det::Transform
            class CV_EXPORTS_W Compose : public Transform{
            public:
                /** @brief Initialize Compose class.
                 *
                 * @param transforms data augmentation methods used to compose
                 */
                CV_WRAP explicit Compose(std::vector<cv::Ptr<cv::imgaug::det::Transform> >& transforms);

                /** @brief Apply data augmentation method on source image and its annotation.
                 *
                 * @param src Source image.
                 * @param dst Destination image.
                 * @param bboxes Annotation of source image, which consists of several bounding boxes of the detected objects in the source image.
                 * In Python, the bounding box is represented as a four-elements tuple (x, y, w, h),
                 * in which x, y is the coordinates of the left top corner of the bounding box and w, h is the width and height of the bounding box.
                 * @param labels Class labels of the detected objects in source image. The order of the labels should correspond to the order of the bboxes.
                 */
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, CV_IN_OUT std::vector<int>& labels) const override;

                std::vector<cv::Ptr<cv::imgaug::det::Transform> > transforms;
            };

            class CV_EXPORTS_W RandomFlip: public Transform{
            public:
                /** @brief Initialize the RandomFlip class.
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
                CV_WRAP explicit RandomFlip(int flipCode=0, float p=0.5);

                /** @brief Apply data augmentation method on source image and its annotation.
                 *
                 * @param src Source image.
                 * @param dst Destination image.
                 * @param bboxes Annotation of source image, which consists of several bounding boxes of the detected objects in the source image.
                 * In Python, the bounding box is represented as a four-elements tuple (x, y, w, h),
                 * in which x, y is the coordinates of the left top corner of the bounding box and w, h is the width and height of the bounding box.
                 * @param labels Class labels of the detected objects in source image. The order of the labels should correspond to the order of the bboxes.
                 */
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, std::vector<int>& labels) const override;

                /** @brief Flip the annotated bounding boxes.
                 *
                 * @param bboxes Bounding box annotations.
                 * @param size The size of the source image.
                 */
                void flipBoundingBox(std::vector<cv::Rect>& bboxes, const Size& size) const;

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


            //! Resize the source image and its annotations into specified size.
            class CV_EXPORTS_W Resize: public Transform{
            public:
                /** @brief Initialize the Resize class
                 *
                 * @param size Size of the resized image.
                 * @param interpolation Interpolation mode when resize image, see #InterpolationFlags for details.
                 */
                CV_WRAP explicit Resize(const Size& size, int interpolation=INTER_NEAREST);

                /** @brief Apply data augmentation method on source image and its annotation.
                 *
                 * @param src Source image.
                 * @param dst Destination image.
                 * @param bboxes Annotation of source image, which consists of several bounding boxes of the detected objects in the source image.
                 * In Python, the bounding box is represented as a four-elements tuple (x, y, w, h),
                 * in which x, y is the coordinates of the left top corner of the bounding box and w, h is the width and height of the bounding box.
                 * @param labels Class labels of the detected objects in source image. The order of the labels should correspond to the order of the bboxes.
                 */
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, std::vector<int>& labels) const override;

                /** @brief Resize the bounding boxes of the detected objects in the source image.
                 *
                 * @param bboxes Bounding box annotations.
                 * @param imgSize The size of the source image.
                 */
                void resizeBoundingBox(std::vector<cv::Rect>& bboxes, const Size& imgSize) const;

                const Size size;
                int interpolation;
            };

            //! Convert the color space of the given image
            class CV_EXPORTS_W Convert: public Transform{
            public:
                /** @brief Initialize the Convert class
                 *
                 * @param code color space conversion code (see #ColorConversionCodes).
                 */
                CV_WRAP explicit Convert(int code);

                /** @brief Apply data augmentation method on source image and its annotation.
                 *
                 * @param src Source image.
                 * @param dst Destination image.
                 * @param bboxes Annotation of source image, which consists of several bounding boxes of the detected objects in the source image.
                 * In Python, the bounding box is represented as a four-elements tuple (x, y, w, h),
                 * in which x, y is the coordinates of the left top corner of the bounding box and w, h is the width and height of the bounding box.
                 * @param labels Class labels of the detected objects in source image. The order of the labels should correspond to the order of the bboxes.
                 */
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, std::vector<int>& labels) const override;

                int code;
            };

            //! Randomly translate the given image.
            //! Bounding boxes which has an area of less than the threshold in the remaining in the transformed image
            //! will be filtered.
            //! The resolution of the image is not changed after the transformation. The remaining area after shift is filled with 0.
            class CV_EXPORTS_W RandomTranslation: public Transform{
            public:
                /** @brief Initialize the RandomTranslation class
                 *
                 * @param translations Contains two elements tx and ty, representing tha maximum translation distances
                 * along x axis and y axis in pixels. tx and ty must be >= 0. The actual translation distances along x and y axes
                 * are sampled uniformly from [-tx, tx] and [-ty, ty].
                 * @param threshold Bounding boxes with area in the remaining image less than threshold will be dropped.
                 */
                CV_WRAP explicit RandomTranslation(const Vec2i& translations, float threshold=0.25);

                /** @brief Apply data augmentation method on source image and its annotation.
                 *
                 * @param src Source image.
                 * @param dst Destination image.
                 * @param bboxes Annotation of source image, which consists of several bounding boxes of the detected objects in the source image.
                 * In Python, the bounding box is represented as a four-elements tuple (x, y, w, h),
                 * in which x, y is the coordinates of the left top corner of the bounding box and w, h is the width and height of the bounding box.
                 * @param labels Class labels of the detected objects in source image. The order of the labels should correspond to the order of the bboxes.
                 */
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, std::vector<int>& labels) const override;

                /** @brief Translate bounding boxes and filter invalid bounding boxes after translation.
                 *
                 * @param bboxes Bounding box annotations.
                 * @param labels Class labels of the detected objects in source image.
                 * @param imgSize Size of the source image.
                 * @param tx Translation in x axis in pixel.
                 * @param ty Translation in y axis in pixel.
                 */
                CV_WRAP void translateBoundingBox(std::vector<cv::Rect>& bboxes, std::vector<int> &labels, const Size& imgSize, int tx, int ty) const;

                Vec2i translations;
                float threshold;
            };

            //! Rotate the given image and its bounding boxes by a random angle.
            //! Filter invalid bounding boxes if its remaining area in the destination image is less than threshold.
            //! The size of the destination image is not changed. The remaining area in the destination image is filled with 0.
            class CV_EXPORTS_W RandomRotation: public Transform{
            public:
                /** @brief Initialize the RandomRotation class.
                 *
                 * @param angles Intervals in which the rotation angle is uniformly sampled from.
                 * @param threshold Bounding boxes with area in the remaining image less than threshold will be dropped.
                 */
                explicit RandomRotation(const Vec2d& angles, double threshold=0.25);

                /** @brief Apply data augmentation method on source image and its annotation.
                 *
                 * @param src Source image.
                 * @param dst Destination image.
                 * @param bboxes Annotation of source image, which consists of several bounding boxes of the detected objects in the source image.
                 * In Python, the bounding box is represented as a four-elements tuple (x, y, w, h),
                 * in which x, y is the coordinates of the left top corner of the bounding box and w, h is the width and height of the bounding box.
                 * @param labels Class labels of the detected objects in source image. The order of the labels should correspond to the order of the bboxes.
                 */
                CV_WRAP void call(InputArray src, OutputArray dst, CV_IN_OUT std::vector<cv::Rect>& bboxes, std::vector<int>& labels) const override;

                /** @brief Rotate bounding boxes and filter out invalid bounding boxes after rotation.
                 *
                 * @param bboxes Bounding box annotations.
                 * @param labels Class labels of the detected objects in source image.
                 * @param angle Rotation angle in degree.
                 * @param cx x coordinate of the rotation center.
                 * @param cy y coordinate of the rotation center.
                 * @param imgSize Size of the destination image, used for clamping the coordinates of bounding boxes.
                 */
                CV_WRAP void rotateBoundingBoxes(std::vector<cv::Rect>& bboxes, std::vector<int> &labels, double angle, int cx, int cy, const Size& imgSize) const;

                Vec2d angles;
                double threshold;
            };

            //! @}
        }
    }
}

#endif //OPENCV_TRANSFORMS_DET_HPP
