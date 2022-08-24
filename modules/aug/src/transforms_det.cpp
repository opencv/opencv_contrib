#include "precomp.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>


namespace cv{
    namespace imgaug{
        extern RNG rng;

        namespace det{
            Compose::Compose(std::vector<Ptr<Transform> >& transforms):
                    transforms(transforms){};

            void Compose::call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target) const{
                for(cv::imgaug::det::Transform* transform:transforms){
                    transform->call(src, dst, target);
                }
            }

            RandomFlip::RandomFlip(int flipCode, float p):
                    flipCode(flipCode), p(p)
            {
                if(p < 0 || p > 1){
                    CV_Error(Error::Code::StsBadArg, "probability p must be between range 0 and 1");
                }
            };

            void RandomFlip::call(InputArray _src, OutputArray _dst, std::vector<cv::Rect>& target) const{
//            RNG rng = RNG(getTickCount());
                bool flag = rng.uniform(0., 1.) < p;

                Mat src = _src.getMat();
                if(!flag){
                    _dst.move(src);
                    return;
                }

                flipBoundingBox(target, src.size());
                flip(src, src, flipCode);
                _dst.move(src);
            }

            void RandomFlip::flipBoundingBox(std::vector<cv::Rect>& target, const Size& size) const{
                /*
                 * flipCode = 0 (flip vertically): (x', y') = (x, img.height - y - bbox.height)
                 * flipCode > 0 (flip horizontally): (x', y') = (img.width - x - bbox.width, y)
                 * flipCode < 0 (flip diagonally): (x', y') = (img.width - x - bbox.width, img.height - y - bbox.height)
                 */
                for(int i = 0; i < target.size(); i++){
                    if(flipCode == 0){
                        target[i].y = size.height - target[i].y - target[i].height;
                    }else if(flipCode > 0){
                        target[i].x = size.width - target[i].x - target[i].width;
                    }else{
                        target[i].x = size.width - target[i].x - target[i].width;
                        target[i].y = size.height - target[i].y - target[i].height;
                    }
                }
            }

//        RandomCrop::RandomCrop(const Size& sz, const Vec4i& padding, bool pad_if_need, const Scalar& fill, int padding_mode):
//        sz(sz), padding(padding), pad_if_need(pad_if_need), fill(fill), padding_mode(padding_mode){};

            Resize::Resize(const Size& size, int interpolation):
                    size(size), interpolation(interpolation){};

            void Resize::call(InputArray _src, OutputArray dst, std::vector<cv::Rect>& target) const{
                Mat src = _src.getMat();
                resize(src, dst, size, 0, 0, interpolation);
                resizeBoundingBox(target, src.size());
            }

            void Resize::resizeBoundingBox(std::vector<cv::Rect>& target, const Size& imgSize) const{
                for(unsigned i=0; i<target.size(); i++){
                    target[i].x = static_cast<double>(size.width) / imgSize.width * target[i].x;
                    target[i].y = static_cast<double>(size.height) / imgSize.height * target[i].y;
                    target[i].width = static_cast<double>(size.width) / imgSize.width * target[i].width;
                    target[i].height = static_cast<double>(size.height) / imgSize.height * target[i].height;
                }
            }

            Convert::Convert(int code):
                    code(code){};

            void Convert::call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target) const{
                cvtColor(src, dst, code);
            }

        }
    }
}