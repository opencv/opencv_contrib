#include <opencv2/structured_light/slmono.hpp>
#include "opencv2/structured_light/slmono_utils.hpp"


namespace cv {
namespace structured_light {

//main phase unwrapping function
void StructuredLightMono::unwrapPhase(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs, OutputArray out)
{
    if (alg_type == "PCG")
    {
        computePhasePCG(refs, imgs, out);
    }
    else if (alg_type == "TPU")
    {
        computePhaseTPU(refs, imgs, out);
    }
}

//algorithm for shadow removing from images
void StructuredLightMono::removeShadows(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs)
{
    vector<Mat>& refs_ = *(vector<Mat>*)refs.getObj();
    vector<Mat>& imgs_ = *(vector<Mat>*)imgs.getObj();
    Size size = refs_[0].size();
    Mat mean(size, CV_32FC1);
    for( int i = 0; i < size.height; ++i )
    {
        for( int j = 0; j < size.width; ++j )
        {
            float average = 0;
            for (int k = 0; k < imgs_.size(); k++)
            {
                average += (float) imgs_[k].at<float>(i, j);
            }
            mean.at<float>(i, j) = average/imgs_.size();
        }
    }
    mean.convertTo(mean, CV_32FC1);
    Mat shadowMask;
    threshold(mean, shadowMask, 0.05, 1, 0);
    for (int k = 0; k < imgs_.size(); k++)
    {
        multiply(shadowMask, refs_[k], refs_[k]);
        multiply(shadowMask, imgs_[k], imgs_[k]);
    }
}

//generate patterns for projection
//TPU algorithm requires low and high frequency patterns

void StructuredLightMono::generatePatterns(OutputArrayOfArrays patterns, float stripes_angle)
{
    vector<Mat>& patterns_ = *(vector<Mat>*) patterns.getObj();
    float phi = (float)projector_size.width/(float)stripes_num;
    float delta = 2*(float)CV_PI/phi;
    float shift = 2*(float)CV_PI/pattern_num;
    Mat pattern(projector_size, CV_32FC1, Scalar(0));
    for(int k = 0; k < pattern_num; k++)
    {
        for(uint i = 0; i < projector_size.height; ++i )
        {
            for(uint j = 0; j < projector_size.width; ++j )
            {
                pattern.at<float>(i, j) = (cos((stripes_angle*i+(1-stripes_angle)*j)*delta+k*shift) + 1)/2;
            }
        }
        Mat temp = pattern.clone();
        patterns_.push_back(temp);
    }
    if (alg_type == "TPU")
    {
        phi = (float)projector_size.width;
        delta = 2*(float)CV_PI/phi;
        for(int k = 0; k < pattern_num; k++)
        {
            for(uint i = 0; i < projector_size.height; ++i )
            {
                for(uint j = 0; j < projector_size.width; ++j )
                {
                    pattern.at<float>(i, j) = (cos((stripes_angle*i+(1-stripes_angle)*j)*delta+k*shift) + 1)/2;
                }
            }
            Mat temp = pattern.clone();
            patterns_.push_back(temp);
        }
    }
}

//phase computation based on PCG algorithm
void StructuredLightMono::computePhasePCG(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs, OutputArray out){

    vector<Mat>& refs_ = *(vector<Mat>* ) refs.getObj();
    Size size = refs_[0].size();
    Mat wrapped = Mat(size, CV_32FC1);
    Mat wrapped_ref = Mat(size, CV_32FC1);
    removeShadows(refs, imgs);
    computeAtanDiff(refs, wrapped_ref);
    computeAtanDiff(imgs, wrapped);
    subtract(wrapped, wrapped_ref, wrapped);
    unwrapPCG(wrapped, out, size);
}

//phase computation based on TPU algorithm
void StructuredLightMono::computePhaseTPU(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs, OutputArray out)
{
    vector<Mat>& refs_ = *(vector<Mat>* ) refs.getObj();
    vector<Mat>& imgs_ = *(vector<Mat>* ) imgs.getObj();
    Size size = refs_[0].size();
    removeShadows(refs, imgs);
    int split = (int)(refs_.size()/2);
    auto hf_refs = vector<Mat>(refs_.begin(), refs_.begin()+split);
    auto lf_refs = vector<Mat>(refs_.begin()+split, refs_.end());
    auto hf_phases = vector<Mat>(imgs_.begin(), imgs_.begin()+split);
    auto lf_phases = vector<Mat>(imgs_.begin()+split, imgs_.end());
    Mat _lf_ref_phase = Mat(size, CV_32FC1);
    Mat _hf_ref_phase= Mat(size, CV_32FC1);
    Mat _lf_phase = Mat(size, CV_32FC1);
    Mat _hf_phase = Mat(size, CV_32FC1);
    computeAtanDiff(lf_refs, _lf_ref_phase);
    computeAtanDiff(hf_refs, _hf_ref_phase);
    computeAtanDiff(lf_phases, _lf_phase);
    computeAtanDiff(hf_phases, _hf_phase);
    subtract(_lf_phase, _lf_ref_phase, _lf_phase);
    subtract(_hf_phase, _hf_ref_phase, _hf_phase);
    unwrapTPU(_lf_phase, _hf_phase, out, stripes_num);
}

}
}
