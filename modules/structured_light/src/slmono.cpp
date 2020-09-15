#include <opencv2/structured_light/slmono.hpp>
#include "opencv2/structured_light/slmono_utils.hpp"


namespace cv {
namespace structured_light {

//read reference images and object images from specified files

//void StructuredLightMono::readImages(vector<string> refs_files, vector<string> imgs_files, OutputArrayOfArrays refs, OutputArrayOfArrays imgs)
//{
//    vector<Mat>& refs_ = *(vector<Mat>*) refs.getObj();
//    vector<Mat>& imgs_ = *(vector<Mat>*) imgs.getObj();
//
//    for(auto i = 0; i < refs_files.size(); i++)
//    {
//        auto img = imread(refs_files[i], IMREAD_COLOR);
//        cvtColor(img, img, COLOR_RGBA2GRAY);
//        img.convertTo(img, CV_32FC1, 1.f/255);
//        refs_.push_back(img);
//
//        img = imread(imgs_files[i], IMREAD_COLOR);
//        cvtColor(img, img, COLOR_RGBA2GRAY);
//        img.convertTo(img, CV_32FC1, 1.f/255);
//        imgs_.push_back(img);
//    }
//}


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

    float phi = projector_size.width/stripes_num;
    float delta = 2*CV_PI/phi;
    float shift = 2*CV_PI/pattern_num;

    Mat pattern(projector_size, CV_32FC1, Scalar(0));

    for(auto k = 0; k < pattern_num; k++)
    {
        for( int i = 0; i < projector_size.height; ++i )
        {
            for( int j = 0; j < projector_size.width; ++j )
            {
                pattern.at<float>(i, j) = (cos((stripes_angle*i+(1-stripes_angle)*j)*delta+k*shift) + 1)/2;
            }
        }
        Mat temp = pattern.clone();
        patterns_.push_back(temp);
    }

    if (alg_type == "TPU")
    {   
        phi = projector_size.width/1;
        delta = 2*CV_PI/phi;

        for(auto k = 0; k < pattern_num; k++)
        {
            for( int i = 0; i < projector_size.height; ++i )
            {
                for( int j = 0; j < projector_size.width; ++j )
                {
                    pattern.at<float>(i, j) = (cos((stripes_angle*i+(1-stripes_angle)*j)*delta+k*shift) + 1)/2;
                }
            }
            Mat temp = pattern.clone();
            patterns_.push_back(temp);
        }
    }
}


//capture reference and object images using camera and projector
//void StructuredLightMono::captureImages(InputArrayOfArrays patterns, OutputArrayOfArrays refs, OutputArrayOfArrays imgs, bool isCaptureRefs)
//{
//    vector<Mat>& patterns_ = *(vector<Mat>*)patterns.getObj();
//    vector<Mat>& refs_ = *(vector<Mat>*)refs.getObj();
//    vector<Mat>& imgs_ = *(vector<Mat>*)imgs.getObj();
//
//    VideoCapture cap;
//    if(cap.open(0))
//    {
//        Mat pause(projector_size, CV_64FC3, Scalar(0));
//        putText(pause, "Place the object", Point(projector_size.width/4, projector_size.height/4), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
//        putText(pause, "Press any key when ready", Point(projector_size.width/4, projector_size.height/4+projector_size.height/15), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
//        namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
//        setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
//        imshow("Display pattern", pause);
//        waitKey();
//
//        if (isCaptureRefs)
//        {
//            for(auto i = 0; i < patterns_.size(); i++)
//            {
//                Mat frame;
//                cap >> frame;
//                if(frame.empty()) break; // end of video stream
//
//                namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
//                setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
//                imshow("Display pattern", patterns_[i]);
//                waitKey();
//
//                Mat grayFrame;
//                cv::cvtColor(frame, grayFrame, COLOR_RGB2GRAY);
//                grayFrame.convertTo(grayFrame, CV_32FC1, 1.f/255);
//                refs_.push_back(grayFrame); //ADD ADDITIONAL SWITCH TO SELECT WHERE to SAVE
//
//            }
//        }
//
//        pause = Mat(projector_size, CV_64FC3, Scalar(0));
//        putText(pause, "Place the object", Point(projector_size.width/4, projector_size.height/4), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
//        putText(pause, "Press any key when ready", Point(projector_size.width/4, projector_size.height/4+projector_size.height/15), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
//        namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
//        setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
//        imshow( "Display pattern", pause);
//        waitKey();
//
//        for(auto i = 0; i < patterns_.size(); i++)
//        {
//            Mat frame;
//            cap >> frame;
//            if( frame.empty() ) break; // end of video stream
//
//            namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
//            setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
//            imshow( "Display pattern", patterns_[i]);
//            waitKey();
//
//            Mat grayFrame;
//            cv::cvtColor(frame, grayFrame, COLOR_RGB2GRAY);
//            grayFrame.convertTo(grayFrame, CV_32FC1, 1.f/255);
//            imgs_.push_back(grayFrame); //ADD ADDITIONAL SWITCH TO SELECT WHERE to SAVE
//
//        }
//
//        cap.release();
//    }
//}


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
    int split = refs_.size()/2;

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
