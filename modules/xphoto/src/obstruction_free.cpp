// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.



#ifndef __OPENCV_OBSTRUCTION_FREE_CPP__
#define __OPENCV_OBSTRUCTION_FREE_CPP__

#include <vector>
#include <iostream>

#include <opencv2/xphoto.hpp>

namespace cv
{
namespace xphoto
{


//constructor
obstructionFree::obstructionFree()
{
    //parameter
    pyramidLevel = 3; // Pyramid level
    coarseIterations = 4; // iteration number for the coarsest level
    upperIterations = 1; // iteration number for the upper level
    fixedPointIterations = 5; // during each level of the pyramid
    sorIterations = 25; // iterations of SOR
    omega = 1.6f; // relaxation factor in SOR
    lambda1 = 1.0f; // weight for alpha map smoothness constraints
    lambda2 = 0.1f; // weight for image smoothness constraints
    lambda3 = 3000.0f; // weight for independence between back/foreground component
    lambda4 = 0.5f; // weight for gradient sparsity
}

obstructionFree::obstructionFree(const std::vector <Mat> &srcImgs)
{
    //parameter
    frameNumber = srcImgs.size();
    referenceNumber=(frameNumber-1)/2; //target frame
    pyramidLevel = 3; // Pyramid level
    coarseIterations = 4; // iteration number for the coarsest level
    upperIterations = 1; // iteration number for the upper level
    fixedPointIterations = 5; // during each level of the pyramid
    sorIterations = 25; // iterations of SOR
    omega = 1.6f; // relaxation factor in SOR
    lambda1 = 1.0f; // weight for alpha map smoothness constraints
    lambda2 = 0.1f; // weight for image smoothness constraints
    lambda3 = 3000.0f; // weight for independence between back/foreground component
    lambda4 = 0.5f; // weight for gradient sparsity
}

//build pyramid by stacking all input image sequences
std::vector<Mat> obstructionFree::buildPyramid(const std::vector <Mat>& srcImgs)
{
    std::vector<Mat> pyramid;

    for (size_t frame_i=0; frame_i<this->frameNumber; frame_i++){
        Mat grey;
        cvtColor(srcImgs[frame_i], grey, COLOR_RGB2GRAY);
        pyramid.push_back(grey.clone());
    }

    Mat thisLevel, nextLevel;
    size_t thisLevelId;
    for (int level_i=1; level_i<this->pyramidLevel; level_i++){
        for (size_t frame_i=0; frame_i<this->frameNumber; frame_i++){
            thisLevelId=(level_i-1)*this->frameNumber+frame_i;
            //nextLevelId=i*this.pyramidLevel+frame_i;
            thisLevel=pyramid[thisLevelId];
            pyrDown(thisLevel, nextLevel);
            pyramid.push_back(nextLevel.clone());
        }
    }
    return pyramid;
}

//extract certain level of image sequences from the stacked image pyramid
std::vector<Mat> obstructionFree::extractLevelImgs(const std::vector<Mat>& pyramid, const int level){
    std::vector<Mat> levelPyramid;
    size_t imgId;
    for (size_t frame_i=0; frame_i<this->frameNumber; frame_i++){
        imgId=level*this->frameNumber+frame_i;
        levelPyramid.push_back(pyramid[imgId].clone());
    }
    return levelPyramid;
}

Mat obstructionFree::indexToMask(const Mat& indexMat, const int rows, const int cols){
    Mat maskMat=Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < indexMat.cols; i++ ) {
        for (int j = 0; j < indexMat.rows; j++) {
            Vec2i mask_loca = indexMat.at<Vec2i>(j, i);
            if (mask_loca[0] !=0 && mask_loca[1] !=0) {
                maskMat.at<uchar>(Point(mask_loca)) = 255;}
    }}
    return  maskMat;
}

//estimate homography matrix using edge flow fields
Mat obstructionFree::flowHomography(const Mat& edges, const Mat& flow, const int ransacThre){
    Mat inlierMask, inlier_edges, inilier_edgeLocations;
    std::vector<Point> edge_Locations;

    findNonZero(edges, edge_Locations);

    std::vector<Point> obj_edgeflow;

    for(size_t i = 0; i<edge_Locations.size();i++){
        int src_x=edge_Locations[i].x;
        int src_y=edge_Locations[i].y;
        Point2f f = flow.at<Point2f>(src_y, src_x);
        obj_edgeflow.push_back(Point2f(src_x + f.x, src_y + f.y));
    }

    Mat Homography = findHomography( edge_Locations, obj_edgeflow, RANSAC, ransacThre, inlierMask);

    Mat(edge_Locations).copyTo(inilier_edgeLocations,inlierMask);

    //convert index matrix to mask matrix
    inlier_edges=indexToMask(inilier_edgeLocations, edges.rows, edges.cols);

    return inlier_edges;
}

Mat obstructionFree::sparseToDense(const Mat& im1, const Mat& im2, const Mat& im1_edges, const Mat& sparseFlow){
    Mat denseFlow;
    std::vector<Point2f> sparseFrom;
    std::vector<Point2f> sparseTo;

    std::vector<Point> edge_Location;
    findNonZero(im1_edges, edge_Location);
    for(size_t i = 0; i<edge_Location.size();i++){
        int src_x=edge_Location[i].x;
        int src_y=edge_Location[i].y;
        sparseFrom.push_back(Point2f(float(src_x), float(src_y)));
        Point2f f = sparseFlow.at<Point2f>(src_y, src_x);
        sparseTo.push_back(Point2f(float(src_x + f.x), float(src_y + f.y)));
    }

    Ptr<cv::ximgproc::SparseMatchInterpolator> epicInterpolation=ximgproc::createEdgeAwareInterpolator();
    epicInterpolation->interpolate(im1,sparseFrom,im2,sparseTo,denseFlow);
    return denseFlow;
}

//show motion fields using color
void obstructionFree::colorFlow(const Mat& flow, std::string figName="optical flow")
{
    //extraxt x and y channels
    Mat xy[2]; //X,Y
    split(flow, xy);

    //calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    //convert to BGR and show
    Mat bgr;//CV_32FC3 matrix
    cvtColor(hsv, bgr, COLOR_HSV2BGR);
    imshow(figName, bgr);
}

//motion decomposition between two images: target frame and source frame
void obstructionFree::initMotionDecompose(const Mat& im1, const Mat& im2, Mat& back_denseFlow, Mat& fore_denseFlow, int back_ransacThre=1, int fore_ransacThre=1){
    if (im1.channels()!= 1)
        cvtColor(im1, im1, COLOR_RGB2GRAY);
    if (im2.channels()!= 1)
        cvtColor(im2, im2, COLOR_RGB2GRAY);

    Mat im1_edge, im2_edge;
    Mat flow;
    Mat edgeflow;  //extracted edgeflow

    //Mat backH, mask_backH;
    Mat back_edges, rest_edges, fore_edges; //edges aligned to the back layer using homography, remaining layer, foreground layers
    Mat back_flow, rest_flow, fore_flow;


    Canny(im1, im1_edge, 10, 100,3,true);
    Canny(im2, im2_edge, 10, 100,3,true);

    ///////////////replace edgeflow
    Ptr<DenseOpticalFlow> deepflow = optflow::createOptFlow_DeepFlow();
    deepflow->calc(im1, im2, flow);
    //colorFlow(flow,"optical_flow");
    flow.copyTo(edgeflow, im1_edge);
    //colorFlow(edgeflow,"edge_flow");

////////flow=>points using homography-ransac filtering, and then extract flow on the filtered edges
    back_edges=flowHomography(im1_edge, edgeflow, back_ransacThre);
    //imshow("back_edges", back_edges);
    edgeflow.copyTo(back_flow,back_edges);
    //colorFlow(back_flow, "back_flow");
    //////////rest edges and flows
    rest_edges=im1_edge-back_edges;
    //imshow("rest_edges", rest_edges);
    rest_flow=edgeflow-back_flow;
   //colorFlow(rest_flow, "rest_flow");

    ////////////align resting flows to another homograghy
    fore_edges=flowHomography(rest_edges, rest_flow, fore_ransacThre);
    //imshow("fore_edges", fore_edges);
    rest_flow.copyTo(fore_flow,fore_edges);
    //colorFlow(fore_flow, "fore_flow");

///////////////////interpolation from sparse edgeFlow to denseFlow/////////////////////
    back_denseFlow=sparseToDense(im1, im2, back_edges, back_flow);
    fore_denseFlow=sparseToDense(im1, im2, fore_edges, fore_flow);
    //colorFlow(back_denseFlow,"inter_back_denseflow");
    //colorFlow(fore_denseFlow,"inter_fore_denseflow");
}

//warping im1 to output through optical flow:
//flow=flow->cal(im1,im2), so warp im2 to back
Mat obstructionFree::imgWarpFlow(const Mat& im1, const Mat& flow){
    Mat flowmap_x(flow.size(), CV_32FC1);
    Mat flowmap_y(flow.size(), CV_32FC1);
    for (int j = 0; j < flowmap_x.rows; j++){
        for (int i = 0; i < flowmap_x.cols; ++i){
            Point2f f = flow.at<Point2f>(j, i);
            flowmap_y.at<float>(j, i) = float(j + f.y);
            flowmap_x.at<float>(j, i) = float(i + f.x);
            }}
    Mat warpedFrame;
    remap(im1, warpedFrame, flowmap_x,flowmap_y ,INTER_CUBIC,BORDER_CONSTANT,255);
    return warpedFrame;
}

//Initialization: decompose the motion fields
void obstructionFree::motionInitDirect(const std::vector<Mat>& video_input, std::vector<Mat>& back_flowfields, std::vector<Mat>& fore_flowfields, std::vector<Mat>& warpedToReference){
    int back_ransacThre=1;
    int fore_ransacThre=1;

    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        Mat im1,im2;//reference frame, other frame
        //Mat foreH, mask_foreH;
        Mat back_denseFlow, fore_denseFlow;

        if (frame_i!=referenceNumber){
            //int frame_i=1;
            im1 = video_input[referenceNumber].clone();
            im2 = video_input[frame_i].clone();

            //decompose motion fields into fore/background
            initMotionDecompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);
            back_flowfields.push_back(back_denseFlow.clone());
            fore_flowfields.push_back(fore_denseFlow.clone());
            //colorFlow(back_denseFlow,"inter_back_denseflow");
            //colorFlow(fore_denseFlow,"inter_fore_denseflow");
//
    ////////////warping images to the reference frame///////////////////
            Mat warpedFrame=imgWarpFlow(im2, back_denseFlow);
            warpedToReference.push_back(warpedFrame.clone());
            //imshow("warped image",warpedFrame);
        }
        else{
            Mat refer_grey=video_input[referenceNumber].clone();
            warpedToReference.push_back(refer_grey.clone());
            back_flowfields.push_back(Mat::zeros(refer_grey.rows,refer_grey.cols,CV_32FC2));
            fore_flowfields.push_back(Mat::zeros(refer_grey.rows,refer_grey.cols,CV_32FC2));
        }
        }
    }

//initialization: decompose the image components in the case of reflection
Mat obstructionFree::imgInitDecomRef(const std::vector <Mat> &warpedImgs){
    Mat background;
    background=warpedImgs[referenceNumber];
    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        background=min(background,warpedImgs[frame_i]);
    }
    imshow("reflection initial background", background);
    return background;
    }

//initialization: decompose the image components in the case of opaque reflection
Mat obstructionFree::imgInitDecomOpaq(const std::vector <Mat> &warpedImgs, Mat& foreground, Mat& alphaMap){
    Mat background;
    Mat sum=Mat::zeros(warpedImgs[referenceNumber].rows,warpedImgs[referenceNumber].cols,CV_32F);
    Mat temp,background_temp;
    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        warpedImgs[frame_i].convertTo(temp,CV_32F);
        sum+=temp;
    }
    background_temp=sum/double(frameNumber);
    background_temp.convertTo(background,CV_8UC1);
    imshow("opaque initial background", background);

    Mat difference;
    difference=abs(background-warpedImgs[referenceNumber]);
    threshold(difference, alphaMap,25.5,255,THRESH_BINARY_INV);
    imshow("alpha map",alphaMap);

    foreground=warpedImgs[referenceNumber]-background;
    imshow("foreground",foreground);

    return background;
}


//core function
void obstructionFree::removeOcc(const std::vector <Mat> &srcImgs, Mat &dst, Mat& foreground, Mat &mask, const int obstructionType){
    //initialization
    std::vector<Mat> warpedToReference;
    std::vector<Mat> srcPyramid=buildPyramid(srcImgs);
    std::vector<Mat> coarseLevel = extractLevelImgs(srcPyramid,pyramidLevel-1);
    motionInitDirect(coarseLevel, backFlowFields, foreFlowFields, warpedToReference);

    CV_Assert(obstructionType==0 || obstructionType==1);
    if (obstructionType==0)
        dst=imgInitDecomRef(warpedToReference);
    else
        dst=imgInitDecomOpaq(warpedToReference,foreground,mask);

    //alternative optimization:TODO


}




}
}
#endif // __OPENCV_OBSTRUCTION_FREE_CPP__
