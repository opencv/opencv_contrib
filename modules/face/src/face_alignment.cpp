// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "face_alignmentimpl.hpp"
#include <vector>

using namespace std;

namespace cv{
namespace face{

FacemarkKazemi::~FacemarkKazemi(){}
FacemarkKazemiImpl:: ~FacemarkKazemiImpl(){}
unsigned long FacemarkKazemiImpl::left(unsigned long index){
    return 2*index+1;
}
unsigned long FacemarkKazemiImpl::right(unsigned long index){
    return 2*index+2;
}
bool FacemarkKazemiImpl::setFaceDetector(FN_FaceDetector f, void* userData){
    faceDetector = f;
    faceDetectorData = userData;
    //printf("face detector is configured\n");
    return true;
}
bool FacemarkKazemiImpl::getFaces(InputArray image, OutputArray faces)
{
    CV_Assert(faceDetector);
    return faceDetector(image, faces, faceDetectorData);
}
FacemarkKazemiImpl::FacemarkKazemiImpl(const FacemarkKazemi::Params& parameters) :
    faceDetector(NULL),
    faceDetectorData(NULL)
{
    minmeanx=8000.0;
    maxmeanx=0.0;
    minmeany=8000.0;
    maxmeany=0.0;
    isModelLoaded =false;
    params = parameters;
}
FacemarkKazemi::Params::Params(){
    //These variables are used for training data
    //These are initialised as described in the research paper
    //referenced above
    cascade_depth = 15;
    tree_depth = 5;
    num_trees_per_cascade_level = 500;
    learning_rate = float(0.1);
    oversampling_amount = 20;
    num_test_coordinates = 500;
    lambda = float(0.1);
    num_test_splits = 20;
}
bool FacemarkKazemiImpl::convertToActual(Rect r,Mat &warp){
    Point2f srcTri[3],dstTri[3];
    srcTri[0]=Point2f(0,0);
    srcTri[1]=Point2f(1,0);
    srcTri[2]=Point2f(0,1);
    dstTri[0]=Point2f((float)r.x,(float)r.y);
    dstTri[1]=Point2f((float)r.x+r.width,(float)r.y);
    dstTri[2]=Point2f((float)r.x,(float)r.y+(float)1.3*r.height);
    warp=getAffineTransform(srcTri,dstTri);
    return true;
}
bool FacemarkKazemiImpl::convertToUnit(Rect r,Mat &warp){
    Point2f srcTri[3],dstTri[3];
    dstTri[0]=Point2f(0,0);
    dstTri[1]=Point2f(1,0);
    dstTri[2]=Point2f(0,1);
    srcTri[0]=Point2f((float)r.x,(float)r.y);
    srcTri[1]=Point2f((float)r.x+r.width,(float)r.y);
    srcTri[2]=Point2f((float)r.x,(float)r.y+(float)1.3*r.height);
    warp=getAffineTransform(srcTri,dstTri);
    return true;
}
bool FacemarkKazemiImpl::setMeanExtreme(){
    if(meanshape.empty()){
        String error_message = "Model not loaded properly.No mean shape found.Aborting...";
        CV_Error(Error::StsBadArg, error_message);
    }
    for(size_t i=0;i<meanshape.size();i++){
        if(meanshape[i].x>maxmeanx)
            maxmeanx = meanshape[i].x;
        if(meanshape[i].x<minmeanx)
            minmeanx = meanshape[i].x;
        if(meanshape[i].y>maxmeany)
            maxmeany = meanshape[i].y;
        if(meanshape[i].y<minmeany)
            minmeany = meanshape[i].y;
    }
    return true;
}
bool FacemarkKazemiImpl::calcMeanShape (vector< vector<Point2f> >& trainlandmarks,vector<Mat>& trainimages,std::vector<Rect>& faces){
    //clear the loaded meanshape
    if(trainimages.empty()||trainlandmarks.size()!=trainimages.size()) {
        // throw error if no data (or simply return -1?)
        CV_Error(Error::StsBadArg, "Number of images is not equal to corresponding landmarks. Aborting...");
    }
    meanshape.clear();
    vector<Mat> finalimages;
    vector< vector<Point2f> > finallandmarks;
    float xmean[200] = {0.0};
    //array to store mean of y coordinates
    float ymean[200] = {0.0};
    size_t k=0;
    //loop to calculate mean
    Mat warp_mat,src,C,D;
    vector<Rect> facesp;
    Rect face;
    for(size_t i = 0;i < trainimages.size();i++){
        src = trainimages[i].clone();
        //get bounding rectangle of image for reference
        //function from facemark class
        facesp.clear();
        if(!getFaces(src,facesp)){
            continue;
        }
        if(facesp.size()>1||facesp.empty())
            continue;
        face = facesp[0];
        convertToUnit(face,warp_mat);
        //loop to bring points to a common reference and adding
        for(k=0;k<trainlandmarks[i].size();k++){
            Point2f pt=trainlandmarks[i][k];
            C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
            D = warp_mat*C;
            pt.x = float(D.at<double>(0,0));
            pt.y = float(D.at<double>(1,0));
            trainlandmarks[i][k] = pt;
            xmean[k] = xmean[k]+pt.x;
            ymean[k] = ymean[k]+pt.y;
        }
        finalimages.push_back(trainimages[i]);
        finallandmarks.push_back(trainlandmarks[i]);
        faces.push_back(face);
    }
    //dividing by size to get mean and initialize meanshape
    for(size_t i=0;i<k;i++){
        xmean[i]=xmean[i]/finalimages.size();
        ymean[i]=ymean[i]/finalimages.size();
        if(xmean[i]>maxmeanx)
            maxmeanx = xmean[i];
        if(xmean[i]<minmeanx)
            minmeanx = xmean[i];
        if(ymean[i]>maxmeany)
            maxmeany = ymean[i];
        if(ymean[i]<minmeany)
            minmeany = ymean[i];
        meanshape.push_back(Point2f(xmean[i],ymean[i]));
    }
    trainimages.clear();
    trainlandmarks.clear();
    trainimages = finalimages;
    trainlandmarks = finallandmarks;
    finalimages.clear();
    finallandmarks.clear();
    return true;
}
bool FacemarkKazemiImpl::scaleData( vector< vector<Point2f> > & trainlandmarks,
                                vector<Mat> & trainimages ,Size s)
{
    if(trainimages.empty()||trainimages.size()!=trainlandmarks.size()){
        // throw error if no data (or simply return -1?)
        CV_Error(Error::StsBadArg, "The data is not loaded properly by train function. Aborting...");
    }
    float scalex,scaley;
    //scale all images and their landmarks according  to input size
    for(size_t i=0;i< trainimages.size();i++){
        //calculating scale for x and y axis
        scalex=float(s.width)/float(trainimages[i].cols);
        scaley=float(s.height)/float(trainimages[i].rows);
        resize(trainimages[i],trainimages[i],s,0,0,INTER_LINEAR_EXACT);
        for (vector<Point2f>::iterator it = trainlandmarks[i].begin(); it != trainlandmarks[i].end(); it++) {
            Point2f pt = (*it);
            pt.x = pt.x*scalex;
            pt.y = pt.y*scaley;
            (*it) = pt;
        }
    }
    return true;
}
Ptr<FacemarkKazemi> FacemarkKazemi::create(const FacemarkKazemi::Params &parameters){
    return Ptr<FacemarkKazemiImpl>(new FacemarkKazemiImpl(parameters));
}

Ptr<Facemark> createFacemarkKazemi() {
    FacemarkKazemi::Params parameters;
    return Ptr<FacemarkKazemiImpl>(new FacemarkKazemiImpl(parameters));
}
}//cv
}//face
