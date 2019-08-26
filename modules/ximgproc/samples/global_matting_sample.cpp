#include "mattingClass.h"
#include<iostream>

int main(int argc,char** argv)
{
    if(argc<3)
    {
      cout<<"Enter the path of image and trimap"<<endl;
      return -1;
    }
    
    string img_path = argv[1];
    string tri_path = argv[2];
    int niter = 9;
    if(argc==4)
    {
      niter = atoi(argv[3]);
    } 
      
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
    cv::Mat trimap = cv::imread(tri_path, CV_LOAD_IMAGE_GRAYSCALE);
    
    if(image.empty() || trimap.empty())
    {
       cout<<"Could not load the inputs"<<endl;
       return -2;
    }
    // (optional) exploit the affinity of neighboring pixels to reduce the 
    // size of the unknown region. please refer to the paper
    // 'Shared Sampling for Real-Time Alpha Matting'.

    cv::Mat foreground, alpha;
    
    GlobalMatting gm;
    
    gm.getMat(image,trimap,foreground,alpha,niter);
    
    
    cv::imwrite("alpha-matte.png", alpha);

    return 0;
}
