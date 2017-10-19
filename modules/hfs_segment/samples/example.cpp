#include "opencv2/hfs_segment.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace cv::hfs;
int main()
{
    String str = "data/000";
    // read in a pictrue to initialize the height and width
    Mat src = imread(str + ".jpg"), res;
    int _h = src.rows, _w = src.cols;

    // initialize the HfsSegment object
    // In this example, we used default paramters.
    // However, bear in mind that you can pass in your
    // own parameters in with this function.
    Ptr<HfsSegment> h = HfsSegment::create( _h, _w );
    

    // segment and write the first result.
    res = h->performSegment(src);
    imwrite( str + "_seg.jpg", res );

    // segment and write the second result.
    str = "data/001";
    src = imread(str + ".jpg");
    res = h->performSegment(src);
    imwrite( str + "_seg.jpg", res );

    // segment and write the third result.
    str = "data/002";
    src = imread(str + ".jpg");
    res = h->performSegment(src);
    imwrite( str + "_seg.jpg", res );

    // also, instead of getting a segmented image
    // from our interface, you can also choose to not to 
    // draw the result on the Mat and only get a matrix
    // of index. Note that the data type of the returned
    // Mat in this case is CV_16U
    str = "data/000";
    src = imread(str + ".jpg");
    Mat idx_mat = h->performSegment( src, false );

    // also, you can change the parameter as you wish
    h->setSlicSpixelSize(10);
    res = h->performSegment(src);
    imwrite( str + "_seg_changed.jpg", res );

    return 0;
}