#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/hfs.hpp"

using namespace cv;
using namespace cv::hfs;
int main(int argc, char *argv[])
{
    // invalid number of command line parameter
    if( argc != 2 ) {
        return EXIT_FAILURE;
    }

    char* path = argv[1];
    // read in a pictrue to initialize the height and width
    Mat src = imread(path), res;
    int _h = src.rows, _w = src.cols;

    // initialize the HfsSegment object
    // In this example, we used default paramters.
    // However, bear in mind that you can pass in your
    // own parameters in with this function.
    Ptr<HfsSegment> h = HfsSegment::create( _h, _w );

    // segment and write the first result.
    res = h->performSegmentGpu(src);
    imwrite( "segment_default_gpu.jpg", res );
    // also, there is CPU interface for that
    res = h->performSegmentCpu(src);
    imwrite( "segment_default_cpu.jpg", res );

    // also, instead of getting a segmented image
    // from our interface, you can also choose to not to
    // draw the result on the Mat and only get a matrix
    // of index. Note that the data type of the returned
    // Mat in this case is CV_16U
    Mat idx_mat = h->performSegmentGpu( src, false );

    // also, you can change any parameters as you want
    h->setSlicSpixelSize(10);
    res = h->performSegmentGpu(src);
    imwrite( "segment_changed_param.jpg", res );

    return 0;
}
