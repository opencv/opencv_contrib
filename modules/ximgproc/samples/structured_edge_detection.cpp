/**************************************************************************************
The structured forests for fast edge detection demo requires you to provide a model.
This model can be found at the opencv_extra repository on Github on the following link:
https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
***************************************************************************************/

#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;

const char* keys =
{
    "{i || input image file name}"
    "{m || model file name}"
    "{o || output image file name}"
};

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates usage of structured forests for fast edge detection");
    parser.printMessage();

    if ( !parser.check() )
    {
        parser.printErrors();
        return -1;
    }

    String modelFilename = parser.get<String>("m");
    String inFilename = parser.get<String>("i");
    String outFilename = parser.get<String>("o");

    //! [imread]
    Mat image = imread(inFilename, IMREAD_COLOR);
    if ( image.empty() )
        CV_Error(Error::StsError, String("Cannot read image file: ") + inFilename);
    //! [imread]

    if ( modelFilename.size() == 0)
        CV_Error(Error::StsError, String("Empty model name"));

    //! [convert]
    image.convertTo(image, DataType<float>::type, 1/255.0);
    //! [convert]

    TickMeter tm;
    tm.start();
    //! [create]
    Ptr<StructuredEdgeDetection> pDollar =
        createStructuredEdgeDetection(modelFilename);
    //! [create]

    tm.stop();
    std::cout << "createStructuredEdgeDetection() time : " << tm << std::endl;

    tm.reset();
    tm.start();
    //! [detect]
    Mat edges;
    pDollar->detectEdges(image, edges);
    //! [detect]
    tm.stop();
    std::cout << "detectEdges() time : " << tm << std::endl;

    tm.reset();
    tm.start();
    //! [nms]
    // computes orientation from edge map
    Mat orientation_map;
    pDollar->computeOrientation(edges, orientation_map);

    // suppress edges
    Mat edge_nms;
    pDollar->edgesNms(edges, orientation_map, edge_nms, 2, 0, 1, true);
    //! [nms]

    tm.stop();
    std::cout << "nms time : " << tm << std::endl;

    //! [imshow]
    if ( outFilename.size() == 0 )
    {
        imshow("edges", edges);
        imshow("edges nms", edge_nms);
        waitKey(0);
    }
    else
        imwrite(outFilename, 255*edges);
    //! [imshow]

    return 0;
}
