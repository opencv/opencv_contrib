#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int main(int argc, char** argv)
{
    string in;
    CommandLineParser parser(argc, argv, "{@input|corridor.jpg|input image}{help h||show help message}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    in = samples::findFile(parser.get<string>("@input"));

    Mat image = imread(in, IMREAD_GRAYSCALE);

    if( image.empty() )
    {
        return -1;
    }

    // Create FLD detector
    // Param               Default value   Description
    // length_threshold    10            - Segments shorter than this will be discarded
    // distance_threshold  1.41421356    - A point placed from a hypothesis line
    //                                     segment farther than this will be
    //                                     regarded as an outlier
    // canny_th1           50            - First threshold for
    //                                     hysteresis procedure in Canny()
    // canny_th2           50            - Second threshold for
    //                                     hysteresis procedure in Canny()
    // canny_aperture_size 3            - Aperturesize for the sobel operator in Canny().
    //                                     If zero, Canny() is not applied and the input
    //                                     image is taken as an edge image.
    // do_merge            false         - If true, incremental merging of segments
    //                                     will be performed
    int length_threshold = 10;
    float distance_threshold = 1.41421356f;
    double canny_th1 = 50.0;
    double canny_th2 = 50.0;
    int canny_aperture_size = 3;
    bool do_merge = false;
    Ptr<FastLineDetector> fld = createFastLineDetector(length_threshold,
            distance_threshold, canny_th1, canny_th2, canny_aperture_size,
            do_merge);
    vector<Vec4f> lines;

    // Because of some CPU's power strategy, it seems that the first running of
    // an algorithm takes much longer. So here we run the algorithm 10 times
    // to see the algorithm's processing time with sufficiently warmed-up
    // CPU performance.
    for (int run_count = 0; run_count < 5; run_count++) {
        double freq = getTickFrequency();
        lines.clear();
        int64 start = getTickCount();
        // Detect the lines with FLD
        fld->detect(image, lines);
        double duration_ms = double(getTickCount() - start) * 1000 / freq;
        cout << "Elapsed time for FLD " << duration_ms << " ms." << endl;
    }

    // Show found lines with FLD
    Mat line_image_fld(image);
    fld->drawSegments(line_image_fld, lines);
    imshow("FLD result", line_image_fld);

    waitKey(1);

    Ptr<EdgeDrawing> ed = createEdgeDrawing();
    ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 38;
    ed->params.AnchorThresholdValue = 8;

    vector<Vec6d> ellipses;

    for (int run_count = 0; run_count < 5; run_count++) {
        double freq = getTickFrequency();
        lines.clear();
        int64 start = getTickCount();

        // Detect edges
        //you should call this before detectLines() and detectEllipses()
        ed->detectEdges(image);

        // Detect lines
        ed->detectLines(lines);
        double duration_ms = double(getTickCount() - start) * 1000 / freq;
        cout << "Elapsed time for EdgeDrawing detectLines " << duration_ms << " ms." << endl;

        start = getTickCount();
        // Detect circles and ellipses
        ed->detectEllipses(ellipses);
        duration_ms = double(getTickCount() - start) * 1000 / freq;
        cout << "Elapsed time for EdgeDrawing detectEllipses " << duration_ms << " ms." << endl;
    }

    Mat edge_image_ed = Mat::zeros(image.size(), CV_8UC3);
    vector<vector<Point> > segments = ed->getSegments();

    for (size_t i = 0; i < segments.size(); i++)
    {
        const Point* pts = &segments[i][0];
        int n = (int)segments[i].size();
        polylines(edge_image_ed, &pts, &n, 1, false, Scalar((rand() & 255), (rand() & 255), (rand() & 255)), 1);
    }

    imshow("EdgeDrawing detected edges", edge_image_ed);

    Mat line_image_ed(image);
    fld->drawSegments(line_image_ed, lines);

    // Draw circles and ellipses
    for (size_t i = 0; i < ellipses.size(); i++)
    {
        Point center((int)ellipses[i][0], (int)ellipses[i][1]);
        Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
        double angle(ellipses[i][5]);
        Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);

        ellipse(line_image_ed, center, axes, angle, 0, 360, color, 2, LINE_AA);
    }

    imshow("EdgeDrawing result", line_image_ed);
    waitKey();
    return 0;
}
