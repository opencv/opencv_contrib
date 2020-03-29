// // This file is part of OpenCV project.
// // It is subject to the license terms in the LICENSE file found in the top-level directory
// // of this distribution and at http://opencv.org/license.html.
#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>


#include  <iostream>
#include  <fstream>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

bool fileExists (const string& filename)
{
    ifstream f(filename.c_str());
    return f.good();
}



int main(int argc, const char * argv[])
{
    if (argc < 3)
    {
        cout << "Insuficient parameters. Aborting!" << endl;
        cout << "Usage: textdetection_swt file_path dark_on_light" << endl;
        cout << "Usage: To detect dark text on light background pass dark_on_light as 1, otherwise pass 0" << endl;
        cout << "Example: ./swt ./scenetext_segmented_word02.jpg 0" << endl;
        exit(1);
    }

    bool dark_on_light = (argv[2][0] != '0');

    Mat image = imread("/home/opencv-dev/OSS/opencv_contrib/modules/text/samples/scenetext_segmented_word03.jpg", IMREAD_COLOR);
    namedWindow( "Input Image", WINDOW_AUTOSIZE );
    imshow("Input Image", image);

    cout << "Starting SWT Text Detection Demo with Dark on Light set to 0" << endl;

    vector<cv::Rect> components;
    Mat out( image.size(), CV_8UC3 );
    vector<cv::Rect> regions;
    cv::text::detectTextSWT(image, components, dark_on_light, out, regions);
    namedWindow( "Letter Candidates", WINDOW_AUTOSIZE );
    imshow ("Letter Candidates", out);
    waitKey();

    cout << components.size() << " letter candidates found" << endl;

    Mat image_copy = image.clone();

    for (unsigned int i = 0; i < regions.size(); i++) {
        rectangle(image_copy, regions[i], cv::Scalar(0, 0, 0), 3);
    }
    cout << regions.size() << "chains were obtained after merging suitable pairs" << endl;
    namedWindow( "Chains After Merging", WINDOW_AUTOSIZE );
    imshow ("Chains After Merging", image_copy);
    cout << "Recognition finished. Press any key to exit.\n";
    waitKey();
    return 0;
}
