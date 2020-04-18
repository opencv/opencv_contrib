// Sample code which demonstrates the working of
// stroke width transform in the text module of OpenCV
#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

static void help(const CommandLineParser& cmd, const string& errorMessage)
{
    cout << errorMessage << endl;
    cout << "Avaible options:" << endl;
    cmd.printMessage();
}

static bool fileExists (const string& filename)
{
    ifstream f(filename.c_str());
    return f.good();
}

int main(int argc, const char * argv[])
{
    const char* keys =
        "{help h usage ? |false | print this message }"
        "{@image         |      | path to image }"
        "{@darkOnLight   |false | indicates whether text to be extracted is dark on a light brackground. Defaults to false. }"
        ;

    CommandLineParser cmd(argc, argv, keys);

    if(cmd.get<bool>("help"))
    {
        help(cmd, "Usage: ./textdetection_swt [options] \nExample: ./textdetection_swt scenetext_segmented_word03.jpg true");
        return EXIT_FAILURE;
    }

    string filepath = cmd.get<string>("@image");

    if (!fileExists(filepath)) {
        help(cmd, "ERROR: Could not find the image file. Please check the path.");
        return EXIT_FAILURE;
    }

    bool dark_on_light = cmd.get<bool>("@darkOnLight");

    Mat image = imread(filepath, IMREAD_COLOR);

    if (image.empty())
    {
        help(cmd, "ERROR: Could not load the image file");
        return EXIT_FAILURE;
    }

    cout << "Starting SWT Text Detection Demo with dark_on_light variable set to " << dark_on_light << endl;

    imshow("Input Image", image);
    waitKey(1);

    vector<cv::Rect> components;
    Mat out;
    vector<cv::Rect> regions;
    cv::text::detectTextSWT(image, components, dark_on_light, out, regions);

    imshow ("Letter Candidates", out);
    waitKey(1);

    cout << components.size() << " letter candidates found." << endl;

    Mat image_copy = image.clone();

    for (unsigned int i = 0; i < regions.size(); i++) {
        rectangle(image_copy, regions[i], cv::Scalar(0, 0, 0), 3);
    }
    cout << regions.size() << " chains were obtained after merging suitable pairs" << endl;
    cout << "Recognition finished. Press any key to exit..." << endl;

    imshow ("Chains After Merging", image_copy);
    waitKey();

    return 0;
}
