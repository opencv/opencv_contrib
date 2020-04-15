// Sample code which demonstrates the working of
// stroke width transform in the text module of OpenCV
#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
        cout << "Example: textdetection_swt ./scenetext_segmented_word02.jpg 0" << endl;
        exit(1);
    }

    string filepath = argv[1];
    if (!fileExists(filepath)) {
        cout << "Error: The input image does not exist" << endl;
        cout << "Please check the path to image file." << endl;
        cout << "Usage: textdetection_swt file_path dark_on_light" << endl;
        cout << "Usage: To detect dark text on light background pass dark_on_light as 1, otherwise pass 0" << endl;
        cout << "Example: ./swt ./scenetext_segmented_word02.jpg 0" << endl;
        exit(1);
    }

    bool dark_on_light = (argv[2][0] != '0');

    Mat image = imread(argv[1], IMREAD_COLOR);
    imshow("Input Image", image);

    cout << "Starting SWT Text Detection Demo with dark_on_light variable set to 0" << endl;

    vector<cv::Rect> components;
    Mat out;
    vector<cv::Rect> regions;
    cv::text::detectTextSWT(image, components, dark_on_light, out, regions);
    imshow ("Letter Candidates", out);
    waitKey();

    cout << components.size() << " letter candidates found" << endl;

    Mat image_copy = image.clone();

    for (unsigned int i = 0; i < regions.size(); i++) {
        rectangle(image_copy, regions[i], cv::Scalar(0, 0, 0), 3);
    }
    cout << regions.size() << "chains were obtained after merging suitable pairs" << endl;
    imshow ("Chains After Merging", image_copy);
    cout << "Recognition finished. Press any key to exit.\n";
    waitKey();
    return 0;
}
