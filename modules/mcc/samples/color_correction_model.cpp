#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/mcc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace mcc;
using namespace ccm;
using namespace std;

const char *about = "Basic chart detection";
const char *keys = {
    "{ help h usage ? |    | show this message }"
    "{t        |      |  chartType: 0-Standard, 1-DigitalSG, 2-Vinyl }"
    "{v        |      | Input from video file, if ommited, input comes from camera }"
    "{ci       | 0    | Camera id if input doesnt come from video (-v) }"
    "{f        | 1    | Path of the file to process (-v) }"
    "{nc       | 1    | Maximum number of charts in the image }"};

int main(int argc, char *argv[])
{

    // ----------------------------------------------------------
    // Scroll down a bit (~40 lines) to find actual relevant code
    // ----------------------------------------------------------

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    int t = parser.get<int>("t");
    int nc = parser.get<int>("nc");
    string filepath = parser.get<string>("f");

    CV_Assert(0 <= t && t <= 2);
    TYPECHART chartType = TYPECHART(t);

    cout << "t: " << t << " , nc: " << nc <<  ", \nf: " << filepath << endl;

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Mat image = cv::imread(filepath, IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Invalid Image!" << endl;
        return 1;
    }

    Mat imageCopy = image.clone();
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    // Marker type to detect
    if (!detector->process(image, chartType, nc))
    {
        printf("ChartColor not detected \n");
        return 2;
    }
    // get checker
    vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();

    for (Ptr<mcc::CChecker> checker : checkers)
    {
        Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
        cdraw->draw(image);
        Mat chartsRGB = checker->getChartsRGB();
        Mat src = chartsRGB.col(1).clone().reshape(3, 18);
        src /= 255.0;

        //compte color correction matrix
        ColorCorrectionModel model1(src, Vinyl_D50_2);

        /* brief More models with different parameters, try it & check the document for details.
        */
        // ColorCorrectionModel model2(src, Vinyl_D50_2, AdobeRGB, CCM_4x3, CIE2000, GAMMA, 2.2, 3);
        // ColorCorrectionModel model3(src, Vinyl_D50_2, WideGamutRGB, CCM_4x3, CIE2000, GRAYPOLYFIT, 2.2, 3);
        // ColorCorrectionModel model4(src, Vinyl_D50_2, ProPhotoRGB, CCM_4x3, RGBL, GRAYLOGPOLYFIT, 2.2, 3);
        // ColorCorrectionModel model5(src, Vinyl_D50_2, DCI_P3_RGB, CCM_3x3, RGB, IDENTITY_, 2.2, 3);
        // ColorCorrectionModel model6(src, Vinyl_D50_2, AppleRGB, CCM_3x3, CIE2000, COLORPOLYFIT, 2.2, 2,{ 0, 0.98 },Mat(),2);
        // ColorCorrectionModel model7(src, Vinyl_D50_2, REC_2020_RGB, CCM_3x3,  CIE94_GRAPHIC_ARTS, COLORLOGPOLYFIT, 2.2, 3);

        /* If you use a customized ColorChecker, you can use your own reference color values and corresponding color space in a way like:
        */
        // cv::Mat ref = = (Mat_<Vec3d>(18, 1) <<
        // Vec3d(1.00000000e+02, 5.20000001e-03, -1.04000000e-02),
        // Vec3d(7.30833969e+01, -8.19999993e-01, -2.02099991e+00),
        // Vec3d(6.24930000e+01, 4.25999999e-01, -2.23099995e+00),
        // Vec3d(5.04640007e+01, 4.46999997e-01, -2.32399988e+00),
        // Vec3d(3.77970009e+01, 3.59999985e-02, -1.29700005e+00),
        // Vec3d(0.00000000e+00, 0.00000000e+00, 0.00000000e+00),
        // Vec3d(5.15880013e+01, 7.35179977e+01, 5.15690002e+01),
        // Vec3d(9.36989975e+01, -1.57340002e+01, 9.19420013e+01),
        // Vec3d(6.94079971e+01, -4.65940018e+01, 5.04869995e+01),
        // Vec3d(6.66100006e+01, -1.36789999e+01, -4.31720009e+01),
        // Vec3d(1.17110004e+01, 1.69799995e+01, -3.71759987e+01),
        // Vec3d(5.19739990e+01, 8.19440002e+01, -8.40699959e+00),
        // Vec3d(4.05489998e+01, 5.04399986e+01, 2.48490009e+01),
        // Vec3d(6.08160019e+01, 2.60690002e+01, 4.94420013e+01),
        // Vec3d(5.22529984e+01, -1.99500008e+01, -2.39960003e+01),
        // Vec3d(5.12859993e+01, 4.84700012e+01, -1.50579996e+01),
        // Vec3d(6.87070007e+01, 1.22959995e+01, 1.62129993e+01),
        // Vec3d(6.36839981e+01, 1.02930002e+01, 1.67639999e+01));

        // ColorCorrectionModel model8(src,ref,Lab_D50_2);

        //make color correction
        Mat calibratedImage = model1.inferImage(filepath);

        // Save the calibrated image to {FILE_NAME}.calibrated.{FILE_EXT}
        string filename = filepath.substr(filepath.find_last_of('/')+1);
        size_t dotIndex = filename.find_last_of('.');
        string baseName = filename.substr(0, dotIndex);
        string ext = filename.substr(dotIndex+1, filename.length()-dotIndex);
        string calibratedFilePath = baseName + ".calibrated." + ext;
        cv::imwrite(calibratedFilePath, calibratedImage);
    }

    return 0;
}
