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
        ColorCorrectionModel model1(src, Vinyl);

        /* brief More models with different parameters, try it & check the document for details.
        */
        // ColorCorrectionModel model2(src, Vinyl, AdobeRGB, CCM_4x3, CIE2000, GAMMA, 2.2, 3);
        // ColorCorrectionModel model3(src, Vinyl, WideGamutRGB, CCM_4x3, CIE2000, GRAYPOLYFIT, 2.2, 3);
        // ColorCorrectionModel model4(src, Vinyl, ProPhotoRGB, CCM_4x3, RGBL, GRAYLOGPOLYFIT, 2.2, 3);
        // ColorCorrectionModel model5(src, Vinyl, DCI_P3_RGB, CCM_3x3, RGB, IDENTITY_, 2.2, 3);
        // ColorCorrectionModel model6(src, Vinyl, AppleRGB, CCM_3x3, CIE2000, COLORPOLYFIT, 2.2, 2,{ 0, 0.98 },Mat(),2);
        // ColorCorrectionModel model7(src, Vinyl, REC_2020_RGB, CCM_3x3,  CIE94_GRAPHIC_ARTS, COLORLOGPOLYFIT, 2.2, 3);

        /* If you use a customized ColorChecker, you can use your own reference color values and corresponding color space in a way like:
        */
        // cv::Mat ref = (Mat_<Vec3d>(18, 1) <<
        // Vec3d(100, 0.00520000001, -0.0104),
        // Vec3d(73.0833969, -0.819999993, -2.02099991),
        // Vec3d(62.493, 0.425999999, -2.23099995),
        // Vec3d(50.4640007, 0.446999997, -2.32399988),
        // Vec3d(37.7970009, 0.0359999985, -1.29700005),
        // Vec3d(0, 0, 0),
        // Vec3d(51.5880013, 73.5179977, 51.5690002),
        // Vec3d(93.6989975, -15.7340002, 91.9420013),
        // Vec3d(69.4079971, -46.5940018, 50.4869995),
        // Vec3d(66.61000060000001, -13.6789999, -43.1720009),
        // Vec3d(11.7110004, 16.9799995, -37.1759987),
        // Vec3d(51.973999, 81.9440002, -8.40699959),
        // Vec3d(40.5489998, 50.4399986, 24.8490009),
        // Vec3d(60.8160019, 26.0690002, 49.4420013),
        // Vec3d(52.2529984, -19.9500008, -23.9960003),
        // Vec3d(51.2859993, 48.4700012, -15.0579996),
        // Vec3d(68.70700069999999, 12.2959995, 16.2129993),
        // Vec3d(63.6839981, 10.2930002, 16.7639999));

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
