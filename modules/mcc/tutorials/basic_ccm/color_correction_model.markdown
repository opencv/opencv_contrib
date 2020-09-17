Color Correction Model{#tutorial_ccm_color_correction_model}
===========================

In this tutorial you will learn how to use the 'Color Correction Model' to do a color correction in a image.

Reference
----

See details of ColorCorrection Algorithm at https://github.com/riskiest/color_calibration/tree/v4/doc/pdf/English/Algorithm.

Building
----

When building OpenCV, run the following command to build all the contrib module:

```make
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules/
```

Or only build the mcc module:

```make
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules/mcc
```

Or make sure you check the mcc module in the GUI version of CMake: cmake-gui.

Source Code of the sample
-----------

The sample has two parts of code, the first is the color checker detector model, see details at[basic_chart_detection](https://github.com/opencv/opencv_contrib/tree/master/modules/mcc/tutorials/basic_chart_detection), the second part is to make collor calibration.

```
Here are the parameters for ColorCorrectionModel
src :
        detected colors of ColorChecker patches;
        NOTICE: the color type is RGB not BGR, and the color values are in [0, 1];
        type: cv::Mat;
dst :
        the reference colors;
        NOTICE: Built-in color card or custom color card are supported;
                Built-in:
                    Macbeth_D50_2: Macbeth ColorChecker with 2deg D50;
                    Macbeth_D65_2: Macbeth ColorChecker with 2deg D65;
                Custom:
                    You should use Color
                    For the list of color spaces supported, see the notes below;
                    If the color type is some RGB, the format is RGB not BGR, and the color values are in [0, 1];
        type: Color;
colorspace :
        the absolute color space that detected colors convert to;
        NOTICE: it should be some RGB color space;
                For the list of RGB color spaces supported, see the notes below;
        type: ColorSpace;
ccm_type :
        the shape of color correction matrix(CCM);
        Supported list:
            "CCM_3x3": 3x3 matrix;
            "CCM_4x3": 4x3 matrix;
        type: enum CCM_TYPE;
distance :
        the type of color distance;
        Supported list:
            "CIE2000";
            "CIE94_GRAPHIC_ARTS";
            "CIE94_TEXTILES";
            "CIE76";
            "CMC_1TO1";
            "CMC_2TO1";
            "RGB" : Euclidean distance of rgb color space;
            "RGBL" : Euclidean distance of rgbl color space;
        type: enum DISTANCE_TYPE;
linear_type :
        the method of linearization;
        NOTICE: see Linearization.pdf for details;
        Supported list:
            "IDENTITY_" : no change is made;
            "GAMMA": gamma correction;
                    Need assign a value to gamma simultaneously;
            "COLORPOLYFIT": polynomial fitting channels respectively;
                            Need assign a value to deg simultaneously;
            "GRAYPOLYFIT": grayscale polynomial fitting;
                            Need assign a value to deg and dst_whites simultaneously;
            "COLORLOGPOLYFIT": logarithmic polynomial fitting channels respectively;
                            Need assign a value to deg simultaneously;
            "GRAYLOGPOLYFIT": grayscale Logarithmic polynomial fitting;
                            Need assign a value to deg and dst_whites simultaneously;
        type: enum LINEAR_TYPE;
gamma :
        the gamma value of gamma correction;
        NOTICE: only valid when linear is set to "gamma";
        type: double;

deg :
        the degree of linearization polynomial;
        NOTICE: only valid when linear is set to "COLORPOLYFIT", "GRAYPOLYFIT",
                "COLORLOGPOLYFIT" and "GRAYLOGPOLYFIT";
        type: int;
saturated_threshold :
        the threshold to determine saturation;
        NOTICE: it is a tuple of [low, up];
                The colors in the closed interval [low, up] are reserved to participate
                in the calculation of the loss function and initialization parameters.
        type: std::vector<double>;
---------------------------------------------------
There are some ways to set weights:
    1. set weights_list only;
    2. set weights_coeff only;
see CCM.pdf for details;

weights_list :
        the list of weight of each color;
        type: cv::Mat;

weights_coeff :
        the exponent number of L* component of the reference color in CIE Lab color space;
        type: double;
---------------------------------------------------
initial_method_type :
        the method of calculating CCM initial value;
        see CCM.pdf for details;
        Supported list:
            'LEAST_SQUARE': least-squre method;
            'WHITE_BALANCE': white balance method;

max_count, epsilon :
        used in MinProblemSolver-DownhillSolver;
        Terminal criteria to the algorithm;
        type: int, double;


---------------------------------------------------
Supported Color Space:
        Supported list of RGB color spaces:
            sRGB;
            AdobeRGB;
            WideGamutRGB;
            ProPhotoRGB;
            DCI_P3_RGB;
            AppleRGB;
            REC_709_RGB;
            REC_2020_RGB;

        Supported list of linear RGB color spaces:
            sRGBL;
            AdobeRGBL;
            WideGamutRGBL;
            ProPhotoRGBL;
            DCI_P3_RGBL;
            AppleRGBL;
            REC_709_RGBL;
            REC_2020_RGBL;

        Supported list of non-RGB color spaces:
            Lab_D50_2;
            Lab_D65_2;
            XYZ_D50_2;
            XYZ_D65_2;

        Supported IO (You can use Lab(io) or XYZ(io) to create color space):
            A_2;
            A_10;
            D50_2;
            D50_10;
            D55_2;
            D55_10;
            D65_2;
            D65_10;
            D75_2;
            D75_10;
            E_2;
            E_10;
```


## Code

@include mcc/samples/color_correction_model.cpp

## Explanation

 The first part is to detect the ColorChecker position.

@code{.cpp}#include <opencv2/core.hpp>#include <opencv2/highgui.hpp>#include <opencv2/imgcodecs.hpp>#include <opencv2/mcc.hpp>#include <iostream>using namespace std;using namespace cv;using namespace mcc;using namespace ccm;using namespace std;@endcode

```
Here is sets of header and namespaces. You can set other namespace like the code above.
```

@code{.cpp}

const char *about = "Basic chart detection";const char *keys = {  "{ help h usage ? |  | show this message }" "{t    |   | chartType: 0-Standard, 1-DigitalSG, 2-Vinyl }"   "{v    |   | Input from video file, if ommited, input comes from camera }"  "{ci    | 0  | Camera id if input doesnt come from video (-v) }"

  "{f    | 1  | Path of the file to process (-v) }"  "{nc    | 1  | Maximum number of charts in the image }"};@ endcode

```
Some arguments for ColorChecker detection.
```

@code{.cpp}  CommandLineParser parser(argc, argv, keys);  parser.about(about);  int t = parser.get<int>("t");  int nc = parser.get<int>("nc");  string filepath = parser.get<string>("f");  CV_Assert(0 <= t && t <= 2);  TYPECHART chartType = TYPECHART(t);  cout << "t: " << t << " , nc: " << nc << ", \nf: " << filepath << endl;if (!parser.check()){    parser.printErrors();return 0;}  Mat image = cv::imread(filepath, IMREAD_COLOR);if (!image.data)  {  cout << "Invalid Image!" << endl;    return 1;  } @endcode

```
Preparation for ColorChecker detection to get messages for the image.
```

@code{.cpp}Mat imageCopy = image.clone();

  Ptr<CCheckerDetector> detector = CCheckerDetector::create();  if (!detector->process(image, chartType, nc))  {  printf("ChartColor not detected \n");  return 2;  }  vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();@endcode

```
The CCheckerDetectorobject is created and uses getListColorChecker function to get ColorChecker message.
```

@code{.cpp} for (Ptr<mcc::CChecker> checker : checkers)  {    Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);   cdraw->draw(image);    Mat chartsRGB = checker->getChartsRGB();Mat src = chartsRGB.col(1).clone().reshape(3, 18);    src /= 255.0;    //compte color correction matrix ColorCorrectionModel model1(src, Vinyl_D50_2);}@endcode

```
For every ColorChecker, we can compute a ccm matrix for color correction. Model1 is an object of ColorCorrectionModel class. The parameters should be changed to get the best effect of color correction. See other parameters' detail at the Parameters.
```

@code{.cpp}cv::Mat ref = = (Mat_<Vec3d>(18, 1) <<Vec3d(1.00000000e+02, 5.20000001e-03, -1.04000000e-02),Vec3d(7.30833969e+01, -8.19999993e-01, -2.02099991e+00), Vec3d(6.24930000e+01, 4.25999999e-01, -2.23099995e+00),Vec3d(5.04640007e+01, 4.46999997e-01, -2.32399988e+00),Vec3d(3.77970009e+01, 3.59999985e-02, -1.29700005e+00),Vec3d(0.00000000e+00, 0.00000000e+00, 0.00000000e+00),Vec3d(5.15880013e+01, 7.35179977e+01, 5.15690002e+01),Vec3d(9.36989975e+01, -1.57340002e+01, 9.19420013e+01),Vec3d(6.94079971e+01, -4.65940018e+01, 5.04869995e+01),Vec3d(6.66100006e+01, -1.36789999e+01, -4.31720009e+01),Vec3d(1.17110004e+01, 1.69799995e+01, -3.71759987e+01),Vec3d(5.19739990e+01, 8.19440002e+01, -8.40699959e+00), Vec3d(4.05489998e+01, 5.04399986e+01, 2.48490009e+01), Vec3d(6.08160019e+01, 2.60690002e+01, 4.94420013e+01), Vec3d(5.22529984e+01, -1.99500008e+01, -2.39960003e+01),Vec3d(5.12859993e+01, 4.84700012e+01, -1.50579996e+01),Vec3d(6.87070007e+01, 1.22959995e+01, 1.62129993e+01),Vec3d(6.36839981e+01, 1.02930002e+01, 1.67639999e+01));

ColorCorrectionModel model8(src,ref,Lab_D50_2); @endcode

```
If you use a customized ColorChecker, you can use your own reference color values and corresponding color space As shown above.
```

@code{.cpp}Mat calibratedImage = model1.inferImage(filepath); @endcode

```
The member function infer_image is to make correction correction using ccm matrix.
```

@code{.cpp}string filename = filepath.substr(filepath.find_last_of('/')+1);int dotIndex = filename.find_last_of('.');  string baseName = filename.substr(0, dotIndex);    string ext = filename.substr(dotIndex+1, filename.length()-dotIndex);    string calibratedFilePath = baseName + ".calibrated." + ext;    cv::imwrite(calibratedFilePath, calibratedImage); @endcode

```
Save the calibrated image.
```
