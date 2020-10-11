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
    constcolor :
            the Built-in color card;
            Supported list:
                Macbeth: Macbeth ColorChecker ;
                Vinyl: DKK ColorChecker ;
                DigitalSG: DigitalSG ColorChecker with 140 squares;
            type: enum CONST_COLOR;
    Mat colors_ :
           the reference color values
           and corresponding color space
           NOTICE: the color values are in [0, 1]
           type: cv::Mat
    ref_cs_ :
           the corresponding color space
           NOTICE: For the list of color spaces supported, see the notes below;
                  If the color type is some RGB, the format is RGB not BGR;
           type:enum COLOR_SPACE;
    cs_ :
            the absolute color space that detected colors convert to;
            NOTICE: it should be some RGB color space;
                    For the list of RGB color spaces supported, see the notes below;
            type: enum COLOR_SPACE;
    dst_ :
            the reference colors;
            NOTICE: custom color card are supported;
                    You should use Color
                    For the list of color spaces supported, see the notes below;
                    If the color type is some RGB, the format is RGB not BGR, and the color values are in [0, 1];

    ccm_type :
            the shape of color correction matrix(CCM);
            Supported list:
                "CCM_3x3": 3x3 matrix;
                "CCM_4x3": 4x3 matrix;
            type: enum CCM_TYPE;
            default: CCM_3x3;
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
            default: CIE2000;
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
            default: IDENTITY_;
    gamma :
            the gamma value of gamma correction;
            NOTICE: only valid when linear is set to "gamma";
            type: double;
            default: 2.2;
    deg :
            the degree of linearization polynomial;
            NOTICE: only valid when linear is set to "COLORPOLYFIT", "GRAYPOLYFIT",
                    "COLORLOGPOLYFIT" and "GRAYLOGPOLYFIT";
            type: int;
            default: 3;
    saturated_threshold :
            the threshold to determine saturation;
            NOTICE: it is a tuple of [low, up];
                    The colors in the closed interval [low, up] are reserved to participate
                    in the calculation of the loss function and initialization parameters.
            type: std::vector<double>;
            default: { 0, 0.98 };
    ---------------------------------------------------
    There are some ways to set weights:
        1. set weights_list only;
        2. set weights_coeff only;
    see CCM.pdf for details;
    weights_list :
            the list of weight of each color;
            type: cv::Mat;
            default: empty array;
    weights_coeff :
            the exponent number of L* component of the reference color in CIE Lab color space;
            type: double;
            default: 0;
    ---------------------------------------------------
    initial_method_type :
            the method of calculating CCM initial value;
            see CCM.pdf for details;
            Supported list:
                'LEAST_SQUARE': least-squre method;
                'WHITE_BALANCE': white balance method;
            type: enum INITIAL_METHOD_TYPE;
    max_count, epsilon :
            used in MinProblemSolver-DownhillSolver;
            Terminal criteria to the algorithm;
            type: int, double;
            default: 5000, 1e-4;
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
                XYZ_D65_10;
                XYZ_D50_10;
                XYZ_A_2;
                XYZ_A_10;
                XYZ_D55_2;
                XYZ_D55_10;
                XYZ_D75_2;
                XYZ_D75_10;
                XYZ_E_2;
                XYZ_E_10;
                Lab_D65_10;
                Lab_D50_10;
                Lab_A_2;
                Lab_A_10;
                Lab_D55_2;
                Lab_D55_10;
                Lab_D75_2;
                Lab_D75_10;
                Lab_E_2;
                Lab_E_10;
```


## Code

@snippet samples/color_correction_model.cpp tutorial

## Explanation

The first part is to detect the ColorChecker position.
@snippet samples/color_correction_model.cpp get_color_checker
@snippet samples/color_correction_model.cpp get_messages_of_image
Preparation for ColorChecker detection to get messages for the image.

@snippet samples/color_correction_model.cpp creat
The CCheckerDetectorobject is created and uses getListColorChecker function to get ColorChecker message.

@snippet samples/color_correction_model.cpp get_ccm_Matrix
For every ColorChecker, we can compute a ccm matrix for color correction. Model1 is an object of ColorCorrectionModel class. The parameters should be changed to get the best effect of color correction. See other parameters' detail at the Parameters.

@snippet samples/color_correction_model.cpp reference_color_values
If you use a customized ColorChecker, you can use your own reference color values and corresponding color space as shown above.

@snippet samples/color_correction_model.cpp make_color_correction
The member function infer_image is to make correction correction using ccm matrix.

@snippet samples/color_correction_model.cpp Save_calibrated_image
Save the calibrated image.