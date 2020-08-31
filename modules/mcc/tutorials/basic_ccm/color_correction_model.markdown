Color Correction Model{#tutorial_ccm_color_correction_model}
===========================

In this tutorial you will learn how to use the 'Color Correction Model' to do a color correction in a image.

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

Parameters

```
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
linear :
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
