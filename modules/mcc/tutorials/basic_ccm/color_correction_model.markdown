Color Correction Model{#tutorial_ccm_color_correction_model}
===========================

In this tutorial you will learn how to use the 'Color Correction Model' to do a color correction in a image.

Reference
----

See details of ColorCorrection Algorithm at https://github.com/riskiest/color_calibration/tree/v4/doc/pdf/English/Algorithm

Building
----

When building OpenCV, run the following command to build all the contrib modules:

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

The sample has two parts of code, the first is the color checker detector model, see details at @ref tutorial_mcc_basic_chart_detection, the second part is to make collor calibration.

```
Here are the parameters for ColorCorrectionModel
    src :
            detected colors of ColorChecker patches;
            NOTICE: the color type is RGB not BGR, and the color values are in [0, 1];
    constcolor :
            the Built-in color card;
            Supported list:
                Macbeth: Macbeth ColorChecker ;
                Vinyl: DKK ColorChecker ;
                DigitalSG: DigitalSG ColorChecker with 140 squares;
    Mat colors :
           the reference color values
           and corresponding color space
           NOTICE: the color values are in [0, 1]
    ref_cs :
           the corresponding color space
                  If the color type is some RGB, the format is RGB not BGR;
    Supported Color Space:
            Supported list of RGB color spaces:
                COLOR_SPACE_sRGB;
                COLOR_SPACE_AdobeRGB;
                COLOR_SPACE_WideGamutRGB;
                COLOR_SPACE_ProPhotoRGB;
                COLOR_SPACE_DCI_P3_RGB;
                COLOR_SPACE_AppleRGB;
                COLOR_SPACE_REC_709_RGB;
                COLOR_SPACE_REC_2020_RGB;
            Supported list of linear RGB color spaces:
                COLOR_SPACE_sRGBL;
                COLOR_SPACE_AdobeRGBL;
                COLOR_SPACE_WideGamutRGBL;
                COLOR_SPACE_ProPhotoRGBL;
                COLOR_SPACE_DCI_P3_RGBL;
                COLOR_SPACE_AppleRGBL;
                COLOR_SPACE_REC_709_RGBL;
                COLOR_SPACE_REC_2020_RGBL;
            Supported list of non-RGB color spaces:
                COLOR_SPACE_Lab_D50_2;
                COLOR_SPACE_Lab_D65_2;
                COLOR_SPACE_XYZ_D50_2;
                COLOR_SPACE_XYZ_D65_2;
                COLOR_SPACE_XYZ_D65_10;
                COLOR_SPACE_XYZ_D50_10;
                COLOR_SPACE_XYZ_A_2;
                COLOR_SPACE_XYZ_A_10;
                COLOR_SPACE_XYZ_D55_2;
                COLOR_SPACE_XYZ_D55_10;
                COLOR_SPACE_XYZ_D75_2;
                COLOR_SPACE_XYZ_D75_10;
                COLOR_SPACE_XYZ_E_2;
                COLOR_SPACE_XYZ_E_10;
                COLOR_SPACE_Lab_D65_10;
                COLOR_SPACE_Lab_D50_10;
                COLOR_SPACE_Lab_A_2;
                COLOR_SPACE_Lab_A_10;
                COLOR_SPACE_Lab_D55_2;
                COLOR_SPACE_Lab_D55_10;
                COLOR_SPACE_Lab_D75_2;
                COLOR_SPACE_Lab_D75_10;
                COLOR_SPACE_Lab_E_2;
                COLOR_SPACE_Lab_E_10;
```


## Code

@snippet samples/color_correction_model.cpp tutorial

## Explanation

The first part is to detect the ColorChecker position.
@snippet samples/color_correction_model.cpp get_color_checker
@snippet samples/color_correction_model.cpp get_messages_of_image
Preparation for ColorChecker detection to get messages for the image.

@snippet samples/color_correction_model.cpp create
The CCheckerDetectorobject is created and uses getListColorChecker function to get ColorChecker message.

@snippet samples/color_correction_model.cpp get_ccm_Matrix
For every ColorChecker, we can compute a ccm matrix for color correction. Model1 is an object of ColorCorrectionModel class. The parameters should be changed to get the best effect of color correction. See other parameters' detail at the Parameters.

@snippet samples/color_correction_model.cpp reference_color_values
If you use a customized ColorChecker, you can use your own reference color values and corresponding color space as shown above.

@snippet samples/color_correction_model.cpp make_color_correction
The member function infer_image is to make correction correction using ccm matrix.

@snippet samples/color_correction_model.cpp Save_calibrated_image
Save the calibrated image.
