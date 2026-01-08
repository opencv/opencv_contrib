/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.

 *  //
 //M*/

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/phase_unwrapping.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>



using namespace cv;
using namespace std;

static const char* keys =
{
    "{@inputPath | | Path of the wrapped phase map saved in a yaml file }"
    "{@outputUnwrappedName | | Path of the unwrapped phase map to be saved in a yaml file and as an 8 bit png}"
};

static void help()
{
    cout << "\nThis example shows how to use the \"Phase unwrapping module\" to unwrap a phase map"
            " saved in a yaml file (see extra_data\\phase_unwrapping\\data\\wrappedpeaks.yml)."
            " The mat name in the file should be \"phaseValue\". The result is saved in a yaml file"
            " too. Two images (wrapped.png and output_name.png) are also created"
            " for visualization purpose."
            "\nTo call: ./example_phase_unwrapping_unwrap <input_path> <output_unwrapped_name> \n"
         << endl;
}
int main(int argc, char **argv)
{
    phase_unwrapping::HistogramPhaseUnwrapping::Params params;

    CommandLineParser parser(argc, argv, keys);
    String inputPath = parser.get<String>(0);
    String outputUnwrappedName = parser.get<String>(1);

    if( inputPath.empty() || outputUnwrappedName.empty() )
    {
        help();
        return -1;
    }
    FileStorage fsInput(inputPath, FileStorage::READ);
    FileStorage fsOutput(outputUnwrappedName + ".yml", FileStorage::WRITE);

    Mat wPhaseMap;
    Mat uPhaseMap;
    Mat reliabilities;
    fsInput["phaseValues"] >> wPhaseMap;
    fsInput.release();
    params.width = wPhaseMap.cols;
    params.height = wPhaseMap.rows;

    Ptr<phase_unwrapping::HistogramPhaseUnwrapping> phaseUnwrapping = phase_unwrapping::HistogramPhaseUnwrapping::create(params);

    phaseUnwrapping->unwrapPhaseMap(wPhaseMap, uPhaseMap);
    fsOutput << "phaseValues" << uPhaseMap;
    fsOutput.release();

    phaseUnwrapping->getInverseReliabilityMap(reliabilities);

    Mat uPhaseMap8, wPhaseMap8, reliabilities8;
    wPhaseMap.convertTo(wPhaseMap8, CV_8U, 255, 128);
    uPhaseMap.convertTo(uPhaseMap8, CV_8U, 1, 128);
    reliabilities.convertTo(reliabilities8, CV_8U, 255,128);

    imshow("reliabilities", reliabilities);
    imshow("wrapped phase map", wPhaseMap8);
    imshow("unwrapped phase map", uPhaseMap8);

    imwrite(outputUnwrappedName + ".png", uPhaseMap8);
    imwrite("reliabilities.png", reliabilities8);

    bool loop = true;
    while( loop )
    {
        char key = (char)waitKey(0);
        if( key == 27 )
        {
            loop = false;
        }
    }
    return 0;
}