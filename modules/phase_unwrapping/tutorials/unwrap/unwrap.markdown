Unwrap two-dimensional phase maps {#tutorial_unwrap}
==============

Goal
----

In this tutorial, you will learn how to use the phase unwrapping module to unwrap two-dimensional phase maps. The implementation is based on @cite histogramUnwrapping.

Code
----
@include phase_unwrapping/samples/unwrap.cpp

Explanation
-----------

To use this example, wrapped phase map values should be stored in a yml file as CV_32FC1 Mat, under the name "phaseValues". Path to the data and a name to save the unwrapped phase map must be set in the command line. The results are saved with floating point precision in a yml file and as an 8-bit image for visualization purpose.

Some parameters can be chosen by the user:
- histThresh is a parameter used to divide the histogram in two parts. Bins before histThresh are smaller than the ones after histThresh. (Default value is 3*pi*pi).
- nbrOfSmallBins is the number of bins between 0 and histThresh. (Default value is 10).
- nbrOfLargeBins is the number of bins between histThresh and 32*pi*pi. (Default value is 5).

@code{.cpp}
phase_unwrapping::HistogramPhaseUnwrapping::Params params;

    CommandLineParser parser(argc, argv, keys);
    String inputPath = parser.get<String>(0);
    String outputUnwrappedName = parser.get<String>(1);
    String outputWrappedName = parser.get<String>(2);

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
@endcode
The wrapped phase map is unwrapped and the result is saved in a yml file. We can also get the reliabilities map for visualization purpose. The unwrapped phase map and the reliabilities map are converted to 8-bit images in order to be saved as png files.

@code{.cpp}
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
@endcode
