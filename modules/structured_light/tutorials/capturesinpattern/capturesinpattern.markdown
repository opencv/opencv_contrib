Capture Sinusoidal pattern tutorial {#tutorial_capture_sinusoidal_pattern}
=============

Goal
----

In this tutorial, you will learn how to use the sinusoidal pattern class to:

- Generate sinusoidal patterns.
- Project the generated patterns.
- Capture the projected patterns.
- Compute a wrapped phase map from these patterns using three different algorithms (Fourier Transform Profilometry, Phase Shifting Profilometry, Fourier-assisted Phase Shifting Profilometry)
- Unwrap the previous phase map.

Code
----
@include structured_light/samples/capsinpattern.cpp

Expalantion
-----------
First, the sinusoidal patterns must be generated. *SinusoidalPattern* class parameters have to be set by the user:

- projector width and height
- number of periods in the patterns
- set cross markers in the patterns (used to convert relative phase map to absolute phase map)
- patterns direction (horizontal or vertical)
- phase shift value (usually set to 2pi/3 to enable a cyclical system)
- number of pixels between two consecutive markers on the same row/column
- id of the method used to compute the phase map (FTP = 0, PSP = 1, FAPS = 2)

The user can also choose to save the patterns and the phase map.

@code{.cpp}
	structured_light::SinusoidalPattern::Params params;
    params.width = parser.get<int>(0);
    params.height = parser.get<int>(1);
    params.nbrOfPeriods = parser.get<int>(2);
    params.setMarkers = parser.get<bool>(3);
    params.horizontal = parser.get<bool>(4);
    params.methodId = parser.get<int>(5);
    params.shiftValue = static_cast<float>(2 * CV_PI / 3);
    params.nbrOfPixelsBetweenMarkers = 70;
    String outputPatternPath = parser.get<String>(6);
    String outputWrappedPhasePath = parser.get<String>(7);
    String outputUnwrappedPhasePath = parser.get<String>(8);

    Ptr<structured_light::SinusoidalPattern> sinus = structured_light::SinusoidalPattern::create(params);
	// Storage for patterns
    vector<Mat> patterns;
    //Generate sinusoidal patterns
    sinus->generate(patterns);
@endcode
The number of patterns is always equal to three, no matter the method used to compute the phase map. Those three patterns are projected in a loop which is fine since the system is cyclical.

Once the patterns have been generated, the camera is opened and the patterns are projected, using fullscreen resolution. In this tutorial, a prosilica camera is used to capture gray images. When the first pattern is displayed by the projector, the user can press any key to start the projection sequence.

@code{.cpp}
VideoCapture cap(CAP_PVAPI);
    if( !cap.isOpened() )
    {
        cout << "Camera could not be opened" << endl;
        return -1;
    }
    cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_MONO8);

    namedWindow("pattern", WINDOW_NORMAL);
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    imshow("pattern", patterns[0]);
    cout << "Press any key when ready" << endl;
    waitKey(0);
@endcode

In this tutorial, 30 images are projected so, each of the three patterns is projected ten times.
The "while" loop takes care of the projection process. The captured images are stored in a vector of Mat. There is a 30 ms delay between two successive captures.
When the projection is done, the user has to press "Enter" to start computing the phase maps.

@code{.cpp}
    int nbrOfImages = 30;
    int count = 0;

    vector<Mat> img(nbrOfImages);
    Size camSize(-1, -1);

    while( count < nbrOfImages )
    {
        for(int i = 0; i < (int)patterns.size(); ++i )
        {
            imshow("pattern", patterns[i]);
            waitKey(30);
            cap >> img[count];
            count += 1;
        }
    }

    cout << "press enter when ready" << endl;
    bool loop = true;
    while ( loop )
    {
        char c = waitKey(0);
        if( c == 10 )
        {
            loop = false;
        }
    }
@endcode
The phase maps are ready to be computed according to the selected method.
For FTP, a phase map is computed for each projected pattern, but we need to compute the shadow mask from three successive patterns, as explained in @cite faps. Therefore, three patterns are set in a vector called captures. Care is taken to fill this vector with three patterns, especially when we reach the last captures. The unwrapping algorithm needs to know the size of the captured images so, we make sure to give it to the "unwrapPhaseMap" method.
The phase maps are converted to 8-bit images in order to save them as png.

@code{.cpp}
switch(params.methodId)
    {
        case structured_light::FTP:
            for( int i = 0; i < nbrOfImages; ++i )
            {
                /*We need three images to compute the shadow mask, as described in the reference paper
                 * even if the phase map is computed from one pattern only
                */
                vector<Mat> captures;
                if( i == nbrOfImages - 2 )
                {
                    captures.push_back(img[i]);
                    captures.push_back(img[i-1]);
                    captures.push_back(img[i+1]);
                }
                else if( i == nbrOfImages - 1 )
                {
                    captures.push_back(img[i]);
                    captures.push_back(img[i-1]);
                    captures.push_back(img[i-2]);
                }
                else
                {
                    captures.push_back(img[i]);
                    captures.push_back(img[i+1]);
                    captures.push_back(img[i+2]);
                }
                sinus->computePhaseMap(captures, wrappedPhaseMap, shadowMask);
                if( camSize.height == -1 )
                {
                    camSize.height = img[i].rows;
                    camSize.width = img[i].cols;
                }
                sinus->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);
                unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);
                wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);

                if( !outputUnwrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    imwrite(outputUnwrappedPhasePath + "_FTP_" + name.str() + ".png", unwrappedPhaseMap8);
                }

                if( !outputWrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    imwrite(outputWrappedPhasePath + "_FTP_" + name.str() + ".png", wrappedPhaseMap8);
                }
            }
            break;
@endcode

For PSP and FAPS, three projected images are used to compute a single phase map. These three images are set in "captures", a vector working as a FIFO.Here again, phase maps are converted to 8-bit images in order to save them as png.
@code{.cpp}
case structured_light::PSP:
        case structured_light::FAPS:
            for( int i = 0; i < nbrOfImages - 2; ++i )
            {
                vector<Mat> captures;
                captures.push_back(img[i]);
                captures.push_back(img[i+1]);
                captures.push_back(img[i+2]);

                sinus->computePhaseMap(captures, wrappedPhaseMap, shadowMask);
                if( camSize.height == -1 )
                {
                    camSize.height = img[i].rows;
                    camSize.width = img[i].cols;
                }
                sinus->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);
                unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);
                wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);

                if( !outputUnwrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    if( params.methodId == structured_light::PSP )
                        imwrite(outputUnwrappedPhasePath + "_PSP_" + name.str() + ".png", unwrappedPhaseMap8);
                    else
                        imwrite(outputUnwrappedPhasePath + "_FAPS_" + name.str() + ".png", unwrappedPhaseMap8);
                }

                if( !outputWrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    if( params.methodId == structured_light::PSP )
                        imwrite(outputWrappedPhasePath + "_PSP_" + name.str() + ".png", wrappedPhaseMap8);
                    else
                        imwrite(outputWrappedPhasePath + "_FAPS_" + name.str() + ".png", wrappedPhaseMap8);
                }
            }
            break;
@endcode
