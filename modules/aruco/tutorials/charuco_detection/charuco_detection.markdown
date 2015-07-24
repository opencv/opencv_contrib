Detection of ChArUco Corners {#tutorial_charuco_detection}
==============================

ArUco markers and boards are very useful due to their useful and fast detection and their versatility.
However, one of the problems of ArUco markers is that the accurate of their corner position is not too high,
even after applying subpixel refinement. 

On the contrary, the corners of chessboard patterns can be refined more accurately since the corner is
composed by two squares. However, finding a chessboard pattern is not as versatile as finding an ArUco board:
it has to be completely visible, occlusions are not permitted.

A ChArUco board tries to combines the benefits of these two approaches:

![Charuco definition](images/charucodefinition.png)

The ArUco are used to interpolate the position of the chessboard corners, so that it has the versatility of marker
boards, since it allows occlusions or partial views. Moreover, since the interpolated corners belong to a chessboard,
they are very accurate in terms of subpixel accuracy.

When high precision is necessary, such as in camera calibration, Charuco boards is a better option than standard
Aruco boards.


ChArUco Board Creation
------

The aruco module provides the CharucoBoard class that represents a Charuco Board and which inherits from the Board class. 

This class, as the rest of Charuco functionalities, are defined in:

``` c++
    #include <opencv2/aruco/charuco.cpp>
```

To define a CharucoBoard, it is necesary:

- Number of chessboard squares in X direction.
- Number of chessboard squares in Y direction.
- Length of square side.
- Length of marker side.
- The dictionary of the markers.
- Ids of all the markers.

As for the GridBoard objects, the aruco module provides a function to create CharucoBoards easily. This function
is the static function cv::aruco::CharucoBoard::create() :

``` c++
    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, DICT_6X6_250);
```

- The first and second parameters are the number of squares in X and Y direction respectively.
- The third and fourth parameters are the length of the squares and the markers respectively. They can be provided
in any unit, having in mind that the estimated pose for this board would be measured in the same units (usually in meters).
- Finally, the dictionary of the markers is provided.

The ids of each of the markers are assigned in numerical order by default, like in GridBoard::create(). 
This can be easily customized by accessing to the ids vector through board.ids, like in the Board parent class.

Once we have our CharucoBoard object, we can create the image of it for printing. This can be done with the 
CharucoBoard::draw() method:

``` c++
    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, DICT_6X6_250);
    cv::Mat boardImage;
    board.draw( cv::Size(600, 500), boardImage, 10, 1 );
```

- The first parameter is the size of the output image in pixels. In this case 600x500 pixels. If this is not proportional
to the board dimensions, it will be centered on the image.
- boardImage: the output image with the board.
- The third parameter is the (optional) margin in pixels, so none of the marker boards is touching the image border.
In this case the margin is 10.
- Finally, the size of the marker border, similarly to drawMarker() function. The default value is 1.

The output image will be something like this:

![](images/charucoboard.jpg)


ChArUco Board Detection
------

When you detect a ChArUco board, what you are actually detecting is each of the chessboard corners of the board.

Each corner on a charuco board has a unique identifier (id) assigned. These ids goes from 0 to the total number of corners
in the board.

So, a detected ChArUco board consists in:

- vector<Point2f> charucoCorners : list of image position of the detected corners.
- vector <int> charucoIds : ids for each of the detected corners in charucoCorners.

The detection of the Charuco corners is based on the previous detected markers. So that, first markers are detected, and then
Charuco corners are interpolated from markers.

The function that detect the Charuco corners is cv::aruco::interpolateCornersCharuco() :

``` c++
    cv::Mat inputImage;
    cv::Mat cameraMatrix, distCoeffs;
    // camera parameters are read from somewhere
    readCameraParameters(cameraMatrix, distCoeffs);
    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, DICT_6X6_250);
    ...
    vector< int > markerIds;
    vector< vector<Point2f> > markerCorners;
    detectMarkers(inputImage, board.dictionary, markerCorners, markerIds);

    if(markerIds.size() > 0) {
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        int valid = interpolateCornersCharuco(markerCorners, markerIds, inputImage, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs);
    }
```

The parameters of the interpolateCornersCharuco() are:
- markerCorners and markerIds: the detected markers from detectMarkers() function.
- inputImage: the original image where the markers were detected. The image is necessary to perform subpixel refinement
in the Charuco corners.
- board: the CharucoBoard object
- charucoCorners and CharucoIds: the output interpolated Charuco corners
- cameraMatrix and distCoeffs: the optional camera calibration parameters
- The function returns the number of Charuco corners interpolated.

In this case, we have call the interpolateCornersCharuco() providing the camera calibration parameters. However these parameters
are optional. A similar example without these parameter would be:

``` c++
    cv::Mat inputImage;
    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, DICT_6X6_250);
    ...
    vector< int > markerIds;
    vector< vector<Point2f> > markerCorners;
    DetectorParameters params;
    params.doCornerRefinement = false;
    detectMarkers(inputImage, board.dictionary, markerCorners, markerIds, params);

    if(markerIds.size() > 0) {
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        int valid = interpolateCornersCharuco(markerCorners, markerIds, inputImage, board, charucoCorners, charucoIds);
    }
```

If calibration parameters are provided, the charuco corners are interpolated by, first, estimating a rough pose from the aruco markers
and, then, reprojecting the Charuco corners back to the image.

On the other hand, if calibration parameters are not provided, the Charuco corners are interpolated by calculating the
corresponding homography between the Charuco plane and the Charuco image projection. 

The main problem of using homography is that the interpolation is more sensible to image distortion. Actually, the homography is only performed
using the closest markers of each Charuco corner to reduce the effect of distortion.

In this case it is also recommended to disable the corner refinement of markers. The reason of this
is that, due to the proximity of the chessboard squares, the subpixel process can produce important
variations in the corner positions and these variations are propagated to the Charuco corner interpolation,
producing poor results.

Furthermore, only those corners whose two surrounding markers have be found are returned. If any of the two surrounding markers have 
not been detected usually means that there is some occlusion in that zone or the image quality is not good in that zone. In any case, it is 
preferable not to consider that corner, since what we want is to be sure that the interpolated Charuco corners are very accurate.

After the Charuco corners have been interpolated, a subpixel refinement is performed to obtain very accurate corners.

Once we have interpolated the Charuco corners, we would probably want to draw them to see if their detection is correct. 
This can be easily done using the drawDetectedCornersCharuco() function:

``` c++
    drawDetectedCornersCharuco(inputImage, outputImage, charucoCorners, charucoIds, color);
```

- The inputImage is the image where the corners have been detected.
- The outputImage will be a clone of inputImage with the corners drawn.
- charucoCorners and charucoIds are the detected Charuco corners from the interpolateCornersCharuco() function.
- Finally, the last parameter is the (optional) color we want to draw the corners, of type cv::Scalar.

For this image:

![Image with Charuco board](images/choriginal.png)

The result will be:

![Charuco board detected](images/chcorners.png)

In the presence of occlusion. like in the following image, although some corners are clearly visible, not all their surrounding markers have been detected due occlusion and, thus, they are not interpolated:

![Charuco detection with occlusion](images/chocclusion.png)

Finally, this a full example of Charuco detection (without using calibration parameters):

``` c++
    cv::VideoCapture inputVideo;
    inputVideo.open(0);

    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, DICT_6X6_250);

    DetectorParameters params;
    params.doCornerRefinement = false;

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
    
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids, params);
    
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, corners, ids);

            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;
            interpolateCornersCharuco(corners, ids, image, board, charucoCorners, charucoIds);
            if(charucoIds.size() > 0) 
                drawDetectedCornersCharuco(imageCopy, imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
        }
    
        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(waitTime);
        if (key == 27)
            break;
    }
```



ChArUco Pose Estimation
------

The final goal of the Charuco boards is finding corners very accurately for a high precision calibration or pose estimation.

The aruco module provides a function to perform Charuco pose estimation easily. As in the GridBoard, the coordinate system
of the CharucoBoard is in the plane of the board with the Z axis pointing out, and centered in the bottom left corner of the board.

The function is estimatePoseCharucoBoard():

``` c++
    estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);
```

- The charucoCorners and charucoIds parameters are the detected charuco corners from the interpolateCornersCharuco()
function.
- The third parameter is the CharucoBoard object.
The cameraMatrix and distCoeffs are the camera calibration parameters which are necessary for pose estimation.
- Finally, the rvec and tvec parameters are the output pose of the Charuco Board.
- The function returns true if the pose was correctly estimated and false otherwise. The main reason of failing is that there are
not enough corners for pose estimation or they are in the same line.

The axis can be drawn using drawAxis() to check the pose is correctly estimated. The result would be (X:red, Y:green, Z:blue):

![Charuco Board Axis](images/chaxis.png)

A full example of Charuco detection with pose estimation (see a more detailed example in board_detector_charuco.cpp):

``` c++
    cv::VideoCapture inputVideo;
    inputVideo.open(0);

    cv::Mat cameraMatrix, distCoeffs;
    // camera parameters are read from somewhere
    readCameraParameters(cameraMatrix, distCoeffs);

    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, DICT_6X6_250);

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
    
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
    
        if (ids.size() > 0) {
            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;
            interpolateCornersCharuco(corners, ids, image, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs);
            if(charucoIds.size() > 0) {
                drawDetectedCornersCharuco(imageCopy, imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
                cv::Mat rvec, tvec;
                bool valid = estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);
                if(valid)
                    cv::aruco::drawAxis(imageCopy, imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
            }
        }
    
        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(waitTime);
        if (key == 27)
            break;
    }
```


