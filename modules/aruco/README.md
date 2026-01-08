ArUco Marker Detection
======================

**ArUco**

![markers](https://github.com/opencv/opencv_contrib/assets/810997/8d587456-f27f-49e4-9540-28a0477d43fc)

ArUco markers are easy to detect pattern grids that yield up to 1024 different patterns. They were built for augmented reality and later used for camera calibration. Since the grid uniquely orients the square, the detection algorithm can determing the pose of the grid.

**ChArUco**

![screen_charuco](https://github.com/opencv/opencv_contrib/assets/810997/64610da1-ee06-406c-a19b-006b02ac44fd)


ArUco markers were improved by interspersing them inside a checkerboard called ChArUco. Checkerboard corner intersections provide more stable corners because the edge location bias on one square is countered by the opposite edge orientation in the connecting square. By interspersing ArUco markers inside the checkerboard, each checkerboard corner gets a label which enables it to be used in complex calibration or pose scenarios where you cannot see all the corners of the checkerboard.

The smallest ChArUco board is 5 checkers and 4 markers called a "Diamond Marker".