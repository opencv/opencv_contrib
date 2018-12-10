Quasi Dense Stereo
======================

**qds**

Quasi Dense Stereo is method for performing dense stereo matching. This module contains a class that implements this process.
The code uses pyramidal Lucas-Kanade with Shi-Tomasi features to get the initial seed correspondences.
Then these seeds are propagated by using mentioned growing scheme.
