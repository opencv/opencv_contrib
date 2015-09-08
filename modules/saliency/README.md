Saliency API, understanding where humans focus given a scene
============================================================

The purpose of this module is to create, group and make available to the users, different saliency algorithms, belonging to different categories.

0. The EdgeBoxes algorithm (Dollar et al, 2014) utilizes edge information in an image to propose a set of bounding boxes which enclose objects. The algorithm searches for windows with fully-enclosed edge-groups, and penalizes windows with edge-groups which cross the boundary of the box.
0. This is an initial implementation in OpenCV

Usage
------

Using the EdgeBoxes algorithm requires providing an image and selecting an edge-finding algorithm, or providing an edge-only image. The class will return a list of boxes (defined by four points of a Vec4i), and their associated "objectness". Objectness is scored between 0 and 1, with 1 being the highest score. Please see the syntax below:

```c++

Mat edgeImage;
edgeImage = imread("C:/edge_circle.png", 0);
edgeImage.convertTo(edgeImage, CV_64F,1/255.0f);

ObjectnessEdgeBoxes oeb;
std::vector<Vec4i> boxList;
std::vector<double> scoreList;
oeb.getBoxScores(edgeImage, orientationImage, boxList, scoreList);


```

Test Cases
----------

You can run a test case by building saliency/EdgeBoxes/EdgeBoxes.cpp

An example of the object proposals is shown below: 

Inline-style: 
![alt text](https://dl.dropboxusercontent.com/u/110033260/edgeboxes.png "Object Proposal Example")


Working/Not-Working
--------------------

0. Conversion of Edge Image into Orientation Image --tested
0. Clustering of edges into groups --tested
0. Generation of Affinity matrix --tested
0. Generation of supporting data structures 
0. Generation of window list --tested
0. Scoring Function -- working, slow, not verified

