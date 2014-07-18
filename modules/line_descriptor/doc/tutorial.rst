Tracking API
============

.. highlight:: cpp


Long-term optical tracking API
------------------------------
Long-term optical tracking is one of most important issue for many computer vision applications in real world scenario.
The development in this area is very fragmented and this API is an unique interface useful for plug several algorithms and compare them.


This algorithms start from a bounding box of the target and with their internal representation they avoid the drift during the tracking.
These long-term trackers are able to evaluate online the quality of the location of the target in the new frame, without ground truth.

There are three main components: the TrackerSampler, the TrackerFeatureSet and the TrackerModel. The first component is the object that computes the patches over the frame based on the last target location.
The TrackerFeatureSet is the class that manages the Features, is possible plug many kind of these (HAAR, HOG, LBP, Feature2D, etc).
The last component is the internal representation of the target, it is the appearence model. It stores all state candidates and compute the trajectory (the most likely target states). The class TrackerTargetState represents a possible state of the target.
The TrackerSampler and the TrackerFeatureSet are the visual representation of the target, instead the TrackerModel is the statistical model.


UML design:
-----------


**Tracker diagram**

.. image:: pics/Trackerline.png
   :width: 80%
   :alt: Tracker diagram
   :align: center



