Binary descriptors for lines extracted from an image
====================================================

.. highlight:: cpp


Introduction
------------


One of the most challenging activities in computer vision is the extraction of useful information from a given image. Such information, usually comes in the form of points that preserve some kind of property (for instance, they are scale-invariant) and are actually representative of input image.

The goal of this module is seeking a new kind of representative information inside an image and providing the functionalities for its extraction and representation. In particular, differently from previous methods for detection of relevant elements inside an image, lines are extracted in place of points; a new class is defined ad hoc to summarize a line's properties, for reuse and plotting purposes.


A class to represent a line: KeyLine
------------------------------------

As aformentioned, it is been necessary to design a class that fully stores the information needed to characterize completely a line and plot it on image it was extracted from, when required.

*KeyLine* class has been created for such goal; it is mainly inspired to Feature2d's KeyPoint class, since KeyLine shares some of *KeyPoint*'s fields, even if a part of them assumes a different meaning, when speaking about lines.
In particular: 

* the *class_id* field is used to gather lines extracted from different octaves which refer to same line inside original image (such lines and the one they represent in original image share the same *class_id* value)
* the *angle* field represents line's slope with respect to (positive) X axis
* the *pt* field represents line's midpoint
* the *response* field is computed as the ratio between the line's length and maximum between image's width and height
* the *size* field is the area of the smallest rectangle containing line

Apart from fields inspired to KeyPoint class, KeyLines stores information about extremes of line in original image and in octave it was extracted from, about line's length and number of pixels it covers. Code relative to KeyLine class is reported in the following snippet:

.. ocv:class:: KeyLine 

::

  class CV_EXPORTS_W KeyLine
    {
    public:
        /* orientation of the line */
        float angle;

        /* object ID, that can be used to cluster keylines by the line they represent */
        int class_id;

        /* octave (pyramid layer), from which the keyline has been extracted */
        int octave;

        /* coordinates of the middlepoint */
        Point pt;

        /* the response, by which the strongest keylines have been selected.
          It's represented by the ratio between line's length and maximum between
          image's width and height */
        float response;

        /* minimum area containing line */
        float size;

        /* lines's extremes in original image */
        float startPointX;
        float startPointY;
        float endPointX;
        float endPointY;

        /* line's extremes in image it was extracted from */
        float sPointInOctaveX;
        float sPointInOctaveY;
        float ePointInOctaveX;
        float ePointInOctaveY;

        /* the length of line */
        float lineLength;

        /* number of pixels covered by the line */
        unsigned int numOfPixels;

        /* constructor */
        KeyLine(){}
    };
