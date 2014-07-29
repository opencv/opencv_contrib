.. _line_descriptor:

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


Computation of binary descriptors
---------------------------------

To obtatin a binary descriptor representing a certain line detected from a certain octave of an image, we first compute a non-binary descriptor as described in [LBD]_. Such algorithm works on lines extracted using EDLine detector, as explained in [EDL]_. Given a line, we consider a rectangular region centered at it and called *line support region (LSR)*. Such region is divided into a set of bands :math:`\{B_1, B_2, ..., B_m\}`, whose length equals the one of line.

If we indicate with :math:`\bf{d}_L` the direction of line, the orthogonal and clockwise direction to line :math:`\bf{d}_{\perp}` can be determined; these two directions, are used to construct a reference frame centered in the middle point of line. The gradients of pixels :math:`\bf{g'}` inside LSR can be projected to the newly determined frame, obtaining their local equivalent :math:`\bf{g'} = (\bf{g}^T \cdot \bf{d}_{\perp}, \bf{g}^T \cdot \bf{d}_L)^T \triangleq (\bf{g'}_{d_{\perp}}, \bf{g'}_{d_L})^T`.

Later on, a Gaussian function is applied to all LSR's pixels along :math:`\bf{d}_\perp` direction; first, we assign a global weighting coefficient :math:`f_g(i) = (1/\sqrt{2\pi}\sigma_g)e^{-d^2_i/2\sigma^2_g}` to *i*-th row in LSR, where :math:`d_i` is the distance of *i*-th row from the center row in LSR, :math:`\sigma_g = 0.5(m \cdot w - 1)` and :math:`w` is the width of bands (the same for every band). Secondly, considering a band :math:`B_j` and its neighbor bands :math:`B_{j-1}, B_{j+1}`, we assign a local weighting  :math:`F_l(k) = (1/\sqrt{2\pi}\sigma_l)e^{-d'^2_k/2\sigma_l^2}`, where :math:`d'_k` is the distance of *k*-th row from the center row in :math:`B_j` and :math:`\sigma_l = w`. Using the global and local weights, we obtain, at the same time, the reduction of role played by gradients far from line and of boundary effect, respectively.

Each band :math:`B_j` in LSR has an associated *band descriptor(BD)* which is computed considering previous and next band (top and bottom bands are ignored when computing descriptor for first and last band). Once each band has been assignen its BD, the LBD descriptor of line is simply given by

.. math::
	LBD = (BD_1^T, BD_2^T, ... , BD^T_m)^T.

To compute a band descriptor :math:`B_j`, each *k*-th row in it is considered and the gradients in such row are accumulated:

.. math::
	\begin{matrix} \bf{V1}^k_j = \lambda \sum\limits_{\bf{g}'_{d_\perp}>0}\bf{g}'_{d_\perp}, &  \bf{V2}^k_j = \lambda \sum\limits_{\bf{g}'_{d_\perp}<0} -\bf{g}'_{d_\perp}, \\ \bf{V3}^k_j = \lambda \sum\limits_{\bf{g}'_{d_L}>0}\bf{g}'_{d_L}, & \bf{V4}^k_j = \lambda \sum\limits_{\bf{g}'_{d_L}<0} -\bf{g}'_{d_L}\end{matrix}.

with :math:`\lambda = f_g(k)f_l(k)`.

By stacking previous results, we obtain the *band description matrix (BDM)*

.. math::
	BDM_j = \left(\begin{matrix} \bf{V1}_j^1 & \bf{V1}_j^2 & \ldots & \bf{V1}_j^n \\ \bf{V2}_j^1 & \bf{V2}_j^2 & \ldots & \bf{V2}_j^n \\ \bf{V3}_j^1 & \bf{V3}_j^2 & \ldots & \bf{V3}_j^n \\ \bf{V4}_j^1 & \bf{V4}_j^2 & \ldots & \bf{V4}_j^n \end{matrix} \right) \in \mathbb{R}^{4\times n},

with :math:`n` the number of rows in band :math:`B_j`:

.. math::
	n = \begin{cases} 2w, & j = 1||m; \\ 3w, & \mbox{else}. \end{cases}

Each :math:`BD_j` can be obtained using the standard deviation vector :math:`S_j` and mean vector :math:`M_j` of :math:`BDM_J`. Thus, finally:

.. math::
	LBD = (M_1^T, S_1^T, M_2^T, S_2^T, \ldots, M_m^T, S_m^T)^T \in \mathbb{R}^{8m}


Once the LBD has been obtained, it must be converted into a binary form. For such purpose, we consider 32 possible pairs of BD inside it; each couple of BD is compared bit by bit and comparison generates an 8 bit string. Concatenating 32 comparison strings, we get the 256-bit final binary representation of a single LBD.

References
----------

.. [LBD] Zhang, Lilian, and Reinhard Koch. *An efficient and robust line segment matching approach based on LBD descriptor and pairwise geometric consistency*, Journal of Visual Communication and Image Representation 24.7 (2013): 794-805.

.. [EDL] Von Gioi, R. Grompone, et al. *LSD: A fast line segment detector with a false detection control*, IEEE Transactions on Pattern Analysis and Machine Intelligence 32.4 (2010): 722-732.


Summary
-------
		
.. toctree::
	:maxdepth: 2

	binary_descriptor
	LSDDetector
	matching
	drawing_functions
	tutorial

