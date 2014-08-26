Dense Optical Flow
===================

Dense optical flow algorithms compute motion for each point

calcOpticalFlowSF
-----------------
Calculate an optical flow using "SimpleFlow" algorithm.

.. ocv:function:: void calcOpticalFlowSF( InputArray from, InputArray to, OutputArray flow, int layers, int averaging_block_size, int max_flow )

.. ocv:function:: calcOpticalFlowSF( InputArray from, InputArray to, OutputArray flow, int layers, int averaging_block_size, int max_flow, double sigma_dist, double sigma_color, int postprocess_window, double sigma_dist_fix, double sigma_color_fix, double occ_thr, int upscale_averaging_radius, double upscale_sigma_dist, double upscale_sigma_color, double speed_up_thr )

    :param prev: First 8-bit 3-channel image.

    :param next: Second 8-bit 3-channel image of the same size as ``prev``

    :param flow: computed flow image that has the same size as ``prev`` and type ``CV_32FC2``

    :param layers: Number of layers

    :param averaging_block_size: Size of block through which we sum up when calculate cost function for pixel

    :param max_flow: maximal flow that we search at each level

    :param sigma_dist: vector smooth spatial sigma parameter

    :param sigma_color: vector smooth color sigma parameter

    :param postprocess_window: window size for postprocess cross bilateral filter

    :param sigma_dist_fix: spatial sigma for postprocess cross bilateralf filter

    :param sigma_color_fix: color sigma for postprocess cross bilateral filter

    :param occ_thr: threshold for detecting occlusions

    :param upscale_averaging_radius: window size for bilateral upscale operation

    :param upscale_sigma_dist: spatial sigma for bilateral upscale operation

    :param upscale_sigma_color: color sigma for bilateral upscale operation

    :param speed_up_thr: threshold to detect point with irregular flow - where flow should be recalculated after upscale

See [Tao2012]_. And site of project - http://graphics.berkeley.edu/papers/Tao-SAN-2012-05/.

.. note::

   * An example using the simpleFlow algorithm can be found at samples/simpleflow_demo.cpp


optflow::OpticalFlowDeepFlow
-----------------------------
DeepFlow optical flow algorithm implementation.

.. ocv:class:: optflow::OpticalFlowDeepFlow: public DenseOpticalFlow


  The class implements the DeepFlow optical flow algorithm described in [Weinzaepfel2013]_ . See also http://lear.inrialpes.fr/src/deepmatching/ .

  Parameters - class fields - that may be modified after creating a class instance:

    .. ocv:member:: float alpha

    Smoothness assumption weight

    .. ocv:member:: float delta

    Color constancy assumption weight

    .. ocv:member:: float gamma

    Gradient constancy weight
        
    .. ocv:member:: float sigma

    Gaussian smoothing parameter
        
    .. ocv:member:: int minSize

    Minimal dimension of an image in the pyramid (next, smaller images in the pyramid are
    generated until one of the dimensions reaches this size)
        
    .. ocv:member:: float downscaleFactor

    Scaling factor in the image pyramid (must be < 1) 
        
    .. ocv:member:: int fixedPointIterations

    How many iterations on each level of the pyramid
        
    .. ocv:member:: int sorIterations

    Iterations of Succesive Over-Relaxation (solver)
        
    .. ocv:member:: float omega

    Relaxation factor in SOR


optflow::createOptFlow_DeepFlow
---------------------------------
Create an OpticalFlowDeepFlow instance

  .. ocv:function:: Ptr<DenseOpticalFlow> optflow::createOptFlow_DeepFlow()

  Returns a pointer to a ``DenseOpticalFlow`` instance - see ``DenseOpticalFlow::calc()``


  

        

.. [Tao2012] Michael Tao, Jiamin Bai, Pushmeet Kohli and Sylvain Paris. SimpleFlow: A Non-iterative, Sublinear Optical Flow Algorithm. Computer Graphics Forum (Eurographics 2012)
.. [Weinzaepfel2013] P. Weinzaepfel, J. Revaud, Z. Harchaoui, and C. Schmid, “DeepFlow: Large Displacement Optical Flow with Deep Matching,” 2013 IEEE Int. Conf. Comput. Vis., pp. 1385–1392, Dec. 2013.
