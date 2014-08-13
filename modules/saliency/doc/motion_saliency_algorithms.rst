Motion Saliency Algorithms
============================

.. highlight:: cpp

Algorithms belonging to this category, are particularly focused to detect salient objects over time (hence also over frame), then there is a temporal component sealing cosider that allows to detect "moving" objects as salient, meaning therefore also the more general sense of detection the changes in the scene.

Presently, the Fast Self-tuning Background Subtraction Algorithm [BinWangApr2014]_ has been implemented.

.. [BinWangApr2014] B. Wang and P. Dudek "A Fast Self-tuning Background Subtraction Algorithm", in proc of IEEE Workshop on Change Detection, 2014

MotionSaliencyBinWangApr2014
----------------------------
.. ocv:class:: MotionSaliencyBinWangApr2014

Implementation of MotionSaliencyBinWangApr2014 from :ocv:class:`MotionSaliency`::

   class CV_EXPORTS MotionSaliencyBinWangApr2014 : public MotionSaliency
   {
    public:
      MotionSaliencyBinWangApr2014();
      ~MotionSaliencyBinWangApr2014();

     void setImagesize( int W, int H );
     bool init();

    protected:
     bool computeSaliencyImpl( const InputArray image, OutputArray saliencyMap );
  
  };
	  
  
MotionSaliencyBinWangApr2014::MotionSaliencyBinWangApr2014
----------------------------------------------------------

Constructor

.. ocv:function:: MotionSaliencyBinWangApr2014::MotionSaliencyBinWangApr2014()


MotionSaliencyBinWangApr2014::setImagesize
------------------------------------------
This is a utility function that allows to set the correct size (taken from the input image) in the corresponding variables that will be used to size the data structures of the algorithm.  

.. ocv:function:: void MotionSaliencyBinWangApr2014::setImagesize( int W, int H )

   :param W: width of input image
   :param H: height of input image
    
MotionSaliencyBinWangApr2014::init
-----------------------------------
This function allows the correct initialization of all data structures that will be used by the algorithm.

.. ocv:function:: bool MotionSaliencyBinWangApr2014::init()


MotionSaliencyBinWangApr2014::computeSaliencyImpl
-------------------------------------------------
Performs all the operations and calls all internal functions necessary for the accomplishment of the Fast Self-tuning Background Subtraction Algorithm algorithm.

.. ocv:function:: bool MotionSaliencyBinWangApr2014::computeSaliencyImpl( const InputArray image, OutputArray saliencyMap )

   :param image: input image. According to the needs of this specialized algorithm, the param image is a single *Mat*.
   :param saliencyMap: Saliency Map. Is a binarized map that, in accordance with the nature of the algorithm, highlights the moving objects or areas of change in the scene.
    The saliency map is given by a single *Mat* (one for each frame of an hypothetical video stream).

