Saliency categories
============================

.. highlight:: cpp

Base classes which give a general interface for each specialized type of saliency algorithm and provide utility methods for each algorithm in its class.

StaticSaliency
--------------

.. ocv:class:: StaticSaliency

StaticSaliency class::

   class CV_EXPORTS StaticSaliency : public virtual Saliency
   {
    public:

     bool computeBinaryMap( const Mat& saliencyMap, Mat& binaryMap );

    protected:
     virtual bool computeSaliencyImpl( const InputArray image, OutputArray saliencyMap ) = 0;
   };

StaticSaliency::computeBinaryMap
--------------------------------

This function perform a binary map of given saliency map. This is obtained in this way:

In a first step, to improve the definition of interest areas and facilitate identification of targets, a segmentation
by clustering is performed, using *K-means algorithm*. Then, to gain a binary representation of clustered saliency map, since values of the map can vary according to the characteristics of frame under analysis, it is not convenient to use a fixed threshold.
So, *Otsu’s algorithm* is used, which assumes that the image to be thresholded contains two classes of pixels or bi-modal histograms (e.g. foreground and back-ground pixels); later on, the algorithm calculates the optimal threshold separating those two classes, so that their
intra-class variance is minimal.

.. ocv:function:: bool computeBinaryMap( const Mat& saliencyMap, Mat& binaryMap )

   :param saliencyMap: the saliency map obtained through one of the specialized algorithms
   :param binaryMap: the binary map


MotionSaliency
--------------

.. ocv:class:: MotionSaliency

MotionSaliency class::

   class CV_EXPORTS MotionSaliency : public virtual Saliency
   {


    protected:
     virtual bool computeSaliencyImpl( const InputArray image, OutputArray saliencyMap ) = 0;
   };


Objectness
----------

.. ocv:class:: Objectness

Objectness class::

   class CV_EXPORTS Objectness : public virtual Saliency
   {

    protected:
     virtual bool computeSaliencyImpl( const InputArray image, OutputArray saliencyMap ) = 0;
   };



