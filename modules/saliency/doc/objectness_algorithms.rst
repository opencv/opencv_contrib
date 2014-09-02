Objectness Algorithms
============================

.. highlight:: cpp

Objectness is usually represented as a value which reflects how likely an image window covers an object of any category. Algorithms belonging to this category, avoid making decisions early on, by proposing a small number of category-independent proposals, that are expected to cover all objects in an image. Being able to perceive objects before identifying them is closely related to bottom up visual attention (saliency)

Presently, the Binarized normed gradients algorithm [BING]_ has been implemented.

.. [BING] Cheng, Ming-Ming, et al. "BING: Binarized normed gradients for objectness estimation at 300fps." IEEE CVPR. 2014.

ObjectnessBING
--------------
.. ocv:class:: ObjectnessBING

Implementation of BING from :ocv:class:`Objectness`::

   class CV_EXPORTS ObjectnessBING : public Objectness
   {
    public:

     ObjectnessBING();
     ~ObjectnessBING();

     void read();
     void write() const;

     vector<float> getobjectnessValues();
     void setTrainingPath( string trainingPath );
     void setBBResDir( string resultsDir );

   protected:
     bool computeSaliencyImpl( const InputArray src, OutputArray dst );

   };
   

ObjectnessBING::ObjectnessBING
------------------------------

Constructor

.. ocv:function:: ObjectnessBING::ObjectnessBING()


ObjectnessBING::getobjectnessValues
-----------------------------------
Return the list of the rectangles' objectness value, in the same order as the  *vector<Vec4i> objectnessBoundingBox* returned by the algorithm (in computeSaliencyImpl function).
The bigger value these scores are, it is more likely to be an object window.

.. ocv:function:: vector<float> ObjectnessBING::getobjectnessValues()


ObjectnessBING::setTrainingPath
--------------------------------
This is a utility function that allows to set the correct path from which the algorithm will load the trained model.

.. ocv:function:: void ObjectnessBING::setTrainingPath( string trainingPath )

    :param trainingPath: trained model path


ObjectnessBING::setBBResDir
---------------------------
This is a utility function that allows to set an arbitrary path in which the algorithm will save the optional results 
(ie writing on file the total number and the list of rectangles returned by objectess, one for each row).

.. ocv:function:: void ObjectnessBING::setBBResDir( string resultsDir )

    :param setBBResDir: results' folder path


ObjectnessBING::computeSaliencyImpl
-----------------------------------
Performs all the operations and calls all internal functions necessary for the accomplishment of the Binarized normed gradients algorithm.

.. ocv:function:: bool ObjectnessBING::computeSaliencyImpl( const InputArray image, OutputArray objectnessBoundingBox )

   :param image: input image. According to the needs of this specialized algorithm, the param image is a single *Mat*
   :param saliencyMap: objectness Bounding Box vector. According to the result given by this specialized  algorithm, the objectnessBoundingBox is a *vector<Vec4i>*.
    Each bounding box is represented by a *Vec4i* for (minX, minY, maxX, maxY).
    
