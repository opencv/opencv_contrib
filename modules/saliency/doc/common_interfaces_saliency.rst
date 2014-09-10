Common Interfaces of Saliency
=============================

.. highlight:: cpp


Saliency : Algorithm
--------------------

.. ocv:class:: Saliency

Base abstract class for Saliency algorithms::

   class CV_EXPORTS Saliency : public virtual Algorithm
   {
     public:
 
     virtual ~Saliency();

     static Ptr<Saliency> create( const String& saliencyType );

     bool computeSaliency( const InputArray image, OutputArray saliencyMap );

     String getClassName() const;
   };


Saliency::create
----------------

Creates a specialized saliency algorithm by its name.

.. ocv:function::  static Ptr<Saliency> Saliency::create( const String& saliencyType )

   :param saliencyType: saliency Type

The following saliency types are now supported:

* ``"SPECTRAL_RESIDUAL"`` -- :ocv:class:`StaticSaliencySpectralResidual`

* ``"BING"`` -- :ocv:class:`ObjectnessBING`


Saliency::computeSaliency
-------------------------

Performs all the operations, according to the specific algorithm created, to obtain the saliency map.

.. ocv:function:: bool Saliency::computeSaliency( const InputArray image, OutputArray saliencyMap )
 
   :param image: image or set of input images. According to InputArray proxy and to the needs of different algorithms (currently plugged),  the param image may be *Mat* or *vector<Mat>*
   :param saliencyMap: saliency map. According to OutputArray proxy and to the results given by different algorithms (currently plugged), the saliency map may be a *Mat* or *vector<Vec4i>* (BING results).

Saliency::getClassName
----------------------

Get the name of the specific Saliency Algorithm.

.. ocv:function:: String Saliency::getClassName() const








