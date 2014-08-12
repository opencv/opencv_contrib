Static Saliency algorithms
============================

.. highlight:: cpp

Algorithms belonging to this category, exploit different image features that allow to detect salient objects in a non dynamic scenarios.

Presently, the Spectral Residual approach [SR]_ has been implemented.

.. [SR] Hou, Xiaodi, and Liqing Zhang. "Saliency detection: A spectral residual approach." Computer Vision and Pattern Recognition, 2007. CVPR'07. IEEE Conference on. IEEE, 2007.


SpectralResidual
----------------

Starting from the principle of natural image statistics, this method simulate the behavior of pre-attentive visual search. The algorithm analyze the log spectrum of each image and obtain the spectral residual. Then transform the spectral residual to spatial domain to obtain the saliency map, which suggests the positions of proto-objects.

.. ocv:class:: StaticSaliencySpectralResidual

Implementation of SpectralResidual from :ocv:class:`StaticSaliency`::

   class CV_EXPORTS StaticSaliencySpectralResidual : public StaticSaliency
   {
    public:

     StaticSaliencySpectralResidual();
     ~StaticSaliencySpectralResidual();

     typedef Ptr<Size> (Algorithm::*SizeGetter)();
     typedef void (Algorithm::*SizeSetter)( const Ptr<Size> & );

     Ptr<Size> getWsize();
     void setWsize( const Ptr<Size> &newSize );

     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;

    protected:
     bool computeSaliencyImpl( const InputArray src, OutputArray dst );

   };


StaticSaliencySpectralResidual::StaticSaliencySpectralResidual
--------------------------------------------------------------

Constructor

.. ocv:function:: StaticSaliencySpectralResidual::StaticSaliencySpectralResidual()


StaticSaliencySpectralResidual::getWsize
----------------------------------------
Return the resized image size.

.. ocv:function:: Ptr<Size> StaticSaliencySpectralResidual::getWsize()


StaticSaliencySpectralResidual::setWsize
----------------------------------------
Set the dimension to which the image should be resized.

.. ocv:function:: StaticSaliencySpectralResidual::void setWsize( const Ptr<Size> &newSize )

    :param newSize: dimension to which the image should be resized
    
    
StaticSaliencySpectralResidual::computeSaliency
-----------------------------------------------
    
Performs all the operations and calls all internal functions necessary for the accomplishment of spectral residual saliency map.

.. ocv:function:: bool StaticSaliencySpectralResidual::computeSaliency( const InputArray image, OutputArray saliencyMap )
 
   :param image: input image. According to the needs of this specialized saliency algorithm, the param image is a single *Mat*
   :param saliencyMap: saliency map. According to the result given by this specialized saliency algorithm, the saliency map is a single *Mat*
    




