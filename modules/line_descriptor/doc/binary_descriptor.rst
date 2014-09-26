.. _binary_descriptor:

BinaryDescriptor Class
======================

.. highlight:: cpp

BinaryDescriptor Class implements both functionalities for detection of lines and computation of their binary descriptor. Class' interface is mainly based on the ones of classical detectors and extractors, such as Feature2d's `FeatureDetector <http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=featuredetector#featuredetector>`_ and `DescriptorExtractor <http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_extractors.html?highlight=extractor#DescriptorExtractor : public Algorithm>`_.
Retrieved information about lines is stored in *KeyLine* objects.


BinaryDescriptor::Params
-----------------------------------------------------------------------

.. ocv:struct:: BinaryDescriptor::Params

List of BinaryDescriptor parameters::

    struct CV_EXPORTS_W_SIMPLE Params{
            CV_WRAP Params();

            /* the number of image octaves (default = 1) */
            CV_PROP_RW int  numOfOctave_;

            /* the width of band; (default = 7) */
            CV_PROP_RW int  widthOfBand_;

            /* image's reduction ratio in construction of Gaussian pyramids (default = 2) */
            CV_PROP_RW int reductionRatio;

            /* read parameters from a FileNode object and store them (struct function) */
            void read( const FileNode& fn );

            /* store parameters to a FileStorage object (struct function) */
            void write( FileStorage& fs ) const;

        };


BinaryDescriptor::BinaryDescriptor
----------------------------------

Constructor

.. ocv:function:: bool BinaryDescriptor::BinaryDescriptor( const BinaryDescriptor::Params &parameters = BinaryDescriptor::Params() )

    :param parameters: configuration parameters :ocv:struct:`BinaryDescriptor::Params`

If no argument is provided, constructor sets default values (see comments in the code snippet in previous section). Default values are strongly reccomended.


BinaryDescriptor::getNumOfOctaves
---------------------------------

Get current number of octaves

.. ocv:function:: int BinaryDescriptor::getNumOfOctaves()


BinaryDescriptor::setNumOfOctaves
---------------------------------

Set number of octaves

.. ocv:function:: void BinaryDescriptor::setNumOfOctaves( int octaves )

    :param octaves: number of octaves


BinaryDescriptor::getWidthOfBand
--------------------------------

Get current width of bands

.. ocv:function:: int BinaryDescriptor::getWidthOfBand()


BinaryDescriptor::setWidthOfBand
--------------------------------

Set width of bands

.. ocv:function:: void BinaryDescriptor::setWidthOfBand( int width )

    :param width: width of bands

BinaryDescriptor::getReductionRatio
-----------------------------------

Get current reduction ratio (used in Gaussian pyramids)

.. ocv:function:: int BinaryDescriptor::getReductionRatio()


BinaryDescriptor::setReductionRatio
-----------------------------------

Set reduction ratio (used in Gaussian pyramids)

.. ocv:function:: void BinaryDescriptor::setReductionRatio( int rRatio )

    :param rRatio: reduction ratio


BinaryDescriptor::createBinaryDescriptor
----------------------------------------

Create a BinaryDescriptor object with default parameters (or with the ones provided) and return a smart pointer to it

.. ocv:function:: Ptr<BinaryDescriptor> BinaryDescriptor::createBinaryDescriptor()
.. ocv:function:: Ptr<BinaryDescriptor> BinaryDescriptor::createBinaryDescriptor( Params parameters )


BinaryDescriptor::operator()
----------------------------

Define operator '()' to perform detection of KeyLines and computation of descriptors in a row.

.. ocv:function:: void BinaryDescriptor::operator()( InputArray image, InputArray mask, vector<KeyLine>& keylines, OutputArray descriptors, bool useProvidedKeyLines=false, bool returnFloatDescr ) const

	:param image: input image

	:param mask: mask matrix to select which lines in KeyLines must be accepted among the ones extracted (used when *keylines* is not empty)

	:param keylines: vector that contains input lines (when filled, the detection part will be skipped and input lines will be passed as input to the algorithm computing descriptors)

	:param descriptors: matrix that will store final descriptors

	:param useProvidedKeyLines: flag (when set to true, detection phase will be skipped and only computation of descriptors will be executed, using lines provided in *keylines*)

	:param returnFloatDescr: flag (when set to true, original non-binary descriptors are returned)


BinaryDescriptor::read
----------------------

Read parameters from a FileNode object and store them

.. ocv:function:: void BinaryDescriptor::read( const FileNode& fn )

	:param fn: source FileNode file


BinaryDescriptor::write
-----------------------

Store parameters to a FileStorage object

.. ocv:function:: void BinaryDescriptor::write( FileStorage& fs ) const

	:param fs: output FileStorage file


BinaryDescriptor::defaultNorm
-----------------------------

Return norm mode

.. ocv:function:: int BinaryDescriptor::defaultNorm() const


BinaryDescriptor::descriptorType
--------------------------------

Return data type

.. ocv:function:: int BinaryDescriptor::descriptorType() const


BinaryDescriptor::descriptorSize
--------------------------------

Return descriptor size

.. ocv:function:: int BinaryDescriptor::descriptorSize() const


BinaryDescriptor::detect
------------------------

Requires line detection (for one or more images)

.. ocv:function:: void detect( const Mat& image, vector<KeyLine>& keylines, Mat& mask=Mat() )
.. ocv:function:: void detect( const vector<Mat>& images, vector<vector<KeyLine> >& keylines, vector<Mat>& masks=vector<Mat>() ) const

	:param image: input image

	:param images: input images

	:param keylines: vector or set of vectors that will store extracted lines for one or more images

	:param mask: mask matrix to detect only KeyLines of interest

	:param masks: vector of mask matrices to detect only KeyLines of interest from each input image


BinaryDescriptor::compute
-------------------------

Requires descriptors computation (for one or more images)

.. ocv:function:: void compute( const Mat& image, vector<KeyLine>& keylines, Mat& descriptors, bool returnFloatDescr ) const
.. ocv:function:: void compute( const vector<Mat>& images, vector<vector<KeyLine> >& keylines, vector<Mat>& descriptors, bool returnFloatDescr ) const

	:param image: input image

	:param images: input images

	:param keylines: vector or set of vectors containing lines for which descriptors must be computed

	:param mask: mask to select for which lines, among the ones provided in input, descriptors must be computed

	:param masks: set of masks to select for which lines, among the ones provided in input, descriptors must be computed

	:param returnFloatDescr: flag (when set to true, original non-binary descriptors are returned)


Related pages
-------------

* :ref:`line_descriptor`
* :ref:`matching`
* :ref:`drawing`
