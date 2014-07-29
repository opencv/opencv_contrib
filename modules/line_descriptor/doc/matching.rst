.. _matching:

Matching with binary descriptors
================================

.. highlight:: cpp

Once descriptors have been extracted from an image (both they represent lines and points), it becomes interesting to be able to match a descriptor with another one extracted from a different image and representing the same line or point, seen from a differente perspective or on a different scale.
In reaching such goal, the main headache is designing an efficient search algorithm to associate a query descriptor to one extracted from a dataset.
In the following, a matching modality based on *Multi-Index Hashing (MiHashing)* will be described.


Multi-Index Hashing
-------------------

The theory described in this section is based on [MIH]_.
Given a dataset populated with binary codes, each code is indexed *m* times into *m* different hash tables, according to *m* substrings it has been divided into. Thus, given a query code, all the entries close to it at least in one substring are returned by search as *neighbor candidates*. Returned entries are then checked for validity by verifying that their full codes are not distant (in Hamming space) more than *r* bits from query code.
In details, each binary code **h** composed of *b* bits is divided into *m* disjoint substrings :math:`\mathbf{h}^{(1)}, ..., \mathbf{h}^{(m)}`, each with length :math:`\lfloor b/m \rfloor` or :math:`\lceil b/m \rceil` bits. Formally, when two codes **h** and **g** differ by at the most *r* bits, in at the least one of their *m* substrings they differ by at the most :math:`\lfloor r/m \rfloor` bits. In particular, when :math:`||\mathbf{h}-\mathbf{g}||_H \le r` (where :math:`||.||_H` is the Hamming norm), there must exist a substring *k* (with :math:`1 \le k \le m`) such that

.. math::
	||\mathbf{h}^{(k)} - \mathbf{g}^{(k)}||_H \le \left\lfloor \frac{r}{m} \right\rfloor .

That means that if Hamming distance between each of the *m* substring is strictly greater than :math:`\lfloor r/m \rfloor`, then :math:`||\mathbf{h}-\mathbf{g}||_H` must be larger that *r* and that is a contradiction.
If the codes in dataset are divided into *m* substrings, then *m* tables will be built. Given a query **q** with substrings :math:`\{\mathbf{q}^{(i)}\}^m_{i=1}`, *i*-th hash table is searched for entries distant at the most :math:`\lfloor r/m \rfloor` from :math:`\mathbf{q}^{(i)}` and a set of candidates :math:`\mathcal{N}_i(\mathbf{q})` is obtained.
The union of sets :math:`\mathcal{N}(\mathbf{q}) = \bigcup_i \mathcal{N}_i(\mathbf{q})` is a superset of the *r*-neighbors of **q**. Then, last step of algorithm is computing the Hamming distance between **q** and each element in :math:`\mathcal{N}(\mathbf{q})`, deleting the codes that are distant more that *r* from **q**.


BinaryDescriptorMatcher Class
=============================

BinaryDescriptorMatcher Class furnishes all functionalities for querying a dataset provided by user or internal to class (that user must, anyway, populate) on the model of Feature2d's `DescriptorMatcher <http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html?highlight=bfmatcher#descriptormatcher>`_.


BinaryDescriptorMatcher::BinaryDescriptorMatcher
--------------------------------------------------

Constructor. 

.. ocv:function:: BinaryDescriptorMatcher::BinaryDescriptorMatcher()

The BinaryDescriptorMatcher constructed is able to store and manage 256-bits long entries.


BinaryDescriptorMatcher::createBinaryDescriptorMatcher
------------------------------------------------------

Create a BinaryDescriptorMatcher object and return a smart pointer to it.

.. ocv:function:: Ptr<BinaryDescriptorMatcher> BinaryDescriptorMatcher::createBinaryDescriptorMatcher()


BinaryDescriptorMatcher::add
----------------------------

Store locally new descriptors to be inserted in dataset, without updating dataset.

.. ocv:function:: void BinaryDescriptorMatcher::add( const std::vector<Mat>& descriptors )

	:param descriptors: matrices containing descriptors to be inserted into dataset

.. note:: Each matrix *i* in **descriptors** should contain descriptors relative to lines extracted from *i*-th image.


BinaryDescriptorMatcher::train
------------------------------

Update dataset by inserting into it all descriptors that were stored locally by *add* function.

.. ocv:function:: void BinaryDescriptorMatcher::train()

.. note:: Every time this function is invoked, current dataset is deleted and locally stored descriptors are inserted into dataset. The locally stored copy of just inserted descriptors is then removed.


BinaryDescriptorMatcher::clear
------------------------------

Clear dataset and internal data

.. ocv:function:: void BinaryDescriptorMatcher::clear()


BinaryDescriptorMatcher::match
------------------------------

For every input query descriptor, retrieve the best matching one from a dataset provided from user or from the one internal to class

.. ocv:function:: void BinaryDescriptorMatcher::match( const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<DMatch>& matches, const Mat& mask=Mat() ) const
.. ocv:function:: void BinaryDescriptorMatcher::match( const Mat& queryDescriptors, std::vector<DMatch>& matches, const std::vector<Mat>& masks=std::vector<Mat>() )

	:param queryDescriptors: query descriptors
	:param trainDescriptors: dataset of descriptors furnished by user
	:param matches: vector to host retrieved matches
	:param mask: mask to select which input descriptors must be matched to one in dataset
	:param masks: vector of masks to select which input descriptors must be matched to one in dataset (the *i*-th mask in vector indicates whether each input query can be matched with descriptors in dataset relative to *i*-th image)


BinaryDescriptorMatcher::knnMatch
---------------------------------

For every input query descriptor, retrieve the best *k* matching ones from a dataset provided from user or from the one internal to class

.. ocv:function:: void BinaryDescriptorMatcher::knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<std::vector<DMatch> >& matches, int k, const Mat& mask=Mat(), bool compactResult=false ) const
.. ocv:function:: void BinaryDescriptorMatcher::knnMatch( const Mat& queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k, const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false )

	:param queryDescriptors: query descriptors
	:param trainDescriptors: dataset of descriptors furnished by user
	:param matches: vector to host retrieved matches
	:param k: number of the closest descriptors to be returned for every input query
	:param mask: mask to select which input descriptors must be matched to ones in dataset
	:param masks: vector of masks to select which input descriptors must be matched to ones in dataset (the *i*-th mask in vector indicates whether each input query can be matched with descriptors in dataset relative to *i*-th image)
	:param compactResult: flag to obtain a compact result (if true, a vector that doesn't contain any matches for a given query is not inserted in final result)


BinaryDescriptorMatcher::radiusMatch
------------------------------------

For every input query descriptor, retrieve, from a dataset provided from user or from the one internal to class, all the descriptors that are not further than *maxDist* from input query 

.. ocv:function:: void BinaryDescriptorMatcher::radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance, const Mat& mask=Mat(), bool compactResult=false ) const
.. ocv:function:: void BinaryDescriptorMatcher::radiusMatch( const Mat& queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance, const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false )

	:param queryDescriptors: query descriptors
	:param trainDescriptors: dataset of descriptors furnished by user
	:param matches: vector to host retrieved matches
	:param maxDist: search radius
	:param mask: mask to select which input descriptors must be matched to ones in dataset
	:param masks: vector of masks to select which input descriptors must be matched to ones in dataset (the *i*-th mask in vector indicates whether each input query can be matched with descriptors in dataset relative to *i*-th image)
	:param compactResult: flag to obtain a compact result (if true, a vector that doesn't contain any matches for a given query is not inserted in final result)



Related pages
-------------

* :ref:`line_descriptor`
* :ref:`binary_descriptor`
* :ref:`drawing`


References
----------

.. [MIH] Norouzi, Mohammad, Ali Punjani, and David J. Fleet. *Fast search in hamming space with multi-index hashing*, Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
