Line Features Tutorial {#tutorial_line_descriptor_main}
======================

In this tutorial it will be shown how to:

-   Use the *BinaryDescriptor* interface to extract the lines and store them in *KeyLine* objects
-   Use the same interface to compute descriptors for every extracted line
-   Use the *BynaryDescriptorMatcher* to determine matches among descriptors obtained from different
    images

Lines extraction and descriptors computation
--------------------------------------------

In the following snippet of code, it is shown how to detect lines from an image. The LSD extractor
is initialized with *LSD\_REFINE\_ADV* option; remaining parameters are left to their default
values. A mask of ones is used in order to accept all extracted lines, which, at the end, are
displayed using random colors for octave 0.

@includelineno line_descriptor/samples/lsd_lines_extraction.cpp

This is the result obtained from the famous cameraman image:

![alternate text](pics/lines_cameraman_edl.png)

Another way to extract lines is using *LSDDetector* class; such class uses the LSD extractor to
compute lines. To obtain this result, it is sufficient to use the snippet code seen above, just
modifying it by the rows

@code{.cpp}
// create a pointer to an LSDDetector object
Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();

// compute lines
std::vector<KeyLine> keylines;
lsd->detect( imageMat, keylines, mask );
@endcode

Here's the result returned by LSD detector again on cameraman picture:

![alternate text](pics/cameraman_lines2.png)

Once keylines have been detected, it is possible to compute their descriptors as shown in the
following:

@includelineno line_descriptor/samples/compute_descriptors.cpp

Matching among descriptors
--------------------------

If we have extracted descriptors from two different images, it is possible to search for matches
among them. One way of doing it is matching exactly a descriptor to each input query descriptor,
choosing the one at closest distance:

@includelineno line_descriptor/samples/matching.cpp

Sometimes, we could be interested in searching for the closest *k* descriptors, given an input one.
This requires modifying previous code slightly:

@code{.cpp}
// prepare a structure to host matches
std::vector<std::vector<DMatch> > matches;

// require knn match
bdm->knnMatch( descr1, descr2, matches, 6 );
@endcode

In the above example, the closest 6 descriptors are returned for every query. In some cases, we
could have a search radius and look for all descriptors distant at the most *r* from input query.
Previous code must be modified like:

@code{.cpp}
// prepare a structure to host matches
std::vector<std::vector<DMatch> > matches;

// compute matches
bdm->radiusMatch( queries, matches, 30 );
@endcode

Here's an example of matching among descriptors extracted from original cameraman image and its
downsampled (and blurred) version:

![alternate text](pics/matching2.png)

Querying internal database
--------------------------

The *BynaryDescriptorMatcher* class owns an internal database that can be populated with
descriptors extracted from different images and queried using one of the modalities described in the
previous section. Population of internal dataset can be done using the *add* function; such function
doesn't directly add new data to the database, but it just stores it them locally. The real update
happens when the function *train* is invoked or when any querying function is executed, since each of
them invokes *train* before querying. When queried, internal database not only returns required
descriptors, but for every returned match, it is able to tell which image matched descriptor was
extracted from. An example of internal dataset usage is described in the following code; after
adding locally new descriptors, a radius search is invoked. This provokes local data to be
transferred to dataset which in turn, is then queried.

@includelineno line_descriptor/samples/radius_matching.cpp
