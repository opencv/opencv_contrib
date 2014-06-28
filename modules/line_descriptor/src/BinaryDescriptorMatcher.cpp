#include "precomp.hpp"

using namespace cv;

/* constructor with smart pointer */
Ptr<BinaryDescriptorMatcher> BinaryDescriptorMatcher::createBinaryDescriptorMatcher()
{
    return Ptr<BinaryDescriptorMatcher>(new BinaryDescriptorMatcher());
}

void BinaryDescriptorMatcher::read( const FileNode& ){}
void BinaryDescriptorMatcher::write( FileStorage& ) const{}

/* for every input descriptor, find the best matching one (for a pair of images) */
void BinaryDescriptorMatcher::match( const Mat& queryDescriptors,
                                     const Mat& trainDescriptors,
                                     std::vector<DMatch>& matches,
                                     const Mat& mask ) const
{
    /* create a new mihasher object */
    Mihasher *mh = new Mihasher(256, 32);

    /* populate mihasher */
    cv::Mat copy = trainDescriptors.clone();
    mh->populate(copy, copy.rows, copy.cols);
    mh->setK(1);

    /* prepare structures for query */
    UINT32 *results = new UINT32[queryDescriptors.rows];
    UINT32 * numres = new UINT32[(256+1)*(queryDescriptors.rows)];

    /* execute query */
    mh->batchquery(results,
                   numres,
                   queryDescriptors,
                   queryDescriptors.rows,
                   queryDescriptors.cols);

    /* compose matches */
    for(size_t counter = 0; counter<queryDescriptors.rows; counter++)
    {
        /* create a DMatch object if required by mask of if there is
           no mask at all */
        if( mask.empty() || (!mask.empty() && mask.at<int>(counter)!=0))
        {
            DMatch dm;
            dm.queryIdx = counter;
            dm.trainIdx = results[counter];
            dm.imgIdx = 0;
            dm.distance = numres[counter];

            matches.push_back(dm);
        }
    }


}
