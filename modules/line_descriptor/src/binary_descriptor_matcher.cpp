/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"

using namespace cv;

/* constructor */
BinaryDescriptorMatcher::BinaryDescriptorMatcher()
{
    dataset = new Mihasher(256, 32);
    nextAddedIndex = 0;
    numImages = 0;
    descrInDS = 0;
}

/* constructor with smart pointer */
Ptr<BinaryDescriptorMatcher> BinaryDescriptorMatcher::createBinaryDescriptorMatcher()
{
    return Ptr<BinaryDescriptorMatcher>(new BinaryDescriptorMatcher());
}

/* store new descriptors to be inserted in dataset */
void BinaryDescriptorMatcher::add( const std::vector<Mat>& descriptors )
{
    for(size_t i = 0; i<descriptors.size(); i++)
    {
        descriptorsMat.push_back(descriptors[i]);

        indexesMap.insert(std::pair<int, int>(nextAddedIndex, numImages));
        nextAddedIndex += descriptors[i].rows;
        numImages++;
    }
}

/* store new descriptors into dataset */
void BinaryDescriptorMatcher::train()
{
    if(!dataset)
        dataset = new Mihasher(256, 32);

    if(descriptorsMat.rows >0)
        dataset->populate(descriptorsMat,
                          descriptorsMat.rows,
                          descriptorsMat.cols);

        descrInDS = descriptorsMat.rows;
        descriptorsMat.release();
}

/* clear dataset and internal data */
void BinaryDescriptorMatcher::clear()
{
    descriptorsMat.release();
    indexesMap.clear();
    dataset = 0;
    nextAddedIndex = 0;
    numImages = 0;
    descrInDS = 0;
}

/* retrieve Hamming distances */
void BinaryDescriptorMatcher::checkKDistances(UINT32 * numres, int k, std::vector<int> & k_distances, int row, int string_length) const
{
    int k_to_found = k;

    UINT32 * numres_tmp = numres + ((string_length+1) * row);
    for(int j = 0; j<(string_length+1)  && k_to_found > 0; j++)
    {
        if((*(numres_tmp+j))>0)
        {
            for(int i = 0; i<(*(numres_tmp+j)) && k_to_found > 0; i++)
            {
                k_distances.push_back(j);
                k_to_found--;
            }
        }
    }
}

/* for every input descriptor,
   find the best matching one (from one image to a set) */
void BinaryDescriptorMatcher::match( const Mat& queryDescriptors,
                                     std::vector<DMatch>& matches,
                                     const std::vector<Mat>& masks )
{
    /* check data validity */
    if(masks.size() !=0 && (int)masks.size() != numImages)
    {
        std::cout << "Error: the number of images in dataset is " <<
                     numImages << " but match function received " <<
                     masks.size() << " masks. Program will be terminated"
                     << std::endl;

        return;
    }

    /* add new descriptors to dataset, if needed */
    train();

    /* set number of requested matches to return for each query */
    dataset->setK(1);

    /* prepare structures for query */
    UINT32 *results = new UINT32[queryDescriptors.rows];
    UINT32 * numres = new UINT32[(256+1)*(queryDescriptors.rows)];

    /* execute query */
    dataset->batchquery(results, numres, queryDescriptors,
                        queryDescriptors.rows, queryDescriptors.cols);

    /* compose matches */
    for(int counter = 0; counter<queryDescriptors.rows; counter++)
    {
            /* create a map iterator */
            std::map<int, int>::iterator itup;

            /* get info about original image of each returned descriptor */
            itup = indexesMap.upper_bound(results[counter] - 1);
            itup--;

            /* data validity check */
            if(!masks.empty() && (masks[itup->second].rows != queryDescriptors.rows
                                  || masks[itup->second].cols !=1))
            {
                std::cout << "Error: mask " << itup->second << " in knnMatch function "
                          << "should have " << queryDescriptors.rows << " and "
                          << "1 column. Program will be terminated"
                          << std::endl;

                CV_Assert(false);
            }

            /* create a DMatch object if required by mask of if there is
               no mask at all */
            else if(masks.empty() || masks[itup->second].at<uchar>(counter) !=0)
            {
                std::vector<int> k_distances;
                checkKDistances(numres, 1, k_distances, counter, 256);

                DMatch dm;
                dm.queryIdx = counter;
                dm.trainIdx = results[counter] - 1;
                dm.imgIdx = itup->second;
                dm.distance = k_distances[0];

                matches.push_back(dm);
            }

    }

    /* delete data */
    delete results;
    delete numres;
}

/* for every input descriptor, find the best matching one (for a pair of images) */
void BinaryDescriptorMatcher::match( const Mat& queryDescriptors,
                                     const Mat& trainDescriptors,
                                     std::vector<DMatch>& matches,
                                     const Mat& mask ) const
{

    /* check data validity */
    if(!mask.empty() && (mask.rows != queryDescriptors.rows && mask.cols != 1))
    {
        std::cout << "Error: input mask should have " <<
                     queryDescriptors.rows << " rows and 1 column. " <<
                     "Program will be terminated" << std::endl;

        return;
    }

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
    mh->batchquery(results, numres, queryDescriptors,
                   queryDescriptors.rows, queryDescriptors.cols);

    /* compose matches */
    for(int counter = 0; counter<queryDescriptors.rows; counter++)
    {
        /* create a DMatch object if required by mask of if there is
           no mask at all */
        if( mask.empty() || (!mask.empty() && mask.at<uchar>(counter)!=0))
        {
            std::vector<int> k_distances;
            checkKDistances(numres, 1, k_distances, counter, 256);

            DMatch dm;
            dm.queryIdx = counter;
            dm.trainIdx = results[counter] - 1;
            dm.imgIdx = 0;
            dm.distance = k_distances[0];

            matches.push_back(dm);
        }
    }

    /* delete data */
    delete mh;
    delete results;
    delete numres;


}

/* for every input descriptor,
   find the best k matching descriptors (for a pair of images) */
void BinaryDescriptorMatcher::knnMatch( const Mat& queryDescriptors,
                                        const Mat& trainDescriptors,
                                        std::vector<std::vector<DMatch> >& matches,
                                        int k,
                                        const Mat& mask,
                                        bool compactResult ) const

{
    /* check data validity */
    if(!mask.empty() && (mask.rows != queryDescriptors.rows || mask.cols != 1))
    {
        std::cout << "Error: input mask should have " <<
                     queryDescriptors.rows << " rows and 1 column. " <<
                     "Program will be terminated" << std::endl;

        return;
    }

    /* create a new mihasher object */
    Mihasher *mh = new Mihasher(256, 32);

    /* populate mihasher */
    cv::Mat copy = trainDescriptors.clone();
    mh->populate(copy, copy.rows, copy.cols);

    /* set K */
    mh->setK(k);

    /* prepare structures for query */
    UINT32 *results = new UINT32[k*queryDescriptors.rows];
    UINT32 * numres = new UINT32[(256+1)*(queryDescriptors.rows)];

    /* execute query */
    mh->batchquery(results, numres, queryDescriptors,
                   queryDescriptors.rows, queryDescriptors.cols);

    /* compose matches */
    int index = 0;
    for(int counter = 0; counter<queryDescriptors.rows; counter++)
    {
        /* initialize a vector of matches */
        std::vector<DMatch> tempVec;

        /* chech whether query should be ignored */
        if(!mask.empty() && mask.at<uchar>(counter) == 0)
        {
            /* if compact result is not requested, add an empty vector */
            if(!compactResult)
                matches.push_back(tempVec);
        }

        /* query matches must be considered */
        else
        {
            std::vector<int> k_distances;
            checkKDistances(numres, k, k_distances, counter, 256);
            for(int j = index; j<index+k; j++)
            {
                DMatch dm;
                dm.queryIdx = counter;
                dm.trainIdx = results[j] - 1;
                dm.imgIdx = 0;
                dm.distance = k_distances[j-index];

                tempVec.push_back(dm);
            }

            matches.push_back(tempVec);
        }

        /* increment pointer */
        index += k;
    }

    /* delete data */
    delete mh;
    delete results;
    delete numres;
}

/* for every input descriptor,
   find the best k matching descriptors (from one image to a set) */
void BinaryDescriptorMatcher::knnMatch( const Mat& queryDescriptors,
                                        std::vector<std::vector<DMatch> >& matches,
                                        int k,
                                        const std::vector<Mat>& masks,
                                        bool compactResult )
{

    /* check data validity */
    if(masks.size() !=0 && (int)masks.size() != numImages)
    {
        std::cout << "Error: the number of images in dataset is " <<
                     numImages << " but knnMatch function received " <<
                     masks.size() << " masks. Program will be terminated"
                     << std::endl;

        return;
    }

    /* add new descriptors to dataset, if needed */
    train();

    /* set number of requested matches to return for each query */
    dataset->setK(k);

    /* prepare structures for query */
    UINT32 *results = new UINT32[k*queryDescriptors.rows];
    UINT32 * numres = new UINT32[(256+1)*(queryDescriptors.rows)];

    /* execute query */
    dataset->batchquery(results, numres, queryDescriptors,
                        queryDescriptors.rows, queryDescriptors.cols);

    /* compose matches */
    int index = 0;
    for(int counter = 0; counter<queryDescriptors.rows; counter++)
    {
        /* create a void vector of matches */
        std::vector<DMatch> tempVector;

        /* loop over k results returned for every query */
        for(int j = index; j<index+k; j++)
        {
            /* retrieve which image returned index refers to */
            int currentIndex = results[j]-1;
            std::map<int, int>::iterator itup;
            itup = indexesMap.upper_bound(currentIndex);
            itup--;

            /* data validity check */
            if(!masks.empty() && (masks[itup->second].rows != queryDescriptors.rows
                                  || masks[itup->second].cols != 1))
            {
                std::cout << "Error: mask " << itup->second << " in knnMatch function "
                          << "should have " << queryDescriptors.rows << " and "
                          << "1 column. Program will be terminated"
                          << std::endl;

                return;
            }

            /* decide if, according to relative mask, returned match should be
               considered */
            else if(masks.size() == 0 || masks[itup->second].at<uchar>(counter) != 0)
            {
                std::vector<int> k_distances;
                checkKDistances(numres, k, k_distances, counter, 256);

                DMatch dm;
                dm.queryIdx = counter;
                dm.trainIdx = results[j] - 1;
                dm.imgIdx = itup->second;
                dm.distance = k_distances[j-index];

                tempVector.push_back(dm);
            }
        }

        /* decide whether temporary vector should be saved */
        if((tempVector.size() == 0 && !compactResult) || tempVector.size()>0)
            matches.push_back(tempVector);

        /* increment pointer */
        index += k;
    }

    /* delete data */
    delete results;
    delete numres;
}

/* for every input desciptor, find all the ones falling in a
   certaing matching radius (for a pair of images) */
void BinaryDescriptorMatcher::radiusMatch( const Mat& queryDescriptors,
                                           const Mat& trainDescriptors,
                                           std::vector<std::vector<DMatch> >& matches,
                                           float maxDistance,
                                           const Mat& mask,
                                           bool compactResult ) const

{

    /* check data validity */
    if(!mask.empty() && (mask.rows != queryDescriptors.rows && mask.cols != 1))
    {
        std::cout << "Error: input mask should have " <<
                     queryDescriptors.rows << " rows and 1 column. " <<
                     "Program will be terminated" << std::endl;

        return;
    }

    /* create a new Mihasher */
    Mihasher* mh = new Mihasher(256, 32);

    /* populate Mihasher */
    Mat copy = queryDescriptors.clone();
    mh->populate(copy, copy.rows, copy.cols);

    /* set K */
    mh->setK(trainDescriptors.rows);

    /* prepare structures for query */
    UINT32 *results = new UINT32[trainDescriptors.rows*queryDescriptors.rows];
    UINT32 * numres = new UINT32[(256+1)*(queryDescriptors.rows)];

    /* execute query */
    mh->batchquery(results, numres, queryDescriptors,
                   queryDescriptors.rows, queryDescriptors.cols);

    /* compose matches */
    int index = 0;
    for (int i = 0; i<queryDescriptors.rows; i++)
    {
        std::vector<int> k_distances;
        checkKDistances(numres, trainDescriptors.rows, k_distances, i, 256);

        std::vector<DMatch> tempVector;
        for(int j = 0; j<index+trainDescriptors.rows; j++)
        {
            if(numres[j] <= maxDistance)
            {
                if(mask.empty() || mask.at<uchar>(i) != 0){
                    DMatch dm;
                    dm.queryIdx = i;
                    dm.trainIdx = results[j] - 1;
                    dm.imgIdx = 0;
                    dm.distance = k_distances[j-index];

                    tempVector.push_back(dm);
                }
            }
        }

        /* decide whether temporary vector should be saved */
        if((tempVector.size() == 0 && !compactResult) || tempVector.size()>0)
            matches.push_back(tempVector);

        /* increment pointer */
        index += trainDescriptors.rows;

    }

    /* delete data */
    delete mh;
    delete results;
    delete numres;
}

/* for every input desciptor, find all the ones falling in a
   certaing atching radius (from one image to a set) */
void BinaryDescriptorMatcher::radiusMatch( const Mat& queryDescriptors,
                                           std::vector<std::vector<DMatch> >& matches,
                                           float maxDistance,
                                           const std::vector<Mat>& masks,
                                           bool compactResult )
{

    /* check data validity */
    if(masks.size() !=0 && (int)masks.size() != numImages)
    {
        std::cout << "Error: the number of images in dataset is " <<
                     numImages << " but radiusMatch function received " <<
                     masks.size() << " masks. Program will be terminated"
                     << std::endl;

        return;
    }

    /* populate dataset */
    train();

    /* set K */
    dataset->setK(descrInDS);

    /* prepare structures for query */
    UINT32 *results = new UINT32[descrInDS*queryDescriptors.rows];
    UINT32 * numres = new UINT32[(256+1)*(queryDescriptors.rows)];

    /* execute query */
    dataset->batchquery(results, numres, queryDescriptors,
                        queryDescriptors.rows, queryDescriptors.cols);

    /* compose matches */
    int index = 0;
    for(int counter = 0; counter<queryDescriptors.rows; counter++)
    {
        std::vector<DMatch> tempVector;
        for(int j = index; j<index+descrInDS; j++)
        {
            std::vector<int> k_distances;
            checkKDistances(numres, descrInDS, k_distances, counter, 256);

            if(k_distances[j-index] <= maxDistance)
            {
                int currentIndex = results[j] - 1;
                std::map<int, int>::iterator itup;
                itup = indexesMap.upper_bound(currentIndex);
                itup--;

                /* data validity check */
                if(!masks.empty() && (masks[itup->second].rows != queryDescriptors.rows
                                      || masks[itup->second].cols !=1))
                {
                    std::cout << "Error: mask " << itup->second << " in radiusMatch function "
                              << "should have " << queryDescriptors.rows << " and "
                              << "1 column. Program will be terminated"
                              << std::endl;

                    return;
                }

                /* add match if necessary */
                else if(masks.empty() || masks[itup->second].at<uchar>(counter) !=0)
                {


                    DMatch dm;
                    dm.queryIdx = counter;
                    dm.trainIdx = results[j] - 1;
                    dm.imgIdx = itup->second;
                    dm.distance = k_distances[j-index];

                    tempVector.push_back(dm);
                }
            }
        }

        /* decide whether temporary vector should be saved */
        if((tempVector.size() == 0 && !compactResult) || tempVector.size()>0)
            matches.push_back(tempVector);

        /* increment pointer */
        index += descrInDS;
    }

    /* delete data */
    delete results;
    delete numres;

}
