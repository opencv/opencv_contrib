// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/video/tracking.hpp>
#include <opencv2/stereo/quasi_dense_stereo.hpp>
#include <queue>


namespace cv {
namespace stereo {

#define NO_MATCH cv::Point(0,0)

typedef std::priority_queue<MatchQuasiDense, std::vector<MatchQuasiDense>, std::less<MatchQuasiDense> > t_matchPriorityQueue;


class QuasiDenseStereoImpl : public QuasiDenseStereo
{
public:
    QuasiDenseStereoImpl(cv::Size monoImgSize, cv::String paramFilepath)
    {
        loadParameters(paramFilepath);
        width = monoImgSize.width;
        height = monoImgSize.height;
        refMap = cv::Mat_<cv::Point2i>(monoImgSize);
        mtcMap = cv::Mat_<cv::Point2i>(monoImgSize);

        cv::Size integralSize = cv::Size(monoImgSize.width+1, monoImgSize.height+1);
        sum0 = cv::Mat_<int32_t>(integralSize);
        sum1 = cv::Mat_<int32_t>(integralSize);
        ssum0 = cv::Mat_<double>(integralSize);
        ssum1 = cv::Mat_<double>(integralSize);
        // the disparity image.
        disparity = cv::Mat_<float>(monoImgSize);
        // texture images.
        textureDescLeft = cv::Mat_<int> (monoImgSize);
        textureDescRight = cv::Mat_<int> (monoImgSize);
    }

    ~QuasiDenseStereoImpl()
    {

        rightFeatures.clear();
        leftFeatures.clear();

        refMap.release();
        mtcMap.release();

        sum0.release();
        sum1.release();
        ssum0.release();
        ssum1.release();
        // the disparity image.
        disparity.release();
        // texture images.
        textureDescLeft.release();
        textureDescRight.release();
    }


    /**
     * @brief Computes sparse stereo. The output is stores in refMap and mthMap.
     *
     * This method used the "goodFeaturesToTrack" function of OpenCV to extracts salient points
     * in the left image. Feature locations are used as inputs in the "calcOpticalFlowPyrLK"
     * function of OpenCV along with the left and right images. The optical flow algorithm estimates
     * tracks the locations of the features in the right image. The two set of locations constitute
     * the sparse set of matches. These are then used as seeds in the intensification stage of the
     * algorithm.
     * @param[in] imgLeft The left Channel of a stereo image.
     * @param[in] imgRight The right Channel of a stereo image.
     * @param[out] featuresLeft (vector of points) The location of the features in the left image.
     * @param[out] featuresRight (vector of points) The location of the features in the right image.
     * @note featuresLeft and featuresRight must have the same length and corresponding features
     * must be indexed the same way in both vectors.
     */
    void sparseMatching(const cv::Mat &imgLeft ,const cv::Mat &imgRight,
                        std::vector< cv::Point2f > &featuresLeft,
                        std::vector< cv::Point2f > &featuresRight)
    {
        std::vector< uchar > featureStatus;
        std::vector< float > error;
        featuresLeft.clear();
        featuresRight.clear();

        cv::goodFeaturesToTrack(imgLeft, featuresLeft, Param.gftMaxNumFeatures,
        Param.gftQualityThres, Param.gftMinSeperationDist);

        cv::Size templateSize(Param.lkTemplateSize,Param.lkTemplateSize);
        cv::TermCriteria termination(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                     Param.lkTermParam1, Param.lkTermParam2);
        cv::calcOpticalFlowPyrLK(imgLeft, imgRight, featuresLeft, featuresRight,
        featureStatus, error,
        templateSize, Param.lkPyrLvl, termination);
        //discard bad features.
        for(size_t i=0; i<featuresLeft.size();)
        {
            if( featureStatus[i]==0 )
            {
                std::swap(featuresLeft[i], featuresLeft.back());
                featuresLeft.pop_back();
                std::swap(featureStatus[i], featureStatus.back());
                featureStatus.pop_back();
                std::swap(featuresRight[i], featuresRight.back());
                featuresRight.pop_back();
            }
            else
                ++i;
        }
    }


    /**
     * @brief Based on the seeds computed in sparse stereo, this method calculates the semi dense
     * set of correspondences.
     *
     * The method initially discards low quality matches based on their zero-normalized cross
     * correlation (zncc) value. This is done by calling the "extractSparseSeeds" method. Remaining
     * high quality Matches stored in a t_matchPriorityQueue sorted according to their zncc value.
     * The priority queue allows for new matches to be added while keeping track of the best Match.
     * The algorithm then process the queue iteratively. In every iteration a Match is popped from
     * the queue. The algorithm then tries to find candidate matches by matching every point in a
     * small patch around the left Match feature, with a point within a same sized patch around the
     * corresponding right feature. For each candidate point match, the zncc is computed and if it
     * surpasses a threshold, the candidate pair is stored in a temporary priority queue. After this
     * process completed the candidate matches are popped from the Local priority queue and if a
     * match is not registered in refMap, it means that is the best match for this point. The
     * algorithm registers this point in refMap and also push it to the Seed queue. If a candidate
     * match is already registered, it means that is not the best and the algorithm discards it.
     *
     * @note This method does not have input arguments, but uses the "leftFeatures" and
     * "rightFeatures" vectors.
     * Also there is no output since the method used refMap and mtcMap to store the results.
     * @param[in] featuresLeft The location of the features in the left image.
     * @param[in] featuresRight The location of the features in the right image.
     */
    void quasiDenseMatching(const std::vector< cv::Point2f > &featuresLeft,
                            const std::vector< cv::Point2f > &featuresRight)
    {
        dMatchesLen = 0;
        refMap = cv::Mat_<cv::Point2i>(cv::Size(width, height), cv::Point2i(0, 0));
        mtcMap = cv::Point2i(0, 0);

        // build texture homogeneity reference maps.
        buildTextureDescriptor(grayLeft, textureDescLeft);
        buildTextureDescriptor(grayRight, textureDescRight);

        // generate the intergal images for fast variable window correlation calculations
        cv::integral(grayLeft, sum0, ssum0);
        cv::integral(grayRight, sum1, ssum1);

        // Seed priority queue. The algorithm wants to pop the best seed available in order to densify
        //the sparse set.
        t_matchPriorityQueue seeds = extractSparseSeeds(featuresLeft, featuresRight,
        refMap, mtcMap);


        // Do the propagation part
        while(!seeds.empty())
        {
            t_matchPriorityQueue Local;

            // Get the best seed at the moment
            MatchQuasiDense m = seeds.top();
            seeds.pop();

            // Ignore the border
            if(!CheckBorder(m, Param.borderX, Param.borderY, width, height))
                continue;

            // For all neighbours of the seed in image 1
            //the neighborghoud is defined with Param.N*2 dimentrion
            for(int y=-Param.neighborhoodSize;y<=Param.neighborhoodSize;y++)
            {
                for(int x=-Param.neighborhoodSize;x<=Param.neighborhoodSize;x++)
                {
                    cv::Point2i p0 = cv::Point2i(m.p0.x+x,m.p0.y+y);

                    // Check if its unique in ref
                    if(refMap.at<cv::Point2i>(p0.y,p0.x) != NO_MATCH)
                        continue;

                    // Check the texture descriptor for a boundary
                    if(textureDescLeft.at<int>(p0.y, p0.x) > Param.textrureThreshold)
                        continue;

                    // For all candidate matches.
                    for(int wy=-Param.disparityGradient; wy<=Param.disparityGradient; wy++)
                    {
                        for(int wx=-Param.disparityGradient; wx<=Param.disparityGradient; wx++)
                        {
                            cv::Point p1 = cv::Point(m.p1.x+x+wx,m.p1.y+y+wy);

                            // Check if its unique in ref
                            if(mtcMap.at<cv::Point2i>(p1.y, p1.x) != NO_MATCH)
                                continue;

                            // Check the texture descriptor for a boundary
                            if(textureDescRight.at<int>(p1.y, p1.x) > Param.textrureThreshold)
                                continue;

                            // Calculate ZNCC and store local match.
                            float corr = iZNCC_c1(p0,p1,Param.corrWinSizeX,Param.corrWinSizeY);

                            // push back if this is valid match
                            if( corr > Param.correlationThreshold )
                            {
                                MatchQuasiDense nm;
                                nm.p0 = p0;
                                nm.p1 = p1;
                                nm.corr = corr;
                                Local.push(nm);
                            }
                        }
                    }
                }
            }

            // Get seeds from the local
            while( !Local.empty() )
            {
                MatchQuasiDense lm = Local.top();
                Local.pop();
                // Check if its unique in both ref and dst.
                if(refMap.at<cv::Point2i>(lm.p0.y, lm.p0.x) != NO_MATCH)
                    continue;
                if(mtcMap.at<cv::Point2i>(lm.p1.y, lm.p1.x) != NO_MATCH)
                    continue;


                // Unique match
                refMap.at<cv::Point2i>(lm.p0.y, lm.p0.x) = lm.p1;
                mtcMap.at<cv::Point2i>(lm.p1.y, lm.p1.x) = lm.p0;
                dMatchesLen++;
                // Add to the seed list
                seeds.push(lm);
            }
        }
    }


    /**
     * @brief Compute the disparity map based on the Euclidean distance of corresponding points.
     * @param[in] matchMap A matrix of points, the same size as the left channel. Each cell of this
     * matrix stores the location of the corresponding point in the right image.
     * @param[out] dispMat The disparity map.
     * @sa getDisparity
     */
    void computeDisparity(const cv::Mat_<cv::Point2i> &matchMap,
                            cv::Mat_<float> &dispMat)
    {
        for(int row=0; row< height; row++)
        {
            for(int col=0; col<width; col++)
            {
                cv::Point2d tmpPoint(col, row);

                if (matchMap.at<cv::Point2i>(tmpPoint) == NO_MATCH)
                {
                    dispMat.at<float>(tmpPoint) = NAN;
                    continue;
                }
                //if a match is found, compute the difference in location of the match and current
                //pixel.
                int dx = col-matchMap.at<cv::Point2i>(tmpPoint).x;
                int dy = row-matchMap.at<cv::Point2i>(tmpPoint).y;
                //calculate disparity of current pixel.
                dispMat.at<float>(tmpPoint) = sqrt(float(dx*dx+dy*dy));
            }
        }
    }


    /**
     * @brief Compute the Zero-mean Normalized Cross-correlation.
     *
     * Compare a patch in the left image, centered in point p0 with a patch in the right image,
     * centered in point p1. Patches are defined by wy, wx and the patch size is (2*wx+1) by
     * (2*wy+1).
     * @param [in] p0 The central point of the patch in the left image.
     * @param [in] p1 The central point of the patch in the right image.
     * @param [in] wx The distance from the center of the patch to the border in the x direction.
     * @param [in] wy The distance from the center of the patch to the border in the y direction.
     * @return The value of the the zero-mean normalized cross correlation.
     * @note Default value for wx, wy is 1. in this case the patch is 3x3.
     */
    float iZNCC_c1(const cv::Point2i p0, const cv::Point2i p1, const int wx=1, const int wy=1)
    {
        float m0=0.0 ,m1=0.0 ,s0=0.0 ,s1=0.0;
        float wa = (float)(2*wy+1)*(2*wx+1);
        float zncc=0.0;


        patchSumSum2(p0, sum0, ssum0, m0, s0, wx, wy);
        patchSumSum2(p1, sum1, ssum1, m1, s1, wx, wy);

        m0 /= wa;
        m1 /= wa;

        // standard deviations
        s0 = sqrt(s0-wa*m0*m0);
        s1 = sqrt(s1-wa*m1*m1);


        for (int col=-wy; col<=wy; col++)
        {
            for (int row=-wx; row<=wx; row++)
            {
                zncc += (float)grayLeft.at<uchar>(p0.y+row, p0.x+col) *
                (float)grayRight.at<uchar>(p1.y+row, p1.x+col);
            }
        }
        zncc = (zncc-wa*m0*m1)/(s0*s1);
        return zncc;
    }


    /**
     * @brief Compute the sum of values and the sum of squared values of a patch with dimensions
     * 2*xWindow+1 by 2*yWindow+1 and centered in point p, using the integral image and integral
     * image of squared pixel values.
     * @param[in] p The center of the patch we want to calculate the sum and sum of squared values.
     * @param[in] s The integral image
     * @param[in] ss The integral image of squared values.
     * @param[out] sum The sum of pixels inside the patch.
     * @param[out] ssum The sum of squared values inside the patch.
     * @param [in] xWindow The distance from the central pixel of the patch to the border in x
     * direction.
     * @param [in] yWindow The distance from the central pixel of the patch to the border in y
     * direction.
     * @note Default value for xWindow, yWindow is 1. in this case the patch is 3x3.
     * @note integral images are very useful to sum values of patches in constant time independent
     * of their size. For more information refer to the cv::Integral function OpenCV page.
     */
    void patchSumSum2(const cv::Point2i p, const cv::Mat &sum, const cv::Mat &ssum,
                        float &s, float &ss, const int xWindow=1, const int yWindow=1)
    {
      cv::Point2i otl(p.x-xWindow, p.y-yWindow);
      //outer top right
      cv::Point2i otr(p.x+xWindow+1, p.y-yWindow);
      //outer bottom left
      cv::Point2i obl(p.x-xWindow, p.y+yWindow+1);
      //outer bottom right
      cv::Point2i obr(p.x+xWindow+1, p.y+yWindow+1);

      // sum and squared sum for right window
      s = (float)(sum.at<int>(otl) - sum.at<int>(otr)
          - sum.at<int>(obl) + sum.at<int>(obr));

      ss = (float)(ssum.at<double>(otl) - ssum.at<double>(otr)
           - ssum.at<double>(obl) + ssum.at<double>(obr));
    }


    /**
     * @brief Create a priority queue containing sparse Matches
     *
     * This method computes the zncc for each Match extracted in "sparseMatching". If the zncc is
     * over the correlation threshold then the Match is inserted in the output priority queue.
     * @param[in] featuresLeft The feature locations in the left image.
     * @param[in] featuresRight The features locations in the right image.
     * @param[out] leftMap A matrix of points, of the same size as the left image. Each cell of this
     * matrix stores the location of the corresponding point in the right image.
     * @param[out] rightMap A matrix of points, the same size as the right image. Each cell of this
     * matrix stores the location of the corresponding point in the left image.
     * @return Priority queue containing sparse matches.
     */
    t_matchPriorityQueue extractSparseSeeds(const std::vector< cv::Point2f > &featuresLeft,
                                            const std::vector< cv::Point2f >  &featuresRight,
                                            cv::Mat_<cv::Point2i> &leftMap,
                                            cv::Mat_<cv::Point2i> &rightMap)
    {
        t_matchPriorityQueue seeds;
        for(uint i=0; i < featuresLeft.size(); i++)
        {
            // Calculate correlation and store match in Seeds.
            MatchQuasiDense m;
            m.p0 = cv::Point2i(featuresLeft[i]);
            m.p1 = cv::Point2i(featuresRight[i]);
            m.corr = 0;

            // Check if too close to boundary.
            if(!CheckBorder(m,Param.borderX,Param.borderY, width, height))
            continue;

            m.corr = iZNCC_c1(m.p0, m.p1, Param.corrWinSizeX, Param.corrWinSizeY);
            // Can we add it to the list
            if( m.corr > Param.correlationThreshold )
            {
                seeds.push(m);
                leftMap.at<cv::Point2i>(m.p0.y, m.p0.x) = m.p1;
                rightMap.at<cv::Point2i>(m.p1.y, m.p1.x) = m.p0;
            }
        }
        return seeds;
    }


    /**
     * @brief Check if a match is close to the boarder of an image.
     * @param[in] m The match containing points in both image.
     * @param[in] bx The offset of the image edge that defines the border in x direction.
     * @param[in] by The offset of the image edge that defines the border in y direction.
     * @param[in] w The width of the image.
     * @param[in] h The height of the image.
     * @retval true If the feature is in the border of the image.
     * @retval false If the feature is not in the border of image.
     */
    bool CheckBorder(MatchQuasiDense m, int bx, int by, int w, int h)
    {
        if(m.p0.x<bx || m.p0.x>w-bx || m.p0.y<by || m.p0.y>h-by ||
        m.p1.x<bx || m.p1.x>w-bx || m.p1.y<by || m.p1.y>h-by)
        {
            return false;
        }

        return true;
    }


    /**
     * @brief Build a texture descriptor
     * @param[in] img The image we need to compute the descriptor for.
     * @param[out] descriptor The texture descriptor of the image.
     */
    void buildTextureDescriptor(cv::Mat &img,cv::Mat &descriptor)
    {
        float a, b, c, d;

        uint8_t center, top, bottom, right, left;
        //reset descriptors

        // traverse every pixel.
        for(int row=1; row<height-1; row++)
        {
            for(int col=1; col<width-1; col++)
            {
                // the values of the current pixel.
                center = img.at<uchar>(row,col);
                top = img.at<uchar>(row-1,col);
                bottom = img.at<uchar>(row+1,col);
                left = img.at<uchar>(row,col-1);
                right = img.at<uchar>(row,col+1);

                a = (float)abs(center - top);
                b = (float)abs(center - bottom);
                c = (float)abs(center - left);
                d = (float)abs(center - right);
                //choose the biggest of them.
                int val = (int) std::max(a, std::max(b, std::max(c, d)));
                descriptor.at<int>(row, col) = val;
            }
        }
    }

    //-------------------------------------------------------------------------


    void getSparseMatches(std::vector<stereo::MatchQuasiDense> &sMatches) override
    {
        MatchQuasiDense tmpMatch;
        sMatches.clear();
        sMatches.reserve(leftFeatures.size());
        for (uint i=0; i<leftFeatures.size(); i++)
        {
            tmpMatch.p0 = leftFeatures[i];
            tmpMatch.p1 = rightFeatures[i];
            sMatches.push_back(tmpMatch);
        }
    }

    int loadParameters(cv::String filepath) override
    {
        cv::FileStorage fs;
        //if user specified a pathfile, try to use it.
        if (!filepath.empty())
        {
            fs.open(filepath, cv::FileStorage::READ);
        }
        // If the file opened, read the parameters.
        if (fs.isOpened())
        {
            fs["borderX"] >> Param.borderX;
            fs["borderY"] >> Param.borderY;
            fs["corrWinSizeX"] >> Param.corrWinSizeX;
            fs["corrWinSizeY"] >> Param.corrWinSizeY;
            fs["correlationThreshold"] >> Param.correlationThreshold;
            fs["textrureThreshold"] >> Param.textrureThreshold;

            fs["neighborhoodSize"] >> Param.neighborhoodSize;
            fs["disparityGradient"] >> Param.disparityGradient;

            fs["lkTemplateSize"] >> Param.lkTemplateSize;
            fs["lkPyrLvl"] >> Param.lkPyrLvl;
            fs["lkTermParam1"] >> Param.lkTermParam1;
            fs["lkTermParam2"] >> Param.lkTermParam2;

            fs["gftQualityThres"] >> Param.gftQualityThres;
            fs["gftMinSeperationDist"] >> Param.gftMinSeperationDist;
            fs["gftMaxNumFeatures"] >> Param.gftMaxNumFeatures;
            fs.release();
            return 1;
        }
        // If the filepath was incorrect or non existent, load default parameters.
        Param.borderX = 15;
        Param.borderY = 15;
        // corr window size
        Param.corrWinSizeX = 5;
        Param.corrWinSizeY = 5;
        Param.correlationThreshold = (float)0.5;
        Param.textrureThreshold = 200;

        Param.neighborhoodSize = 5;
        Param.disparityGradient = 1;

        Param.lkTemplateSize = 3;
        Param.lkPyrLvl = 3;
        Param.lkTermParam1 = 3;
        Param.lkTermParam2 = (float)0.003;

        Param.gftQualityThres = (float)0.01;
        Param.gftMinSeperationDist = 10;
        Param.gftMaxNumFeatures = 500;
        // Return 0 if there was no filepath provides.
        // Return -1 if there was a problem opening the filepath provided.
        if(filepath.empty())
        {
            return 0;
        }
        return -1;
    }

    int saveParameters(cv::String filepath) override
    {
        cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
        if (fs.isOpened())
        {
            fs << "borderX" << Param.borderX;
            fs << "borderY" << Param.borderY;
            fs << "corrWinSizeX" << Param.corrWinSizeX;
            fs << "corrWinSizeY" << Param.corrWinSizeY;
            fs << "correlationThreshold" << Param.correlationThreshold;
            fs << "textrureThreshold" << Param.textrureThreshold;

            fs << "neighborhoodSize" << Param.neighborhoodSize;
            fs << "disparityGradient" << Param.disparityGradient;

            fs << "lkTemplateSize" << Param.lkTemplateSize;
            fs << "lkPyrLvl" << Param.lkPyrLvl;
            fs << "lkTermParam1" << Param.lkTermParam1;
            fs << "lkTermParam2" << Param.lkTermParam2;

            fs << "gftQualityThres" << Param.gftQualityThres;
            fs << "gftMinSeperationDist" << Param.gftMinSeperationDist;
            fs << "gftMaxNumFeatures" << Param.gftMaxNumFeatures;
            fs.release();
        }
        return -1;
    }

    void getDenseMatches(std::vector<stereo::MatchQuasiDense> &denseMatches) override
    {
        MatchQuasiDense tmpMatch;
        denseMatches.clear();
        denseMatches.reserve(dMatchesLen);
        for (int row=0; row<height; row++)
        {
            for(int col=0; col<width; col++)
            {
                tmpMatch.p0 = cv::Point(col, row);
                tmpMatch.p1 = refMap.at<Point2i>(row, col);
                if (tmpMatch.p1 == NO_MATCH)
                {
                    continue;
                }
                denseMatches.push_back(tmpMatch);
            }
        }
    }

    void process(const cv::Mat &imgLeft , const cv::Mat &imgRight) override
    {
        if (imgLeft.channels()>1)
        {
            cv::cvtColor(imgLeft, grayLeft, cv::COLOR_BGR2GRAY);
            cv::cvtColor(imgRight, grayRight, cv::COLOR_BGR2GRAY);
        }
        else
        {
            grayLeft = imgLeft.clone();
            grayRight = imgRight.clone();
        }
        sparseMatching(grayLeft, grayRight, leftFeatures, rightFeatures);
        quasiDenseMatching(leftFeatures, rightFeatures);
    }

    cv::Point2f getMatch(const int x, const int y) override
    {
        return refMap.at<cv::Point2i>(y, x);
    }

    cv::Mat getDisparity() override
    {
        computeDisparity(refMap, disparity);
        return disparity;
    }

    // Variables used at sparse feature extraction.
    // Container for left images' features, extracted with GFT algorithm.
    std::vector< cv::Point2f > leftFeatures;
    // Container for right images' features, matching is done with LK flow algorithm.
    std::vector< cv::Point2f > rightFeatures;

    // Width and height of a single image.
    int width;
    int height;
    int dMatchesLen;
    // Containers to store input images.
    cv::Mat grayLeft;
    cv::Mat grayRight;
    // Containers to store the locations of each points pair.
    cv::Mat_<cv::Point2i> refMap;
    cv::Mat_<cv::Point2i> mtcMap;
    cv::Mat_<int32_t> sum0;
    cv::Mat_<int32_t> sum1;
    cv::Mat_<double> ssum0;
    cv::Mat_<double> ssum1;
    // Container to store the disparity un-normalized
    cv::Mat_<float> disparity;
    // Containers to store textures descriptors.
    cv::Mat_<int> textureDescLeft;
    cv::Mat_<int> textureDescRight;

};

cv::Ptr<QuasiDenseStereo> QuasiDenseStereo::create(cv::Size monoImgSize, cv::String paramFilepath)
{
    return cv::makePtr<QuasiDenseStereoImpl>(monoImgSize, paramFilepath);
}

QuasiDenseStereo::~QuasiDenseStereo(){

}


}
}
