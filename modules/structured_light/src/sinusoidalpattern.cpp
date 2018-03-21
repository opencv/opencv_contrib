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
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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

namespace cv {
namespace structured_light {
class CV_EXPORTS_W SinusoidalPatternProfilometry_Impl CV_FINAL : public SinusoidalPattern
{
public:
    // Constructor
    explicit SinusoidalPatternProfilometry_Impl( const SinusoidalPattern::Params &parameters =
                                                 SinusoidalPattern::Params() );
    // Destructor
    virtual ~SinusoidalPatternProfilometry_Impl() CV_OVERRIDE {};

    // Generate sinusoidal patterns
    bool generate( OutputArrayOfArrays patternImages ) CV_OVERRIDE;

    bool decode( const std::vector< std::vector<Mat> >& patternImages, OutputArray disparityMap,
                InputArrayOfArrays blackImages = noArray(), InputArrayOfArrays whiteImages =
                noArray(), int flags = 0 ) const CV_OVERRIDE;

    // Compute a wrapped phase map from the sinusoidal patterns
    void computePhaseMap( InputArrayOfArrays patternImages, OutputArray wrappedPhaseMap,
                         OutputArray shadowMask = noArray(), InputArray fundamental = noArray()) CV_OVERRIDE;
    // Unwrap the wrapped phase map to retrieve correspondences
    void unwrapPhaseMap( InputArray wrappedPhaseMap,
                         OutputArray unwrappedPhaseMap,
                         cv::Size camSize,
                         InputArray shadowMask = noArray() ) CV_OVERRIDE;
    // Find correspondences between the devices
    void findProCamMatches( InputArray projUnwrappedPhaseMap, InputArray camUnwrappedPhaseMap,
                            OutputArrayOfArrays matches ) CV_OVERRIDE;

    void computeDataModulationTerm( InputArrayOfArrays patternImages,
                                    OutputArray dataModulationTerm,
                                    InputArray shadowMask ) CV_OVERRIDE;

private:
    // Compute The Fourier transform of a pattern. Output is complex. Taken from the DFT example in OpenCV
    void computeDft( InputArray patternImage, OutputArray FourierTransform );
    // Compute the inverse Fourier transform. Output can be complex or real
    void computeInverseDft( InputArray FourierTransform, OutputArray inverseFourierTransform,
                            bool realOutput );
    // Compute the DFT magnitude which is used to find maxima in the spectrum
    void computeDftMagnitude( InputArray FourierTransform, OutputArray FourierTransformMagnitude );
    // Compute phase map from the complex signal given by non-symmetrical filtering of DFT
    void computeFtPhaseMap( InputArray inverseFourierTransform,
                            InputArray shadowMask,
                            OutputArray wrappedPhaseMap );
    // Swap DFT quadrants. Come from opencv example
    void swapQuadrants( InputOutputArray image, int centerX, int centerY );
    // Filter (non)-symmetrically the DFT.
    void frequencyFiltering( InputOutputArray FourierTransform, int centerX1, int centerY1,
                             int halfRegionWidth, int halfRegionHeight, bool keepInsideRegion,
                             int centerX2 = -1, int centerY2 = -1 );
    // Find maxima in the spectrum so that we know how it should be filtered
    bool findMaxInHalvesTransform( InputArray FourierTransformMag, Point &maxPosition1,
                                  Point &maxPosition2 );
    // Compute phase map from the three sinusoidal patterns
    void computePsPhaseMap( InputArrayOfArrays patternImages,
                            InputArray shadowMask,
                            OutputArray wrappedPhaseMap );

    void computeFapsPhaseMap( InputArray a, InputArray b, InputArray theta1, InputArray theta2,
                              InputArray shadowMask, OutputArray wrappedPhaseMap );
    // Compute a shadow mask to discard shadow regions
    void computeShadowMask( InputArrayOfArrays patternImages, OutputArray shadowMask );
    // Data modulation term is used to isolate cross markers

    void extractMarkersLocation( InputArray dataModulationTerm,
                                 std::vector<Point> &markersLocation );

    void convertToAbsolutePhaseMap( InputArrayOfArrays camPatterns,
                                    InputArray unwrappedProjPhaseMap,
                                    InputArray unwrappedCamPhaseMap,
                                    InputArray shadowMask,
                                    InputArray fundamentalMatrix );

    Params params;
    phase_unwrapping::HistogramPhaseUnwrapping::Params unwrappingParams;
    // Class describing markers that are added to the patterns
    class Marker{
    private:
        Point center, up, right, left, down;
    public:
        Marker();
        Marker( Point c );
        void drawMarker( OutputArray pattern );
    };
};
// Default parameters value
SinusoidalPattern::Params::Params()
{
    width = 800;
    height = 600;
    nbrOfPeriods = 20;
    shiftValue = (float)(2 * CV_PI / 3);
    methodId = FAPS;
    nbrOfPixelsBetweenMarkers = 56;
    horizontal = false;
    setMarkers = false;
}
SinusoidalPatternProfilometry_Impl::Marker::Marker(){};

SinusoidalPatternProfilometry_Impl::Marker::Marker( Point c )
{
    center = c;
    up.x = c.x;
    up.y = c.y - 1;
    left.x = c.x - 1;
    left.y = c.y;

    down.x = c.x;
    down.y = c.y + 1;
    right.x = c.x + 1;
    right.y = c.y;
}
// Draw marker on a pattern
void SinusoidalPatternProfilometry_Impl::Marker::drawMarker( OutputArray pattern )
{
    Mat &pattern_ = *(Mat*) pattern.getObj();

    pattern_.at<uchar>(center.x, center.y) = 255;
    pattern_.at<uchar>(up.x, up.y) = 255;
    pattern_.at<uchar>(right.x, right.y) = 255;
    pattern_.at<uchar>(left.x, left.y) = 255;
    pattern_.at<uchar>(down.x, down.y) = 255;
}

SinusoidalPatternProfilometry_Impl::SinusoidalPatternProfilometry_Impl(
        const SinusoidalPattern::Params &parameters ) : params(parameters)
{

}
// Generate sinusoidal patterns. Markers are optional
bool SinusoidalPatternProfilometry_Impl::generate( OutputArrayOfArrays pattern )
{
    // Three patterns are used in the reference paper.
    int nbrOfPatterns = 3;
    float meanAmpl = 127.5;
    float sinAmpl = 127.5;
    // Period in number of pixels
    int period;
    float frequency;
    // m and n are parameters described in the reference paper
    int m = params.nbrOfPixelsBetweenMarkers;
    int n;
    // Offset for the first marker of the first row.
    int firstMarkerOffset = 10;
    int mnRatio;
    int nbrOfMarkersOnOneRow;
    std::vector<Mat> &pattern_ = *(std::vector<Mat>*) pattern.getObj();

    n = params.nbrOfPeriods / nbrOfPatterns;
    mnRatio = m / n;

    pattern_.resize(nbrOfPatterns);

    if( params.horizontal )
    {
        period = params.height / params.nbrOfPeriods;
        nbrOfMarkersOnOneRow = (int)floor(static_cast<float>((params.width - firstMarkerOffset) / m));
    }
    else
    {
        period = params.width / params.nbrOfPeriods;
        nbrOfMarkersOnOneRow = (int)floor(static_cast<float>((params.height - firstMarkerOffset) / m));
    }
    frequency = (float) 1 / period;

    for( int i = 0; i < nbrOfPatterns; ++i )
    {
        pattern_[i] = Mat(params.height, params.width, CV_8UC1);

        if( params.horizontal )
        pattern_[i] = pattern_[i].t();
    }
    // Patterns vary along one direction only so, a row Mat can be created and copied to the pattern's rows
    for( int i = 0; i < nbrOfPatterns; ++i )
    {
        Mat rowValues(1, pattern_[i].cols, CV_8UC1);

        for( int j = 0; j < pattern_[i].cols; ++j )
        {
            rowValues.at<uchar>(0, j) = saturate_cast<uchar>(
                    meanAmpl + sinAmpl * sin(2 * CV_PI * frequency * j + i * params.shiftValue));
        }

        for( int j = 0; j < pattern_[i].rows; ++j )
        {
            rowValues.row(0).copyTo(pattern_[i].row(j));
        }
    }
    // Add cross markers to the patterns.
    if( params.setMarkers )
    {
        for( int i = 0; i < nbrOfPatterns; ++i )
        {
            for( int j = 0; j < n; ++j )
            {
                for( int k = 0; k < nbrOfMarkersOnOneRow; ++k )
                {
                    Marker mark(Point(firstMarkerOffset + k * m + j * mnRatio,
                            3 * period / 4 + j * period + i * period * n  - i * period / 3));
                    mark.drawMarker(pattern_[i]);
                    params.markersLocation.push_back(Point2f((float)(firstMarkerOffset + k * m + j * mnRatio),
                            (float) (3 * period / 4 + j * period + i * period * n  - i * period / 3)));
                }
            }
        }
    }
    if( params.horizontal )
        for( int i = 0; i < nbrOfPatterns; ++i )
        {
            pattern_[i] = pattern_[i].t();
        }
    return true;
}

bool SinusoidalPatternProfilometry_Impl::decode(const std::vector< std::vector<Mat> >& patternImages,
                                                OutputArray disparityMap,
                                                InputArrayOfArrays blackImages,
                                                InputArrayOfArrays whiteImages, int flags ) const
{
    (void) patternImages;
    (void) disparityMap;
    (void) blackImages;
    (void) whiteImages;
    (void) flags;
    return true;
}
// Most of the steps described in the paper to get the wrapped phase map take place here
void SinusoidalPatternProfilometry_Impl::computePhaseMap( InputArrayOfArrays patternImages,
                                                          OutputArray wrappedPhaseMap,
                                                          OutputArray shadowMask,
                                                          InputArray fundamental  )
{
    std::vector<Mat> &pattern_ = *(std::vector<Mat>*) patternImages.getObj();
    Mat &wrappedPhaseMap_ = *(Mat*) wrappedPhaseMap.getObj();
    int rows = pattern_[0].rows;
    int cols = pattern_[0].cols;
    int dcWidth = 5;
    int dcHeight = 5;
    int bpWidth = 21;
    int bpHeight = 21;
    // Compute wrapped phase map for FTP
    if( params.methodId == FTP )
    {
        Mat &shadowMask_ = *(Mat*) shadowMask.getObj();
        Mat dftImage, complexInverseDft;
        Mat dftMag;
        int halfWidth = cols/2;
        int halfHeight = rows/2;
        Point m1, m2;
        computeShadowMask(pattern_, shadowMask_);

        computeDft(pattern_[0], dftImage); //compute the complex pattern DFT
        swapQuadrants(dftImage, halfWidth, halfHeight); //swap quadrants to get 0 frequency in (halfWidth, halfHeight)
        frequencyFiltering(dftImage, halfHeight, halfWidth, dcHeight, dcWidth, false); //get rid of 0 frequency
        computeDftMagnitude(dftImage, dftMag); //compute magnitude to find maxima
        findMaxInHalvesTransform(dftMag, m1, m2); //look for maxima in the magnitude. Useful information is located around maxima
        frequencyFiltering(dftImage, m2.y, m2.x, bpHeight, bpWidth, true); //keep useful information only
        swapQuadrants(dftImage,halfWidth, halfHeight); //swap quadrants again to compute inverse dft
        computeInverseDft(dftImage, complexInverseDft, false); //compute inverse dft. Result is complex since we only keep half of the spectrum
        computeFtPhaseMap(complexInverseDft, shadowMask_, wrappedPhaseMap_); //compute phaseMap from the complex image.
    }
    // Compute wrapped pahse map for PSP
    else if( params.methodId == PSP )
    {
        Mat &shadowMask_ = *(Mat*) shadowMask.getObj();
        //Mat &fundamental_ = *(Mat*) fundamental.getObj();
        (void) fundamental;
        Mat dmt;
        int nbrOfPatterns = static_cast<int>(pattern_.size());
        std::vector<Mat> filteredPatterns(nbrOfPatterns);
        std::vector<Mat> dftImages(nbrOfPatterns);
        std::vector<Mat> dftMags(nbrOfPatterns);
        int halfWidth = cols/2;
        int halfHeight = rows/2;
        Point m1, m2;

        computeShadowMask(pattern_, shadowMask_);

        //this loop symmetrically filters pattern to remove cross markers.
        for( int i = 0; i < nbrOfPatterns; ++i )
        {
            computeDft(pattern_[i], dftImages[i]);
            swapQuadrants(dftImages[i], halfWidth, halfHeight);
            frequencyFiltering(dftImages[i], halfHeight, halfWidth, dcHeight, dcWidth, false);
            computeDftMagnitude(dftImages[i], dftMags[i]);
            findMaxInHalvesTransform(dftMags[i], m1, m2);
            frequencyFiltering(dftImages[i], m1.y, m1.x, bpHeight, bpWidth, true, m2.y, m2.x);//symmetrical filtering
            swapQuadrants(dftImages[i], halfWidth, halfHeight);
            computeInverseDft(dftImages[i], filteredPatterns[i], true);

        }
        computePsPhaseMap(filteredPatterns, shadowMask_, wrappedPhaseMap_);
    }
    else if( params.methodId == FAPS )
    {
        Mat &shadowMask_ = *(Mat*) shadowMask.getObj();
        int nbrOfPatterns = static_cast<int>(pattern_.size());
        std::vector<Mat> unwrappedFTPhaseMaps;
        std::vector<Mat> filteredPatterns(nbrOfPatterns);
        Mat dmt;
        Mat theta1, theta2, a, b;
        std::vector<Point> markersLoc;
        cv::Size camSize;
        camSize.height = pattern_[0].rows;
        camSize.width = pattern_[0].cols;
        computeShadowMask(pattern_, shadowMask_);

        for( int i = 0; i < nbrOfPatterns; ++i )
        {
            Mat dftImage, complexInverseDft;
            Mat dftMag;
            Mat tempWrappedPhaseMap;
            Mat tempUnwrappedPhaseMap;
            int halfWidth = cols/2;
            int halfHeight = rows/2;
            Point m1, m2;

            computeDft(pattern_[i], dftImage); //compute the complex pattern DFT
            swapQuadrants(dftImage, halfWidth, halfHeight); //swap quadrants to get 0 frequency in (halfWidth, halfHeight)
            frequencyFiltering(dftImage, halfHeight, halfWidth, dcHeight, dcWidth, false); //get rid of 0 frequency
            computeDftMagnitude(dftImage, dftMag); //compute magnitude to find maxima
            findMaxInHalvesTransform(dftMag, m1, m2); //look for maxima in the magnitude. Useful information is located around maxima
            frequencyFiltering(dftImage, m2.y, m2.x, bpHeight, bpWidth, true); //keep useful information only
            swapQuadrants(dftImage,halfWidth, halfHeight); //swap quadrants again to compute inverse dft
            computeInverseDft(dftImage, complexInverseDft, false); //compute inverse dft. Result is complex since we only keep half of the spectrum
            computeFtPhaseMap(complexInverseDft, shadowMask_, tempWrappedPhaseMap); //compute phaseMap from the complex image.
            unwrapPhaseMap(tempWrappedPhaseMap, tempUnwrappedPhaseMap, camSize, shadowMask);
            unwrappedFTPhaseMaps.push_back(tempUnwrappedPhaseMap);
            computeInverseDft(dftImage, filteredPatterns[i], true);
        }

        theta1.create(camSize.height, camSize.width, unwrappedFTPhaseMaps[0].type());
        theta2.create(camSize.height, camSize.width, unwrappedFTPhaseMaps[0].type());
        a.create(camSize.height, camSize.width, CV_32FC1);
        b.create(camSize.height, camSize.width, CV_32FC1);

        a = filteredPatterns[0] - filteredPatterns[1];
        b = filteredPatterns[1] - filteredPatterns[2];

        theta1 = unwrappedFTPhaseMaps[1] - unwrappedFTPhaseMaps[0];
        theta2 = unwrappedFTPhaseMaps[2] - unwrappedFTPhaseMaps[1];

        computeFapsPhaseMap(a, b, theta1, theta2, shadowMask_, wrappedPhaseMap_);
    }
}

void SinusoidalPatternProfilometry_Impl::unwrapPhaseMap( InputArray wrappedPhaseMap,
                                                         OutputArray unwrappedPhaseMap,
                                                         cv::Size camSize,
                                                         InputArray shadowMask )
{
    int rows = params.height;
    int cols = params.width;
    unwrappingParams.width = camSize.width;
    unwrappingParams.height = camSize.height;

    Mat &wPhaseMap = *(Mat*) wrappedPhaseMap.getObj();
    Mat &uPhaseMap = *(Mat*) unwrappedPhaseMap.getObj();
    Mat mask;

    if( shadowMask.empty() )
    {
        mask.create(rows, cols, CV_8UC1);
        mask = Scalar::all(255);
    }
    else
    {
        Mat &temp = *(Mat*) shadowMask.getObj();
        temp.copyTo(mask);
    }

    Ptr<phase_unwrapping::HistogramPhaseUnwrapping> phaseUnwrapping =
            phase_unwrapping::HistogramPhaseUnwrapping::create(unwrappingParams);

    phaseUnwrapping->unwrapPhaseMap(wPhaseMap, uPhaseMap, mask);
}

void SinusoidalPatternProfilometry_Impl::findProCamMatches( InputArray projUnwrappedPhaseMap,
                                                            InputArray camUnwrappedPhaseMap,
                                                            OutputArrayOfArrays matches )
{
    (void) projUnwrappedPhaseMap;
    (void) camUnwrappedPhaseMap;
    (void) matches;
}

void SinusoidalPatternProfilometry_Impl::computeDft( InputArray patternImage,
                                                     OutputArray FourierTransform )
{
    Mat &pattern_ = *(Mat*) patternImage.getObj();
    Mat &FourierTransform_ = *(Mat*) FourierTransform.getObj();
    Mat padded;
    int m = getOptimalDFTSize(pattern_.rows);
    int n = getOptimalDFTSize(pattern_.cols);
    copyMakeBorder(pattern_, padded, 0, m - pattern_.rows, 0, n - pattern_.cols, BORDER_CONSTANT,
                   Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planes, 2, FourierTransform_);
    dft(FourierTransform_, FourierTransform_);
}

void SinusoidalPatternProfilometry_Impl::computeInverseDft( InputArray FourierTransform,
                                                           OutputArray inverseFourierTransform,
                                                           bool realOutput )
{
    Mat &FourierTransform_ = *(Mat*) FourierTransform.getObj();
    Mat &inverseFourierTransform_ = *(Mat*) inverseFourierTransform.getObj();
    if( realOutput )
        idft(FourierTransform_, inverseFourierTransform_, DFT_SCALE | DFT_REAL_OUTPUT);
    else
        idft(FourierTransform_, inverseFourierTransform_, DFT_SCALE);
}

void SinusoidalPatternProfilometry_Impl::computeDftMagnitude( InputArray FourierTransform,
                                                              OutputArray FourierTransformMagnitude )
{
    Mat &FourierTransform_ = *(Mat*) FourierTransform.getObj();
    Mat &FourierTransformMagnitude_ = *(Mat*) FourierTransformMagnitude.getObj();
    Mat planes[2];
    split(FourierTransform_, planes);
    magnitude(planes[0], planes[1], planes[0]);
    FourierTransformMagnitude_ = planes[0];
    FourierTransformMagnitude_ += Scalar::all(1);
    log(FourierTransformMagnitude_, FourierTransformMagnitude_);
    FourierTransformMagnitude_ = FourierTransformMagnitude_(
            Rect(0, 0, FourierTransformMagnitude_.cols & -2, FourierTransformMagnitude_.rows & - 2));
    normalize(FourierTransformMagnitude_, FourierTransformMagnitude_, 0, 1, NORM_MINMAX);
}

void SinusoidalPatternProfilometry_Impl::computeFtPhaseMap( InputArray inverseFourierTransform,
                                                            InputArray shadowMask,
                                                            OutputArray wrappedPhaseMap )
{

    Mat &inverseFourierTransform_ = *(Mat*) inverseFourierTransform.getObj();
    Mat &wrappedPhaseMap_ = *(Mat*) wrappedPhaseMap.getObj();
    Mat &shadowMask_ = *(Mat*) shadowMask.getObj();
    Mat planes[2];

    int rows = inverseFourierTransform_.rows;
    int cols = inverseFourierTransform_.cols;

    if( wrappedPhaseMap_.empty () )
        wrappedPhaseMap_.create(rows, cols, CV_32FC1);

    split(inverseFourierTransform_, planes);

    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            if( shadowMask_.at<uchar>(i, j) != 0 )
            {
                float im = planes[1].at<float>(i, j);
                float re = planes[0].at<float>(i, j);
                wrappedPhaseMap_.at<float>(i, j) = atan2(re, im);
            }
            else
            {
                wrappedPhaseMap_.at<float>(i, j) = 0;
            }
        }
    }
}
void SinusoidalPatternProfilometry_Impl::swapQuadrants( InputOutputArray image,
                                                       int centerX, int centerY )
{
    Mat &image_ = *(Mat*) image.getObj();
    Mat q0(image_, Rect(0, 0, centerX, centerY));
    Mat q1(image_, Rect(centerX, 0, centerX, centerY));
    Mat q2(image_, Rect(0, centerY, centerX, centerY));
    Mat q3(image_, Rect(centerX, centerY, centerX, centerY));
    Mat tmp;

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void SinusoidalPatternProfilometry_Impl::frequencyFiltering( InputOutputArray FourierTransform,
                                                             int centerX1, int centerY1,
                                                             int halfRegionWidth, int halfRegionHeight,
                                                             bool keepInsideRegion, int centerX2,
                                                             int centerY2 )
{
    Mat &FourierTransform_ = *(Mat*) FourierTransform.getObj();
    int rows = FourierTransform_.rows;
    int cols = FourierTransform_.cols;
    int type = FourierTransform_.type();
    if( keepInsideRegion )
    {
        Mat maskedTransform(rows, cols, type);
        maskedTransform = Scalar::all(0);
        Mat roi1 = FourierTransform_(
                Rect(centerY1 - halfRegionHeight, centerX1 - halfRegionWidth,
                    2 * halfRegionHeight, 2 * halfRegionWidth));
        Mat dstRoi1 = maskedTransform(
                Rect(centerY1 - halfRegionHeight, centerX1 - halfRegionWidth,
                    2 * halfRegionHeight, 2 * halfRegionWidth));
        roi1.copyTo(dstRoi1);

        if( centerY2 != -1 || centerX2 != -1 )
        {
            Mat roi2 = FourierTransform_(
                    Rect(centerY2 - halfRegionHeight, centerX2 - halfRegionWidth,
                        2 * halfRegionHeight, 2 * halfRegionWidth));
            Mat dstRoi2 = maskedTransform(
                    Rect(centerY2 - halfRegionHeight, centerX2 - halfRegionWidth,
                        2 * halfRegionHeight, 2 * halfRegionWidth));
            roi2.copyTo(dstRoi2);
        }
        FourierTransform_ = maskedTransform;
    }
    else
    {
        Mat roi(2 * halfRegionHeight, 2 * halfRegionWidth, type);
        roi = Scalar::all(0);

        Mat dstRoi1 = FourierTransform_(
                Rect(centerY1 - halfRegionHeight, centerX1 - halfRegionWidth,
                    2 * halfRegionHeight, 2 * halfRegionWidth));
        roi.copyTo(dstRoi1);

        if( centerY2 != -1 || centerX2 != -1 )
        {
            Mat dstRoi2 = FourierTransform_(
                    Rect(centerY2 - halfRegionHeight, centerX2 - halfRegionWidth,
                        2 * halfRegionHeight, 2 * halfRegionWidth));
            roi.copyTo(dstRoi2);
        }
    }
}
bool SinusoidalPatternProfilometry_Impl::findMaxInHalvesTransform( InputArray FourierTransformMag,
                                                                   Point &maxPosition1,
                                                                   Point &maxPosition2 )
{
    Mat &FourierTransformMag_ = *(Mat*) FourierTransformMag.getObj();

    int centerX = FourierTransformMag_.cols / 2;
    int centerY = FourierTransformMag_.rows / 2;
    Mat h0, h1;
    double maxV1 = -1;
    double maxV2 = -1;
    int margin = 5;

    if( params.horizontal )
    {
        h0 = FourierTransformMag_(Rect(0, 0, FourierTransformMag_.cols, centerY - margin));
        h1 = FourierTransformMag_(
                Rect(0, centerY + margin, FourierTransformMag_.cols, centerY - margin));
    }
    else
    {
        h0 = FourierTransformMag_(Rect(0, 0, centerX - margin, FourierTransformMag_.rows));
        h1 = FourierTransformMag_(
                Rect(centerX + margin, 0, centerX - margin, FourierTransformMag_.rows));
    }

    minMaxLoc(h0, NULL, &maxV1, NULL, &maxPosition1);
    minMaxLoc(h1, NULL, &maxV2, NULL, &maxPosition2);

    if( params.horizontal )
    {
        maxPosition2.y = maxPosition2.y + centerY + margin;
    }
    else
    {
        maxPosition2.x = maxPosition2.x + centerX + margin;
    }

    if( maxV1 == -1 || maxV2 == -1 )
    {
        return false;
    }

    return true;
}

void SinusoidalPatternProfilometry_Impl::computePsPhaseMap( InputArrayOfArrays patternImages,
                                                            InputArray shadowMask,
                                                            OutputArray wrappedPhaseMap )
{
    std::vector<Mat> &pattern_ = *(std::vector<Mat>*) patternImages.getObj();
    Mat &wrappedPhaseMap_ = *(Mat*) wrappedPhaseMap.getObj();
    Mat &shadowMask_ = *(Mat*) shadowMask.getObj();

    int rows = pattern_[0].rows;
    int cols = pattern_[0].cols;

    float i1 = 0;
    float i2 = 0;
    float i3 = 0;

    if( wrappedPhaseMap_.empty() )
        wrappedPhaseMap_.create(rows, cols, CV_32FC1);

    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            if( shadowMask_.at<uchar>(i, j) != 0 )
            {
                if( pattern_[0].type() == CV_8UC1 )
                {
                    i1 = pattern_[0].at<uchar>(i, j);
                    i2 = pattern_[1].at<uchar>(i, j);
                    i3 = pattern_[2].at<uchar>(i, j);
                }
                else if( pattern_[0].type() == CV_32FC1 )
                {
                    i1 = pattern_[0].at<float>(i, j);
                    i2 = pattern_[1].at<float>(i, j);
                    i3 = pattern_[2].at<float>(i, j);
                }
                float num = (1- cos(params.shiftValue)) * (i3 - i2);
                float den = sin(params.shiftValue) * (2 * i1 - i2 - i3);
                wrappedPhaseMap_.at<float>(i,j) = atan2(num, den);
            }
            else
            {
                wrappedPhaseMap_.at<float>(i,j) = 0;
            }
        }
    }
}

void SinusoidalPatternProfilometry_Impl::computeFapsPhaseMap( InputArray a,
                                                              InputArray b,
                                                              InputArray theta1,
                                                              InputArray theta2,
                                                              InputArray shadowMask,
                                                              OutputArray wrappedPhaseMap )
{
    Mat &a_ = *(Mat*) a.getObj();
    Mat &b_ = *(Mat*) b.getObj();
    Mat &theta1_ = *(Mat*) theta1.getObj();
    Mat &theta2_ = *(Mat*) theta2.getObj();
    Mat &wrappedPhaseMap_ = *(Mat*) wrappedPhaseMap.getObj();
    Mat &shadowMask_ = *(Mat*) shadowMask.getObj();

    int rows = a_.rows;
    int cols = a_.cols;

    if( wrappedPhaseMap_.empty() )
        wrappedPhaseMap_.create(rows, cols, CV_32FC1);

    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            if( shadowMask_.at<uchar>(i, j ) != 0 )
            {
                float num = (1 - cos(theta2_.at<float>(i, j))) * a_.at<float>(i, j) +
                            (1 - cos(theta1_.at<float>(i, j))) * b_.at<float>(i, j);

                float den = sin(theta1_.at<float>(i, j)) * b_.at<float>(i, j) -
                            sin(theta2_.at<float>(i, j)) * a_.at<float>(i, j);

                wrappedPhaseMap_.at<float>(i, j) = atan2(num, den);
            }
            else
            {
                wrappedPhaseMap_.at<float>(i, j) = 0;
            }
        }
    }
}

//compute shadow mask from three patterns. Valid pixels are lit at least by one pattern
void SinusoidalPatternProfilometry_Impl::computeShadowMask( InputArrayOfArrays patternImages,
                                                            OutputArray shadowMask )
{
    std::vector<Mat> &patternImages_ = *(std::vector<Mat>*) patternImages.getObj();
    Mat &shadowMask_ = *(Mat*) shadowMask.getObj();
    Mat mean;
    int rows = patternImages_[0].rows;
    int cols = patternImages_[0].cols;
    float i1, i2, i3;

    mean.create(rows, cols, CV_32FC1);

    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            i1 = (float) patternImages_[0].at<uchar>(i, j);
            i2 = (float) patternImages_[1].at<uchar>(i, j);
            i3 = (float) patternImages_[2].at<uchar>(i, j);
            mean.at<float>(i, j) = (i1 + i2 + i3) / 3;
        }
    }
    mean.convertTo(mean, CV_8UC1);
    threshold(mean, shadowMask_, 10, 255, 0);

}
// Compute the data modulation term according to the formula given in the reference paper
void SinusoidalPatternProfilometry_Impl::computeDataModulationTerm( InputArrayOfArrays patternImages,
                                                                    OutputArray dataModulationTerm,
                                                                    InputArray shadowMask )
{
    std::vector<Mat> &patternImages_ = *(std::vector<Mat>*) patternImages.getObj();
    Mat &dataModulationTerm_ = *(Mat*) dataModulationTerm.getObj();
    Mat &shadowMask_ = *(Mat*) shadowMask.getObj();
    int rows = patternImages_[0].rows;
    int cols = patternImages_[0].cols;
    float num = 0;
    float den = 0;
    float i1 = 0;
    float i2 = 0;
    float i3 = 0;

    int iOffset, jOffset;
    Mat dmt(rows, cols, CV_32FC1);
    Mat threshedDmt;

    if( dataModulationTerm_.empty() )
    {
            dataModulationTerm_.create(rows, cols, CV_8UC1);
    }
    if( shadowMask_.empty() )
    {
        shadowMask_.create(rows, cols, CV_8U);
        shadowMask_ = Scalar::all(255);
    }
    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            if( shadowMask_.at<uchar>(i, j) != 0 ){
                if( i - 2 == - 2 )
                {
                    iOffset = 0;
                }
                else if( i - 2 == - 1 )
                {
                    iOffset = -1;
                }
                else if( i - 2 + 4 == rows + 1 )
                {
                    iOffset = -3;
                }
                else
                {
                    iOffset = -2;
                }
                if( j - 2 == -2 )
                {
                    jOffset = 0;
                }
                else if( j - 2 == -1 )
                {
                    jOffset = -1;
                }
                else if( j - 2 + 4 == cols + 1 )
                {
                    jOffset = -3;
                }
                else
                {
                    jOffset = -2;
                }
                Mat roi = shadowMask_(Rect(j + jOffset, i + iOffset, 4, 4));
                Scalar nbrOfValidPixels = sum(roi);
                if( nbrOfValidPixels[0] < 14*255 )
                {
                    dmt.at<float>(i, j) = 0;
                }
                else
                {
                    i1 = patternImages_[0].at<uchar>(i, j);
                    i2 = patternImages_[1].at<uchar>(i, j);
                    i3 = patternImages_[2].at<uchar>(i, j);

                    num = sqrt(3 * ( i1 - i3 ) * ( i1 - i3 ) + ( 2 * i2 - i1 - i3 ) * ( 2 * i2 - i1 - i3 ));
                    den = i1 + i2 + i3;
                    dmt.at<float>(i, j) = 1 - num / den;
                }
            }
            else
            {
                dmt.at<float>(i, j) = 0;
            }
        }
    }
    Mat kernel(3, 3, CV_32F);
    kernel.at<float>(0, 0) = 1.f/16.f;
    kernel.at<float>(1, 0) = 2.f/16.f;
    kernel.at<float>(2, 0) = 1.f/16.f;

    kernel.at<float>(0, 1) = 2.f/16.f;
    kernel.at<float>(1, 1) = 4.f/16.f;
    kernel.at<float>(2, 1) = 2.f/16.f;

    kernel.at<float>(0, 2) = 1.f/16.f;
    kernel.at<float>(1, 2) = 2.f/16.f;
    kernel.at<float>(2, 2) = 1.f/16.f;

    Point anchor = Point(-1, -1);
    double delta = 0;
    int ddepth = -1;

    filter2D(dmt, dmt, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

    threshold(dmt, threshedDmt, 0.4, 1, THRESH_BINARY);
    threshedDmt.convertTo(dataModulationTerm_, CV_8UC1, 255, 0);
}

//Extract marker location on the DMT. Duplicates are removed
void SinusoidalPatternProfilometry_Impl::extractMarkersLocation( InputArray dataModulationTerm,
                                                                 std::vector<Point> &markersLocation )
{
    Mat &dmt = *(Mat*) dataModulationTerm.getObj();
    int rows = dmt.rows;
    int cols = dmt.cols;
    int halfRegionSize = 6;

    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            if( dmt.at<uchar>(i,j) != 0 )
            {
                bool addToVector = true;
                for(int k = 0; k < (int)markersLocation.size(); ++k)
                {
                    if( markersLocation[k].x - halfRegionSize < i &&
                        markersLocation[k].x + halfRegionSize > i &&
                        markersLocation[k].y - halfRegionSize < j &&
                        markersLocation[k].y + halfRegionSize > j ){
                        addToVector = false;
                    }
                }
                if(addToVector)
                {
                    Point temp(i,j);
                    markersLocation.push_back(temp);
                }
            }
        }
    }
}
void SinusoidalPatternProfilometry_Impl::convertToAbsolutePhaseMap( InputArrayOfArrays camPatterns,
                                                                    InputArray unwrappedProjPhaseMap,
                                                                    InputArray unwrappedCamPhaseMap,
                                                                    InputArray shadowMask,
                                                                    InputArray fundamentalMatrix )
{
    std::vector<Mat> &camPatterns_ = *(std::vector<Mat>*) camPatterns.getObj();
    (void) unwrappedCamPhaseMap;
    (void) unwrappedProjPhaseMap;

    Mat &fundamental = *(Mat*) fundamentalMatrix.getObj();

    Mat camDmt;

    std::vector<Point> markersLocation;

    computeDataModulationTerm(camPatterns_, camDmt, shadowMask);

    std::vector<Vec3f> epilines;
    computeCorrespondEpilines(params.markersLocation, 2, fundamental, epilines);

}
Ptr<SinusoidalPattern> SinusoidalPattern::create( Ptr<SinusoidalPattern::Params> params )
{
    return makePtr<SinusoidalPatternProfilometry_Impl>(*params);
}
}
}
