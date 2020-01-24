//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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
// Author: Tolga Birdal <tbirdal AT gmail.com>

#include "precomp.hpp"

namespace cv
{
namespace ppf_match_3d
{
static void subtractColumns(Mat srcPC, Vec3d& mean)
{
  int height = srcPC.rows;

  for (int i=0; i<height; i++)
  {
    float *row = srcPC.ptr<float>(i);
    {
      row[0]-=(float)mean[0];
      row[1]-=(float)mean[1];
      row[2]-=(float)mean[2];
    }
  }
}

// as in PCA
static void computeMeanCols(Mat srcPC, Vec3d& mean)
{
  int height = srcPC.rows;

  double mean1=0, mean2 = 0, mean3 = 0;

  for (int i=0; i<height; i++)
  {
    const float *row = srcPC.ptr<float>(i);
    {
      mean1 += (double)row[0];
      mean2 += (double)row[1];
      mean3 += (double)row[2];
    }
  }

  mean1/=(double)height;
  mean2/=(double)height;
  mean3/=(double)height;

  mean[0] = mean1;
  mean[1] = mean2;
  mean[2] = mean3;
}

// as in PCA
/*static void subtractMeanFromColumns(Mat srcPC, Vec3d& mean)
{
    computeMeanCols(srcPC, mean);
    subtractColumns(srcPC, mean);
}*/

// compute the average distance to the origin
static double computeDistToOrigin(Mat srcPC)
{
  int height = srcPC.rows;
  double dist = 0;

  for (int i=0; i<height; i++)
  {
    const float *row = srcPC.ptr<float>(i);
    dist += sqrt(row[0]*row[0]+row[1]*row[1]+row[2]*row[2]);
  }

  return dist;
}

// From numerical receipes: Finds the median of an array
static float medianF(float arr[], int n)
{
  int low, high ;
  int median;
  int middle, ll, hh;

  low = 0 ;
  high = n-1 ;
  median = (low + high) >>1;
  for (;;)
  {
    if (high <= low) /* One element only */
      return arr[median] ;

    if (high == low + 1)
    {
      /* Two elements only */
      if (arr[low] > arr[high])
        std::swap(arr[low], arr[high]) ;
      return arr[median] ;
    }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) >>1;
    if (arr[middle] > arr[high])
      std::swap(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])
      std::swap(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])
      std::swap(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    std::swap(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;)
    {
      do
        ll++;
      while (arr[low] > arr[ll]) ;
      do
        hh--;
      while (arr[hh]  > arr[low]) ;

      if (hh < ll)
        break;

      std::swap(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    std::swap(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
      low = ll;
    if (hh >= median)
      high = hh - 1;
  }
}

static float getRejectionThreshold(float* r, int m, float outlierScale)
{
  float* t=(float*)calloc(m, sizeof(float));
  int i=0;
  float s=0, medR, threshold;

  memcpy(t, r, m*sizeof(float));
  medR=medianF(t, m);

  for (i=0; i<m; i++)
    t[i] = (float)fabs((double)r[i]-(double)medR);

  s = 1.48257968f * medianF(t, m);

  threshold = (outlierScale*s+medR);

  free(t);
  return threshold;
}

// Kok Lim Low's linearization
static void minimizePointToPlaneMetric(Mat Src, Mat Dst, Vec3d& rpy, Vec3d& t)
{
  //Mat sub = Dst - Src;
  Mat A = Mat(Src.rows, 6, CV_64F);
  Mat b = Mat(Src.rows, 1, CV_64F);
  Mat rpy_t;

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<Src.rows; i++)
  {
    const Vec3d srcPt(Src.ptr<double>(i));
    const Vec3d dstPt(Dst.ptr<double>(i));
    const Vec3d normals(Dst.ptr<double>(i) + 3);
    const Vec3d sub = dstPt - srcPt;
    const Vec3d axis = srcPt.cross(normals);

    *b.ptr<double>(i) = sub.dot(normals);
    hconcat(axis.reshape<1, 3>(), normals.reshape<1, 3>(), A.row(i));
  }

  cv::solve(A, b, rpy_t, DECOMP_SVD);
  rpy_t.rowRange(0, 3).copyTo(rpy);
  rpy_t.rowRange(3, 6).copyTo(t);
}

static void getTransformMat(Vec3d& euler, Vec3d& t, Matx44d& Pose)
{
  Matx33d R;
  eulerToDCM(euler, R);
  rtToPose(R, t, Pose);
}

/* Fast way to look up the duplicates
duplicates is pre-allocated
make sure that the max element in array will not exceed maxElement
*/
static hashtable_int* getHashtable(int* data, size_t length, int numMaxElement)
{
  hashtable_int* hashtable = hashtableCreate(static_cast<size_t>(numMaxElement*2), 0);
  for (size_t i = 0; i < length; i++)
  {
    const KeyType key = (KeyType)data[i];
    hashtableInsertHashed(hashtable, key+1, reinterpret_cast<void*>(i+1));
  }

  return hashtable;
}

// source point clouds are assumed to contain their normals
int ICP::registerModelToScene(const Mat& srcPC, const Mat& dstPC, double& residual, Matx44d& pose)
{
  int n = srcPC.rows;
  CV_CheckGT(n, 0, "");

  const bool useRobustReject = m_rejectionScale>0;

  Mat srcTemp = srcPC.clone();
  Mat dstTemp = dstPC.clone();
  Vec3d meanSrc, meanDst;
  computeMeanCols(srcTemp, meanSrc);
  computeMeanCols(dstTemp, meanDst);
  Vec3d meanAvg = 0.5 * (meanSrc + meanDst);
  subtractColumns(srcTemp, meanAvg);
  subtractColumns(dstTemp, meanAvg);

  double distSrc = computeDistToOrigin(srcTemp);
  double distDst = computeDistToOrigin(dstTemp);

  double scale = (double)n / ((distSrc + distDst)*0.5);

  srcTemp(cv::Range(0, srcTemp.rows), cv::Range(0,3)) *= scale;
  dstTemp(cv::Range(0, dstTemp.rows), cv::Range(0,3)) *= scale;

  Mat srcPC0 = srcTemp;
  Mat dstPC0 = dstTemp;

  // initialize pose
  pose = Matx44d::eye();

  Mat M = Mat::eye(4,4,CV_64F);

  double tempResidual = 0;


  // walk the pyramid
  for (int level = m_numLevels-1; level >=0; level--)
  {
    const int numSamples = divUp(n, 1 << level);
    const double TolP = m_tolerance*(double)(level+1)*(level+1);
    const int MaxIterationsPyr = cvRound((double)m_maxIterations/(level+1));

    // Obtain the sampled point clouds for this level: Also rotates the normals
    Mat srcPCT = transformPCPose(srcPC0, pose);

    const int sampleStep = cvRound((double)n/(double)numSamples);

    srcPCT = samplePCUniform(srcPCT, sampleStep);
    /*
    Tolga Birdal thinks that downsampling the scene points might decrease the accuracy.
    Hamdi Sahloul, however, noticed that accuracy increased (pose residual decreased slightly).
    */
    Mat dstPCS = samplePCUniform(dstPC0, sampleStep);
    void* flann = indexPCFlann(dstPCS);

    double fval_old=9999999999;
    double fval_perc=0;
    double fval_min=9999999999;
    Mat Src_Moved = srcPCT.clone();

    int i=0;

    size_t numElSrc = (size_t)Src_Moved.rows;
    int sizesResult[2] = {(int)numElSrc, 1};
    float* distances = new float[numElSrc];
    int* indices = new int[numElSrc];

    Mat Indices(2, sizesResult, CV_32S, indices, 0);
    Mat Distances(2, sizesResult, CV_32F, distances, 0);

    // use robust weighting for outlier treatment
    int* indicesModel = new int[numElSrc];
    int* indicesScene = new int[numElSrc];

    int* newI = new int[numElSrc];
    int* newJ = new int[numElSrc];

    Matx44d PoseX = Matx44d::eye();

    while ( (!(fval_perc<(1+TolP) && fval_perc>(1-TolP))) && i<MaxIterationsPyr)
    {
      uint di=0, selInd = 0;

      queryPCFlann(flann, Src_Moved, Indices, Distances);

      for (di=0; di<numElSrc; di++)
      {
        newI[di] = di;
        newJ[di] = indices[di];
      }

      if (useRobustReject)
      {
        int numInliers = 0;
        float threshold = getRejectionThreshold(distances, Distances.rows, m_rejectionScale);
        Mat acceptInd = Distances<threshold;

        uchar *accPtr = (uchar*)acceptInd.data;
        for (int l=0; l<acceptInd.rows; l++)
        {
          if (accPtr[l])
          {
            newI[numInliers] = l;
            newJ[numInliers] = indices[l];
            numInliers++;
          }
        }
        numElSrc=numInliers;
      }

      // Step 2: Picky ICP
      // Among the resulting corresponding pairs, if more than one scene point p_i
      // is assigned to the same model point m_j, then select p_i that corresponds
      // to the minimum distance

      hashtable_int* duplicateTable = getHashtable(newJ, numElSrc, dstPCS.rows);

      for (di=0; di<duplicateTable->size; di++)
      {
        hashnode_i *node = duplicateTable->nodes[di];

        if (node)
        {
          // select the first node
          size_t idx = reinterpret_cast<size_t>(node->data)-1, dn=0;
          int dup = (int)node->key-1;
          size_t minIdxD = idx;
          float minDist = distances[idx];

          while ( node )
          {
            idx = reinterpret_cast<size_t>(node->data)-1;

            if (distances[idx] < minDist)
            {
              minDist = distances[idx];
              minIdxD = idx;
            }

            node = node->next;
            dn++;
          }

          indicesModel[ selInd ] = newI[ minIdxD ];
          indicesScene[ selInd ] = dup ;
          selInd++;
        }
      }

      hashtableDestroy(duplicateTable);

      if (selInd >= 6)
      {

        Mat Src_Match = Mat(selInd, srcPCT.cols, CV_64F);
        Mat Dst_Match = Mat(selInd, srcPCT.cols, CV_64F);

        for (di=0; di<selInd; di++)
        {
          const int indModel = indicesModel[di];
          const int indScene = indicesScene[di];
          const float *srcPt = srcPCT.ptr<float>(indModel);
          const float *dstPt = dstPCS.ptr<float>(indScene);
          double *srcMatchPt = Src_Match.ptr<double>(di);
          double *dstMatchPt = Dst_Match.ptr<double>(di);
          int ci=0;

          for (ci=0; ci<srcPCT.cols; ci++)
          {
            srcMatchPt[ci] = (double)srcPt[ci];
            dstMatchPt[ci] = (double)dstPt[ci];
          }
        }

        Vec3d rpy, t;
        minimizePointToPlaneMetric(Src_Match, Dst_Match, rpy, t);
        if (cvIsNaN(cv::trace(rpy)) || cvIsNaN(cv::norm(t)))
          break;
        getTransformMat(rpy, t, PoseX);
        Src_Moved = transformPCPose(srcPCT, PoseX);

        double fval = cv::norm(Src_Match, Dst_Match)/(double)(Src_Moved.rows);

        // Calculate change in error between iterations
        fval_perc=fval/fval_old;

        // Store error value
        fval_old=fval;

        if (fval < fval_min)
          fval_min = fval;
      }
      else
        break;

      i++;

    }

    pose = PoseX * pose;
    residual = tempResidual;

    delete[] newI;
    delete[] newJ;
    delete[] indicesModel;
    delete[] indicesScene;
    delete[] distances;
    delete[] indices;

    tempResidual = fval_min;
    destroyFlann(flann);
  }

  Matx33d Rpose;
  Vec3d Cpose;
  poseToRT(pose, Rpose, Cpose);
  Cpose = Cpose / scale + meanAvg - Rpose * meanAvg;
  rtToPose(Rpose, Cpose, pose);

  residual = tempResidual;

  return 0;
}

// source point clouds are assumed to contain their normals
int ICP::registerModelToScene(const Mat& srcPC, const Mat& dstPC, std::vector<Pose3DPtr>& poses)
{
  #if defined _OPENMP
  #pragma omp parallel for
  #endif
  for (int i=0; i<(int)poses.size(); i++)
  {
    Matx44d poseICP = Matx44d::eye();
    Mat srcTemp = transformPCPose(srcPC, poses[i]->pose);
    registerModelToScene(srcTemp, dstPC, poses[i]->residual, poseICP);
    poses[i]->appendPose(poseICP);
  }
  return 0;
}

} // namespace ppf_match_3d

} // namespace cv
