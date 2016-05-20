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
static void subtractColumns(Mat srcPC, double mean[3])
{
  int height = srcPC.rows;

  for (int i=0; i<height; i++)
  {
    float *row = (float*)(&srcPC.data[i*srcPC.step]);
    {
      row[0]-=(float)mean[0];
      row[1]-=(float)mean[1];
      row[2]-=(float)mean[2];
    }
  }
}

// as in PCA
static void computeMeanCols(Mat srcPC, double mean[3])
{
  int height = srcPC.rows;

  double mean1=0, mean2 = 0, mean3 = 0;

  for (int i=0; i<height; i++)
  {
    const float *row = (float*)(&srcPC.data[i*srcPC.step]);
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
/*static void subtractMeanFromColumns(Mat srcPC, double mean[3])
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
    const float *row = (float*)(&srcPC.data[i*srcPC.step]);
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
static void minimizePointToPlaneMetric(Mat Src, Mat Dst, Mat& X)
{
  //Mat sub = Dst - Src;
  Mat A = Mat(Src.rows, 6, CV_64F);
  Mat b = Mat(Src.rows, 1, CV_64F);

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<Src.rows; i++)
  {
    const double *srcPt = (double*)&Src.data[i*Src.step];
    const double *dstPt = (double*)&Dst.data[i*Dst.step];
    const double *normals = &dstPt[3];
    double *bVal = (double*)&b.data[i*b.step];
    double *aRow = (double*)&A.data[i*A.step];

    const double sub[3]={dstPt[0]-srcPt[0], dstPt[1]-srcPt[1], dstPt[2]-srcPt[2]};

    *bVal = TDot3(sub, normals);
    TCross(srcPt, normals, aRow);

    aRow[3] = normals[0];
    aRow[4] = normals[1];
    aRow[5] = normals[2];
  }

  cv::solve(A, b, X, DECOMP_SVD);
}


static void getTransformMat(Mat X, double Pose[16])
{
  Mat DCM;
  double *r1, *r2, *r3;
  double* x = (double*)X.data;

  const double sx = sin(x[0]);
  const double cx = cos(x[0]);
  const double sy = sin(x[1]);
  const double cy = cos(x[1]);
  const double sz = sin(x[2]);
  const double cz = cos(x[2]);

  Mat R1 = Mat::eye(3,3, CV_64F);
  Mat R2 = Mat::eye(3,3, CV_64F);
  Mat R3 = Mat::eye(3,3, CV_64F);

  r1= (double*)R1.data;
  r2= (double*)R2.data;
  r3= (double*)R3.data;

  r1[4]= cx;
  r1[5]= -sx;
  r1[7]= sx;
  r1[8]= cx;

  r2[0]= cy;
  r2[2]= sy;
  r2[6]= -sy;
  r2[8]= cy;

  r3[0]= cz;
  r3[1]= -sz;
  r3[3]= sz;
  r3[4]= cz;

  DCM = R1*(R2*R3);

  Pose[0] = DCM.at<double>(0,0);
  Pose[1] = DCM.at<double>(0,1);
  Pose[2] = DCM.at<double>(0,2);
  Pose[4] = DCM.at<double>(1,0);
  Pose[5] = DCM.at<double>(1,1);
  Pose[6] = DCM.at<double>(1,2);
  Pose[8] = DCM.at<double>(2,0);
  Pose[9] = DCM.at<double>(2,1);
  Pose[10] = DCM.at<double>(2,2);
  Pose[3]=x[3];
  Pose[7]=x[4];
  Pose[11]=x[5];
  Pose[15]=1;
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
int ICP::registerModelToScene(const Mat& srcPC, const Mat& dstPC, double& residual, double pose[16])
{
  int n = srcPC.rows;

  const bool useRobustReject = m_rejectionScale>0;

  Mat srcTemp = srcPC.clone();
  Mat dstTemp = dstPC.clone();
  double meanSrc[3], meanDst[3];
  computeMeanCols(srcTemp, meanSrc);
  computeMeanCols(dstTemp, meanDst);
  double meanAvg[3]={0.5*(meanSrc[0]+meanDst[0]), 0.5*(meanSrc[1]+meanDst[1]), 0.5*(meanSrc[2]+meanDst[2])};
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
  matrixIdentity(4, pose);

  void* flann = indexPCFlann(dstPC0);
  Mat M = Mat::eye(4,4,CV_64F);

  double tempResidual = 0;


  // walk the pyramid
  for (int level = m_numLevels-1; level >=0; level--)
  {
    const double impact = 2;
    double div = pow((double)impact, (double)level);
    //double div2 = div*div;
    const int numSamples = cvRound((double)(n/(div)));
    const double TolP = m_tolerance*(double)(level+1)*(level+1);
    const int MaxIterationsPyr = cvRound((double)m_maxIterations/(level+1));

    // Obtain the sampled point clouds for this level: Also rotates the normals
    Mat srcPCT = transformPCPose(srcPC0, pose);

    const int sampleStep = cvRound((double)n/(double)numSamples);
    std::vector<int> srcSampleInd;

    /*
    Note by Tolga Birdal
    Downsample the model point clouds. If more optimization is required,
    one could also downsample the scene points, but I think this might
    decrease the accuracy. That's why I won't be implementing it at this
    moment.

    Also note that you have to compute a KD-tree for each level.
    */
    srcPCT = samplePCUniformInd(srcPCT, sampleStep, srcSampleInd);

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

    double PoseX[16]={0};
    matrixIdentity(4, PoseX);

    while ( (!(fval_perc<(1+TolP) && fval_perc>(1-TolP))) && i<MaxIterationsPyr)
    {
      size_t di=0, selInd = 0;

      queryPCFlann(flann, Src_Moved, Indices, Distances);

      for (di=0; di<numElSrc; di++)
      {
        newI[di] = (int)di;
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

      hashtable_int* duplicateTable = getHashtable(newJ, numElSrc, dstPC0.rows);

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

      if (selInd)
      {

        Mat Src_Match = Mat((int)selInd, srcPCT.cols, CV_64F);
        Mat Dst_Match = Mat((int)selInd, srcPCT.cols, CV_64F);

        for (di=0; di<selInd; di++)
        {
          const int indModel = indicesModel[di];
          const int indScene = indicesScene[di];
          const float *srcPt = (float*)&srcPCT.data[indModel*srcPCT.step];
          const float *dstPt = (float*)&dstPC0.data[indScene*dstPC0.step];
          double *srcMatchPt = (double*)&Src_Match.data[di*Src_Match.step];
          double *dstMatchPt = (double*)&Dst_Match.data[di*Dst_Match.step];
          int ci=0;

          for (ci=0; ci<srcPCT.cols; ci++)
          {
            srcMatchPt[ci] = (double)srcPt[ci];
            dstMatchPt[ci] = (double)dstPt[ci];
          }
        }

        Mat X;
        minimizePointToPlaneMetric(Src_Match, Dst_Match, X);

        getTransformMat(X, PoseX);
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

    double TempPose[16];
    matrixProduct44(PoseX, pose, TempPose);

    // no need to copy the last 4 rows
    for (int c=0; c<12; c++)
      pose[c] = TempPose[c];

    residual = tempResidual;

    delete[] newI;
    delete[] newJ;
    delete[] indicesModel;
    delete[] indicesScene;
    delete[] distances;
    delete[] indices;

    tempResidual = fval_min;
  }

  // Pose(1:3, 4) = Pose(1:3, 4)./scale;
  pose[3] = pose[3]/scale + meanAvg[0];
  pose[7] = pose[7]/scale + meanAvg[1];
  pose[11] = pose[11]/scale + meanAvg[2];

  // In MATLAB this would be : Pose(1:3, 4) = Pose(1:3, 4)./scale + meanAvg' - Pose(1:3, 1:3)*meanAvg';
  double Rpose[9], Cpose[3];
  poseToR(pose, Rpose);
  matrixProduct331(Rpose, meanAvg, Cpose);
  pose[3] -= Cpose[0];
  pose[7] -= Cpose[1];
  pose[11] -= Cpose[2];

  residual = tempResidual;

  destroyFlann(flann);
  return 0;
}

// source point clouds are assumed to contain their normals
int ICP::registerModelToScene(const Mat& srcPC, const Mat& dstPC, std::vector<Pose3DPtr>& poses)
{
  for (size_t i=0; i<poses.size(); i++)
  {
    double poseICP[16]={0};
    Mat srcTemp = transformPCPose(srcPC, poses[i]->pose);
    registerModelToScene(srcTemp, dstPC, poses[i]->residual, poseICP);
    poses[i]->appendPose(poseICP);
  }
  return 0;
}

} // namespace ppf_match_3d

} // namespace cv
