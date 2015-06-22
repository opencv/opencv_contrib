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

typedef cv::flann::L2<float> Distance_32F;
typedef cv::flann::GenericIndex< Distance_32F > FlannIndex;

void shuffle(int *array, size_t n);
Mat genRandomMat(int rows, int cols, double mean, double stddev, int type);
void getRandQuat(double q[4]);
void getRandomRotation(double R[9]);
void meanCovLocalPC(const float* pc, const int ws, const int point_count, double CovMat[3][3], double Mean[4]);
void meanCovLocalPCInd(const float* pc, const int* Indices, const int ws, const int point_count, double CovMat[3][3], double Mean[4]);

Mat loadPLYSimple(const char* fileName, int withNormals)
{
  Mat cloud;
  int numVertices=0;

  std::ifstream ifs(fileName);

  if (!ifs.is_open())
  {
    printf("Cannot open file...\n");
    return Mat();
  }

  std::string str;
  while (str.substr(0, 10) !="end_header")
  {
    std::string entry = str.substr(0, 14);
    if (entry == "element vertex")
    {
      numVertices = atoi(str.substr(15, str.size()-15).c_str());
    }
    std::getline(ifs, str);
  }

  if (withNormals)
    cloud=Mat(numVertices, 6, CV_32FC1);
  else
    cloud=Mat(numVertices, 3, CV_32FC1);

  for (int i = 0; i < numVertices; i++)
  {
    float* data = (float*)(&cloud.data[i*cloud.step[0]]);
    if (withNormals)
    {
      ifs >> data[0] >> data[1] >> data[2] >> data[3] >> data[4] >> data[5];

      // normalize to unit norm
      double norm = sqrt(data[3]*data[3] + data[4]*data[4] + data[5]*data[5]);
      if (norm>0.00001)
      {
        data[3]/=(float)norm;
        data[4]/=(float)norm;
        data[5]/=(float)norm;
      }
    }
    else
    {
      ifs >> data[0] >> data[1] >> data[2];
    }
  }

  //cloud *= 5.0f;
  return cloud;
}

void writePLY(Mat PC, const char* FileName)
{
  std::ofstream outFile( FileName );

  if ( !outFile )
  {
    //cerr << "Error opening output file: " << FileName << "!" << endl;
    printf("Error opening output file: %s!\n", FileName);
    exit( 1 );
  }

  ////
  // Header
  ////

  const int pointNum = ( int ) PC.rows;
  const int vertNum  = ( int ) PC.cols;

  outFile << "ply" << std::endl;
  outFile << "format ascii 1.0" << std::endl;
  outFile << "element vertex " << pointNum << std::endl;
  outFile << "property float x" << std::endl;
  outFile << "property float y" << std::endl;
  outFile << "property float z" << std::endl;
  if (vertNum==6)
  {
    outFile << "property float nx" << std::endl;
    outFile << "property float ny" << std::endl;
    outFile << "property float nz" << std::endl;
  }
  outFile << "end_header" << std::endl;

  ////
  // Points
  ////

  for ( int pi = 0; pi < pointNum; ++pi )
  {
    const float* point = (float*)(&PC.data[ pi*PC.step ]);

    outFile << point[0] << " "<<point[1]<<" "<<point[2];

    if (vertNum==6)
    {
      outFile<<" " << point[3] << " "<<point[4]<<" "<<point[5];
    }

    outFile << std::endl;
  }

  return;
}

void writePLYVisibleNormals(Mat PC, const char* FileName)
{
  std::ofstream outFile(FileName);

  if (!outFile)
  {
    //cerr << "Error opening output file: " << FileName << "!" << endl;
    printf("Error opening output file: %s!\n", FileName);
    exit(1);
  }

  ////
  // Header
  ////

  const int pointNum = (int)PC.rows;
  const int vertNum = (int)PC.cols;
  const bool hasNormals = vertNum == 6;

  outFile << "ply" << std::endl;
  outFile << "format ascii 1.0" << std::endl;
  outFile << "element vertex " << (hasNormals? 2*pointNum:pointNum) << std::endl;
  outFile << "property float x" << std::endl;
  outFile << "property float y" << std::endl;
  outFile << "property float z" << std::endl;
  if (hasNormals)
  {
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
  }
  outFile << "end_header" << std::endl;

  ////
  // Points
  ////

  for (int pi = 0; pi < pointNum; ++pi)
  {
    const float* point = (float*)(&PC.data[pi*PC.step]);

    outFile << point[0] << " " << point[1] << " " << point[2];

    if (hasNormals)
    {
      outFile << " 127 127 127" << std::endl;
      outFile << point[0]+point[3] << " " << point[1]+point[4] << " " << point[2]+point[5];
      outFile << " 255 0 0";
    }

    outFile << std::endl;
  }

  return;
}

Mat samplePCUniform(Mat PC, int sampleStep)
{
  int numRows = PC.rows/sampleStep;
  Mat sampledPC = Mat(numRows, PC.cols, PC.type());

  int c=0;
  for (int i=0; i<PC.rows && c<numRows; i+=sampleStep)
  {
    PC.row(i).copyTo(sampledPC.row(c++));
  }

  return sampledPC;
}

Mat samplePCUniformInd(Mat PC, int sampleStep, std::vector<int> &indices)
{
  int numRows = cvRound((double)PC.rows/(double)sampleStep);
  indices.resize(numRows);
  Mat sampledPC = Mat(numRows, PC.cols, PC.type());

  int c=0;
  for (int i=0; i<PC.rows && c<numRows; i+=sampleStep)
  {
    indices[c] = i;
    PC.row(i).copyTo(sampledPC.row(c++));
  }

  return sampledPC;
}

void* indexPCFlann(Mat pc)
{
  Mat dest_32f;
  pc.colRange(0,3).copyTo(dest_32f);
  return new FlannIndex(dest_32f, cvflann::KDTreeSingleIndexParams(8));
}

void destroyFlann(void* flannIndex)
{
  delete ((FlannIndex*)flannIndex);
}

// For speed purposes this function assumes that PC, Indices and Distances are created with continuous structures
void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances)
{
  queryPCFlann(flannIndex, pc, indices, distances, 1);
}

void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, const int numNeighbors)
{
  Mat obj_32f;
  pc.colRange(0, 3).copyTo(obj_32f);
  ((FlannIndex*)flannIndex)->knnSearch(obj_32f, indices, distances, numNeighbors, cvflann::SearchParams(32));
}

// uses a volume instead of an octree
// TODO: Right now normals are required.
// This is much faster than sample_pc_octree
Mat samplePCByQuantization(Mat pc, float xrange[2], float yrange[2], float zrange[2], float sampleStep, int weightByCenter)
{
  std::vector< std::vector<int> > map;

  int numSamplesDim = (int)(1.0/sampleStep);

  float xr = xrange[1] - xrange[0];
  float yr = yrange[1] - yrange[0];
  float zr = zrange[1] - zrange[0];

  int numPoints = 0;

  map.resize((numSamplesDim+1)*(numSamplesDim+1)*(numSamplesDim+1));

  // OpenMP might seem like a good idea, but it didn't speed this up for me
  //#pragma omp parallel for
  for (int i=0; i<pc.rows; i++)
  {
    const float* point = (float*)(&pc.data[i * pc.step]);

    // quantize a point
    const int xCell =(int) ((float)numSamplesDim*(point[0]-xrange[0])/xr);
    const int yCell =(int) ((float)numSamplesDim*(point[1]-yrange[0])/yr);
    const int zCell =(int) ((float)numSamplesDim*(point[2]-zrange[0])/zr);
    const int index = xCell*numSamplesDim*numSamplesDim+yCell*numSamplesDim+zCell;

    /*#pragma omp critical
        {*/
    map[index].push_back(i);
    //  }
  }

  for (unsigned int i=0; i<map.size(); i++)
  {
    numPoints += (map[i].size()>0);
  }

  Mat pcSampled = Mat(numPoints, pc.cols, CV_32F);
  int c = 0;

  for (unsigned int i=0; i<map.size(); i++)
  {
    double px=0, py=0, pz=0;
    double nx=0, ny=0, nz=0;

    std::vector<int> curCell = map[i];
    int cn = (int)curCell.size();
    if (cn>0)
    {
      if (weightByCenter)
      {
        int xCell, yCell, zCell;
        double xc, yc, zc;
        double weightSum = 0 ;
        zCell = i % numSamplesDim;
        yCell = ((i-zCell)/numSamplesDim) % numSamplesDim;
        xCell = ((i-zCell-yCell*numSamplesDim)/(numSamplesDim*numSamplesDim));

        xc = ((double)xCell+0.5) * (double)xr/numSamplesDim + (double)xrange[0];
        yc = ((double)yCell+0.5) * (double)yr/numSamplesDim + (double)yrange[0];
        zc = ((double)zCell+0.5) * (double)zr/numSamplesDim + (double)zrange[0];

        for (int j=0; j<cn; j++)
        {
          const int ptInd = curCell[j];
          float* point = (float*)(&pc.data[ptInd * pc.step]);
          const double dx = point[0]-xc;
          const double dy = point[1]-yc;
          const double dz = point[2]-zc;
          const double d = sqrt(dx*dx+dy*dy+dz*dz);
          double w = 0;

          if (d>EPS)
          {
            // it is possible to use different weighting schemes.
            // inverse weigthing was just good for me
            // exp( - (distance/h)**2 )
            //const double w = exp(-d*d);
            w = 1.0/d;
          }

          //float weights[3]={1,1,1};
          px += w*(double)point[0];
          py += w*(double)point[1];
          pz += w*(double)point[2];
          nx += w*(double)point[3];
          ny += w*(double)point[4];
          nz += w*(double)point[5];

          weightSum+=w;
        }
        px/=(double)weightSum;
        py/=(double)weightSum;
        pz/=(double)weightSum;
        nx/=(double)weightSum;
        ny/=(double)weightSum;
        nz/=(double)weightSum;
      }
      else
      {
        for (int j=0; j<cn; j++)
        {
          const int ptInd = curCell[j];
          float* point = (float*)(&pc.data[ptInd * pc.step]);

          px += (double)point[0];
          py += (double)point[1];
          pz += (double)point[2];
          nx += (double)point[3];
          ny += (double)point[4];
          nz += (double)point[5];
        }

        px/=(double)cn;
        py/=(double)cn;
        pz/=(double)cn;
        nx/=(double)cn;
        ny/=(double)cn;
        nz/=(double)cn;

      }

      float *pcData = (float*)(&pcSampled.data[c*pcSampled.step[0]]);
      pcData[0]=(float)px;
      pcData[1]=(float)py;
      pcData[2]=(float)pz;

      // normalize the normals
      double norm = sqrt(nx*nx+ny*ny+nz*nz);

      if (norm>EPS)
      {
        pcData[3]=(float)(nx/norm);
        pcData[4]=(float)(ny/norm);
        pcData[5]=(float)(nz/norm);
      }
      //#pragma omp atomic
      c++;

      curCell.clear();
    }
  }

  map.clear();
  return pcSampled;
}

void shuffle(int *array, size_t n)
{
  size_t i;
  for (i = 0; i < n - 1; i++)
  {
    size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
    int t = array[j];
    array[j] = array[i];
    array[i] = t;
  }
}

// compute the standard bounding box
void computeBboxStd(Mat pc, float xRange[2], float yRange[2], float zRange[2])
{
  Mat pcPts = pc.colRange(0, 3);
  int num = pcPts.rows;

  float* points = (float*)pcPts.data;

  xRange[0] = points[0];
  xRange[1] = points[0];
  yRange[0] = points[1];
  yRange[1] = points[1];
  zRange[0] = points[2];
  zRange[1] = points[2];

  for  ( int  ind = 0; ind < num; ind++ )
  {
    const float* row = (float*)(pcPts.data + (ind * pcPts.step));
    const float x = row[0];
    const float y = row[1];
    const float z = row[2];

    if (x<xRange[0])
      xRange[0]=x;
    if (x>xRange[1])
      xRange[1]=x;

    if (y<yRange[0])
      yRange[0]=y;
    if (y>yRange[1])
      yRange[1]=y;

    if (z<zRange[0])
      zRange[0]=z;
    if (z>zRange[1])
      zRange[1]=z;
  }
}

Mat normalizePCCoeff(Mat pc, float scale, float* Cx, float* Cy, float* Cz, float* MinVal, float* MaxVal)
{
  double minVal=0, maxVal=0;

  Mat x,y,z, pcn;
  pc.col(0).copyTo(x);
  pc.col(1).copyTo(y);
  pc.col(2).copyTo(z);

  float cx = (float) cv::mean(x).val[0];
  float cy = (float) cv::mean(y).val[0];
  float cz = (float) cv::mean(z).val[0];

  cv::minMaxIdx(pc, &minVal, &maxVal);

  x=x-cx;
  y=y-cy;
  z=z-cz;
  pcn.create(pc.rows, 3, CV_32FC1);
  x.copyTo(pcn.col(0));
  y.copyTo(pcn.col(1));
  z.copyTo(pcn.col(2));

  cv::minMaxIdx(pcn, &minVal, &maxVal);
  pcn=(float)scale*(pcn)/((float)maxVal-(float)minVal);

  *MinVal=(float)minVal;
  *MaxVal=(float)maxVal;
  *Cx=(float)cx;
  *Cy=(float)cy;
  *Cz=(float)cz;

  return pcn;
}

Mat transPCCoeff(Mat pc, float scale, float Cx, float Cy, float Cz, float MinVal, float MaxVal)
{
  Mat x,y,z, pcn;
  pc.col(0).copyTo(x);
  pc.col(1).copyTo(y);
  pc.col(2).copyTo(z);

  x=x-Cx;
  y=y-Cy;
  z=z-Cz;
  pcn.create(pc.rows, 3, CV_32FC1);
  x.copyTo(pcn.col(0));
  y.copyTo(pcn.col(1));
  z.copyTo(pcn.col(2));

  pcn=(float)scale*(pcn)/((float)MaxVal-(float)MinVal);

  return pcn;
}

Mat transformPCPose(Mat pc, double Pose[16])
{
  Mat pct = Mat(pc.rows, pc.cols, CV_32F);

  double R[9], t[3];
  poseToRT(Pose, R, t);

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<pc.rows; i++)
  {
    const float *pcData = (float*)(&pc.data[i*pc.step]);
    float *pcDataT = (float*)(&pct.data[i*pct.step]);
    const float *n1 = &pcData[3];
    float *nT = &pcDataT[3];

    double p[4] = {(double)pcData[0], (double)pcData[1], (double)pcData[2], 1};
    double p2[4];

    matrixProduct441(Pose, p, p2);

    // p2[3] should normally be 1
    if (fabs(p2[3])>EPS)
    {
      pcDataT[0] = (float)(p2[0]/p2[3]);
      pcDataT[1] = (float)(p2[1]/p2[3]);
      pcDataT[2] = (float)(p2[2]/p2[3]);
    }

    // If the point cloud has normals,
    // then rotate them as well
    if (pc.cols == 6)
    {
      double n[3] = { (double)n1[0], (double)n1[1], (double)n1[2] }, n2[3];

      matrixProduct331(R, n, n2);
      double nNorm = sqrt(n2[0]*n2[0]+n2[1]*n2[1]+n2[2]*n2[2]);

      if (nNorm>EPS)
      {
        nT[0]=(float)(n2[0]/nNorm);
        nT[1]=(float)(n2[1]/nNorm);
        nT[2]=(float)(n2[2]/nNorm);
      }
    }
  }

  return pct;
}

Mat genRandomMat(int rows, int cols, double mean, double stddev, int type)
{
  Mat meanMat = mean*Mat::ones(1,1,type);
  Mat sigmaMat= stddev*Mat::ones(1,1,type);
  RNG rng(time(0));
  Mat matr(rows, cols,type);
  rng.fill(matr, RNG::NORMAL, meanMat, sigmaMat);

  return matr;
}

void getRandQuat(double q[4])
{
  q[0] = (float)rand()/(float)(RAND_MAX);
  q[1] = (float)rand()/(float)(RAND_MAX);
  q[2] = (float)rand()/(float)(RAND_MAX);
  q[3] = (float)rand()/(float)(RAND_MAX);

  double n = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  q[0]/=n;
  q[1]/=n;
  q[2]/=n;
  q[3]/=n;

  q[0]=fabs(q[0]);
}

void getRandomRotation(double R[9])
{
  double q[4];
  getRandQuat(q);
  quatToDCM(q, R);
}

void getRandomPose(double Pose[16])
{
  double R[9], t[3];

  srand((unsigned int)time(0));
  getRandomRotation(R);

  t[0] = (float)rand()/(float)(RAND_MAX);
  t[1] = (float)rand()/(float)(RAND_MAX);
  t[2] = (float)rand()/(float)(RAND_MAX);

  rtToPose(R,t,Pose);
}

Mat addNoisePC(Mat pc, double scale)
{
  Mat randT = genRandomMat(pc.rows,pc.cols,0,scale,CV_32FC1);
  return randT + pc;
}

/*
The routines below use the eigenvectors of the local covariance matrix
to compute the normals of a point cloud.
The algorithm uses FLANN and Joachim Kopp's fast 3x3 eigenvector computations
to improve accuracy and increase speed
Also, view point flipping as in point cloud library is implemented
*/

void meanCovLocalPC(const float* pc, const int ws, const int point_count, double CovMat[3][3], double Mean[4])
{
  int i;
  double accu[16]={0};

  // For each point in the cloud
  for (i = 0; i < point_count; ++i)
  {
    const float* cloud = &pc[i*ws];
    accu [0] += cloud[0] * cloud[0];
    accu [1] += cloud[0] * cloud[1];
    accu [2] += cloud[0] * cloud[2];
    accu [3] += cloud[1] * cloud[1]; // 4
    accu [4] += cloud[1] * cloud[2]; // 5
    accu [5] += cloud[2] * cloud[2]; // 8
    accu [6] += cloud[0];
    accu [7] += cloud[1];
    accu [8] += cloud[2];
  }

  for (i = 0; i < 9; ++i)
    accu[i]/=(double)point_count;

  Mean[0] = accu[6];
  Mean[1] = accu[7];
  Mean[2] = accu[8];
  Mean[3] = 0;
  CovMat[0][0] = accu [0] - accu [6] * accu [6];
  CovMat[0][1] = accu [1] - accu [6] * accu [7];
  CovMat[0][2] = accu [2] - accu [6] * accu [8];
  CovMat[1][1] = accu [3] - accu [7] * accu [7];
  CovMat[1][2] = accu [4] - accu [7] * accu [8];
  CovMat[2][2] = accu [5] - accu [8] * accu [8];
  CovMat[1][0] = CovMat[0][1];
  CovMat[2][0] = CovMat[0][2];
  CovMat[2][1] = CovMat[1][2];

}

void meanCovLocalPCInd(const float* pc, const int* Indices, const int ws, const int point_count, double CovMat[3][3], double Mean[4])
{
  int i;
  double accu[16]={0};

  for (i = 0; i < point_count; ++i)
  {
    const float* cloud = &pc[ Indices[i] * ws ];
    accu [0] += cloud[0] * cloud[0];
    accu [1] += cloud[0] * cloud[1];
    accu [2] += cloud[0] * cloud[2];
    accu [3] += cloud[1] * cloud[1]; // 4
    accu [4] += cloud[1] * cloud[2]; // 5
    accu [5] += cloud[2] * cloud[2]; // 8
    accu [6] += cloud[0];
    accu [7] += cloud[1];
    accu [8] += cloud[2];
  }

  for (i = 0; i < 9; ++i)
    accu[i]/=(double)point_count;

  Mean[0] = accu[6];
  Mean[1] = accu[7];
  Mean[2] = accu[8];
  Mean[3] = 0;
  CovMat[0][0] = accu [0] - accu [6] * accu [6];
  CovMat[0][1] = accu [1] - accu [6] * accu [7];
  CovMat[0][2] = accu [2] - accu [6] * accu [8];
  CovMat[1][1] = accu [3] - accu [7] * accu [7];
  CovMat[1][2] = accu [4] - accu [7] * accu [8];
  CovMat[2][2] = accu [5] - accu [8] * accu [8];
  CovMat[1][0] = CovMat[0][1];
  CovMat[2][0] = CovMat[0][2];
  CovMat[2][1] = CovMat[1][2];

}

CV_EXPORTS int computeNormalsPC3d(const Mat& PC, Mat& PCNormals, const int NumNeighbors, const bool FlipViewpoint, const double viewpoint[3])
{
  int i;

  if (PC.cols!=3 && PC.cols!=6) // 3d data is expected
  {
    //return -1;
    CV_Error(cv::Error::BadImageSize, "PC should have 3 or 6 elements in its columns");
  }

  int sizes[2] = {PC.rows, 3};
  int sizesResult[2] = {PC.rows, NumNeighbors};
  float* dataset = new float[PC.rows*3];
  float* distances = new float[PC.rows*NumNeighbors];
  int* indices = new int[PC.rows*NumNeighbors];

  for (i=0; i<PC.rows; i++)
  {
    const float* src = (float*)(&PC.data[i*PC.step]);
    float* dst = (float*)(&dataset[i*3]);

    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
  }

  Mat PCInput(2, sizes, CV_32F, dataset, 0);

  void* flannIndex = indexPCFlann(PCInput);

  Mat Indices(2, sizesResult, CV_32S, indices, 0);
  Mat Distances(2, sizesResult, CV_32F, distances, 0);

  queryPCFlann(flannIndex, PCInput, Indices, Distances, NumNeighbors);
  destroyFlann(flannIndex);
  flannIndex = 0;

  PCNormals = Mat(PC.rows, 6, CV_32F);

  for (i=0; i<PC.rows; i++)
  {
    double C[3][3], mu[4];
    const float* pci = &dataset[i*3];
    float* pcr = (float*)(&PCNormals.data[i*PCNormals.step]);
    double nr[3];

    int* indLocal = &indices[i*NumNeighbors];

    // compute covariance matrix
    meanCovLocalPCInd(dataset, indLocal, 3, NumNeighbors, C, mu);

    // eigenvectors of covariance matrix
    Mat cov(3, 3, CV_64F), eigVect, eigVal;
    double* covData = (double*)cov.data;
    covData[0] = C[0][0];
    covData[1] = C[0][1];
    covData[2] = C[0][2];
    covData[3] = C[1][0];
    covData[4] = C[1][1];
    covData[5] = C[1][2];
    covData[6] = C[2][0];
    covData[7] = C[2][1];
    covData[8] = C[2][2];
    eigen(cov, eigVal, eigVect);
    Mat lowestEigVec;
    //the eigenvector for the lowest eigenvalue is in the last row
    eigVect.row(eigVect.rows - 1).copyTo(lowestEigVec);
    double* eigData = (double*)lowestEigVec.data;
    nr[0] = eigData[0];
    nr[1] = eigData[1];
    nr[2] = eigData[2];

    pcr[0] = pci[0];
    pcr[1] = pci[1];
    pcr[2] = pci[2];

    if (FlipViewpoint)
    {
      flipNormalViewpoint(pci, viewpoint[0], viewpoint[1], viewpoint[2], &nr[0], &nr[1], &nr[2]);
    }

    pcr[3] = (float)nr[0];
    pcr[4] = (float)nr[1];
    pcr[5] = (float)nr[2];
  }

  delete[] indices;
  delete[] distances;
  delete[] dataset;

  return 1;
}

} // namespace ppf_match_3d

} // namespace cv
