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
#include "hash_murmur.hpp"

namespace cv 
{
namespace ppf_match_3d
{

static const size_t PPF_LENGTH = 5;

// routines for assisting sort
static bool pose3DPtrCompare(const Pose3DPtr& a, const Pose3DPtr& b)
{
  CV_Assert(!a.empty() && !b.empty());
  return ( a->numVotes > b->numVotes );
}

static int sortPoseClusters(const PoseCluster3DPtr& a, const PoseCluster3DPtr& b)
{
  CV_Assert(!a.empty() && !b.empty());
  return ( a->numVotes > b->numVotes );
}

// simple hashing
/*static int hashPPFSimple(const double f[4], const double AngleStep, const double DistanceStep)
{
  const unsigned char d1 = (unsigned char) (floor ((double)f[0] / (double)AngleStep));
  const unsigned char d2 = (unsigned char) (floor ((double)f[1] / (double)AngleStep));
  const unsigned char d3 = (unsigned char) (floor ((double)f[2] / (double)AngleStep));
  const unsigned char d4 = (unsigned char) (floor ((double)f[3] / (double)DistanceStep));

  int hashKey = (d1 | (d2<<8) | (d3<<16) | (d4<<24));
  return hashKey;
}*/

// quantize ppf and hash it for proper indexing
static KeyType hashPPF(const double f[4], const double AngleStep, const double DistanceStep)
{
  const int d1 = (int) (floor ((double)f[0] / (double)AngleStep));
  const int d2 = (int) (floor ((double)f[1] / (double)AngleStep));
  const int d3 = (int) (floor ((double)f[2] / (double)AngleStep));
  const int d4 = (int) (floor ((double)f[3] / (double)DistanceStep));
  int key[4]={d1,d2,d3,d4};
  KeyType hashKey=0;

  murmurHash(key, 4*sizeof(int), 42, &hashKey);

  return hashKey;
}

/*static size_t hashMurmur(unsigned int key)
{
  size_t hashKey=0;
  hashMurmurx86((void*)&key, 4, 42, &hashKey);
  return hashKey;
}*/

static double computeAlpha(const double p1[4], const double n1[4], const double p2[4])
{
  double Tmg[3], mpt[3], row2[3], row3[3], alpha;

  computeTransformRTyz(p1, n1, row2, row3, Tmg);

  // checked row2, row3: They are correct

  mpt[1] = Tmg[1] + row2[0] * p2[0] + row2[1] * p2[1] + row2[2] * p2[2];
  mpt[2] = Tmg[2] + row3[0] * p2[0] + row3[1] * p2[1] + row3[2] * p2[2];

  alpha=atan2(-mpt[2], mpt[1]);

  if ( alpha != alpha)
  {
    return 0;
  }

  if (sin(alpha)*mpt[2]<0.0)
    alpha=-alpha;

  return (-alpha);
}

PPF3DDetector::PPF3DDetector()
{
  sampling_step_relative = 0.05;
  distance_step_relative = 0.05;
  scene_sample_step = (int)(1/0.04);
  angle_step_relative = 30;
  angle_step_radians = (360.0/angle_step_relative)*M_PI/180.0;
  angle_step = angle_step_radians;
  trained = false;

  setSearchParams();
}

PPF3DDetector::PPF3DDetector(const double RelativeSamplingStep, const double RelativeDistanceStep, const double NumAngles)
{
  sampling_step_relative = RelativeSamplingStep;
  distance_step_relative = RelativeDistanceStep;
  angle_step_relative = NumAngles;
  angle_step_radians = (360.0/angle_step_relative)*M_PI/180.0;
  //SceneSampleStep = 1.0/RelativeSceneSampleStep;
  angle_step = angle_step_radians;
  trained = false;

  setSearchParams();
}

void PPF3DDetector::setSearchParams(const double positionThreshold, const double rotationThreshold, const bool useWeightedClustering)
{
  if (positionThreshold<0)
    position_threshold = sampling_step_relative;
  else
    position_threshold = positionThreshold;

  if (rotationThreshold<0)
    rotation_threshold = ((360/angle_step) / 180.0 * M_PI);
  else
    rotation_threshold = rotationThreshold;

  use_weighted_avg = useWeightedClustering;
}

// compute per point PPF as in paper
void PPF3DDetector::computePPFFeatures(const double p1[4], const double n1[4],
                                       const double p2[4], const double n2[4],
                                       double f[4])
{
  /*
  Vectors will be defined as of length 4 instead of 3, because of:
  - Further SIMD vectorization
  - Cache alignment
  */

  double d[4] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2], 0};

  double norm = TNorm3(d);
  f[3] = norm;

  if (norm)
  {
    d[0] /= f[3];
    d[1] /= f[3];
    d[2] /= f[3];
  }
  else
  {
    // TODO: Handle this
    f[0] = 0;
    f[1] = 0;
    f[2] = 0;
    return ;
  }

  /*
  Tolga Birdal's note:
  Issues of numerical stability is of concern here.
  Bertram's suggestion: atan2(a dot b, |axb|)
  My correction :
  I guess it should be: angle = atan2(norm(cross(a,b)), dot(a,b))
  The macro is implemented accordingly.
  TAngle3 actually outputs in range [0, pi] as
  Bertram suggests
  */

  f[0] = TAngle3(n1, d);
  f[1] = TAngle3(n2, d);
  f[2] = TAngle3(n1, n2);
}

void PPF3DDetector::clearTrainingModels()
{
  if (this->hash_nodes)
  {
    free(this->hash_nodes);
    this->hash_nodes=0;
  }

  if (this->hash_table)
  {
    hashtableDestroy(this->hash_table);
    this->hash_table=0;
  }
}

PPF3DDetector::~PPF3DDetector()
{
  clearTrainingModels();
}

// TODO: Check all step sizes to be positive
void PPF3DDetector::trainModel(const Mat &PC)
{
  CV_Assert(PC.type() == CV_32F || PC.type() == CV_32FC1);

  // compute bbox
  float xRange[2], yRange[2], zRange[2];
  computeBboxStd(PC, xRange, yRange, zRange);

  // compute sampling step from diameter of bbox
  float dx = xRange[1] - xRange[0];
  float dy = yRange[1] - yRange[0];
  float dz = zRange[1] - zRange[0];
  float diameter = sqrt ( dx * dx + dy * dy + dz * dz );

  float distanceStep = (float)(diameter * sampling_step_relative);

  Mat sampled = samplePCByQuantization(PC, xRange, yRange, zRange, (float)sampling_step_relative,0);

  int size = sampled.rows*sampled.rows;

  hashtable_int* hashTable = hashtableCreate(size, NULL);

  int numPPF = sampled.rows*sampled.rows;
  ppf = Mat(numPPF, PPF_LENGTH, CV_32FC1);
  int ppfStep = (int)ppf.step;
  int sampledStep = (int)sampled.step;

  // TODO: Maybe I could sample 1/5th of them here. Check the performance later.
  int numRefPoints = sampled.rows;

  // pre-allocate the hash nodes
  hash_nodes = (THash*)calloc(numRefPoints*numRefPoints, sizeof(THash));

  // TODO : This can easily be parallelized. But we have to lock hashtable_insert.
  // I realized that performance drops when this loop is parallelized (unordered
  // inserts into the hashtable
  // But it is still there to be investigated. For now, I leave this unparallelized
  // since this is just a training part.
  for (int i=0; i<numRefPoints; i++)
  {
    float* f1 = (float*)(&sampled.data[i * sampledStep]);
    const double p1[4] = {f1[0], f1[1], f1[2], 0};
    const double n1[4] = {f1[3], f1[4], f1[5], 0};

    //printf("///////////////////// NEW REFERENCE ////////////////////////\n");
    for (int j=0; j<numRefPoints; j++)
    {
      // cannnot compute the ppf with myself
      if (i!=j)
      {
        float* f2 = (float*)(&sampled.data[j * sampledStep]);
        const double p2[4] = {f2[0], f2[1], f2[2], 0};
        const double n2[4] = {f2[3], f2[4], f2[5], 0};

        double f[4]={0};
        computePPFFeatures(p1, n1, p2, n2, f);
        KeyType hashValue = hashPPF(f, angle_step_radians, distanceStep);
        double alpha = computeAlpha(p1, n1, p2);
        unsigned int corrInd = i*numRefPoints+j;
        unsigned int ppfInd = corrInd*ppfStep;

        THash* hashNode = &hash_nodes[i*numRefPoints+j];
        hashNode->id = hashValue;
        hashNode->i = i;
        hashNode->ppfInd = ppfInd;

        hashtableInsertHashed(hashTable, hashValue, (void*)hashNode);

        float* ppfRow = (float*)(&(ppf.data[ ppfInd ]));
        ppfRow[0] = (float)f[0];
        ppfRow[1] = (float)f[1];
        ppfRow[2] = (float)f[2];
        ppfRow[3] = (float)f[3];
        ppfRow[4] = (float)alpha;
      }
    }
  }

  angle_step = angle_step_radians;
  distance_step = distanceStep;
  hash_table = hashTable;
  ppf_step = ppfStep;
  num_ref_points = numRefPoints;
  sampled_pc = sampled;
  trained = true;
}



///////////////////////// MATCHING ////////////////////////////////////////


bool PPF3DDetector::matchPose(const Pose3D& sourcePose, const Pose3D& targetPose)
{
  // translational difference
  double dv[3] = {targetPose.t[0]-sourcePose.t[0], targetPose.t[1]-sourcePose.t[1], targetPose.t[2]-sourcePose.t[2]};
  double dNorm = sqrt(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]);

  const double phi = fabs ( sourcePose.angle - targetPose.angle );

  return (phi<this->rotation_threshold && dNorm < this->position_threshold);
}

void PPF3DDetector::clusterPoses(std::vector<Pose3DPtr> poseList, int numPoses, std::vector<Pose3DPtr> &finalPoses)
{
  std::vector<PoseCluster3DPtr> poseClusters;

  finalPoses.clear();

  // sort the poses for stability
  std::sort(poseList.begin(), poseList.end(), pose3DPtrCompare);

  for (int i=0; i<numPoses; i++)
  {
    Pose3DPtr pose = poseList[i];
    bool assigned = false;

    // search all clusters
    for (size_t j=0; j<poseClusters.size() && !assigned; j++)
    {
      const Pose3DPtr poseCenter = poseClusters[j]->poseList[0];
      if (matchPose(*pose, *poseCenter))
      {
        poseClusters[j]->addPose(pose);
        assigned = true;
      }
    }

    if (!assigned)
    {
      poseClusters.push_back(PoseCluster3DPtr(new PoseCluster3D(pose)));
    }
  }

  // sort the clusters so that we could output multiple hypothesis
  std::sort(poseClusters.begin(), poseClusters.end(), sortPoseClusters);

  finalPoses.resize(poseClusters.size());

  // TODO: Use MinMatchScore

  if (use_weighted_avg)
  {
#if defined _OPENMP
#pragma omp parallel for
#endif
    // uses weighting by the number of votes
    for (int i=0; i<static_cast<int>(poseClusters.size()); i++)
    {
      // We could only average the quaternions. So I will make use of them here
      double qAvg[4]={0}, tAvg[3]={0};

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      int curSize = (int)curPoses.size();
      int numTotalVotes = 0;

      for (int j=0; j<curSize; j++)
        numTotalVotes += curPoses[j]->numVotes;

      double wSum=0;

      for (int j=0; j<curSize; j++)
      {
        const double w = (double)curPoses[j]->numVotes / (double)numTotalVotes;

        qAvg[0]+= w*curPoses[j]->q[0];
        qAvg[1]+= w*curPoses[j]->q[1];
        qAvg[2]+= w*curPoses[j]->q[2];
        qAvg[3]+= w*curPoses[j]->q[3];

        tAvg[0]+= w*curPoses[j]->t[0];
        tAvg[1]+= w*curPoses[j]->t[1];
        tAvg[2]+= w*curPoses[j]->t[2];
        wSum+=w;
      }

      tAvg[0]/=wSum;
      tAvg[1]/=wSum;
      tAvg[2]/=wSum;

      qAvg[0]/=wSum;
      qAvg[1]/=wSum;
      qAvg[2]/=wSum;
      qAvg[3]/=wSum;

      curPoses[0]->updatePoseQuat(qAvg, tAvg);
      curPoses[0]->numVotes=curCluster->numVotes;

      finalPoses[i]=curPoses[0]->clone();
    }
  }
  else
  {
#if defined _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<static_cast<int>(poseClusters.size()); i++)
    {
      // We could only average the quaternions. So I will make use of them here
      double qAvg[4]={0}, tAvg[3]={0};

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      const int curSize = (int)curPoses.size();

      for (int j=0; j<curSize; j++)
      {
        qAvg[0]+= curPoses[j]->q[0];
        qAvg[1]+= curPoses[j]->q[1];
        qAvg[2]+= curPoses[j]->q[2];
        qAvg[3]+= curPoses[j]->q[3];

        tAvg[0]+= curPoses[j]->t[0];
        tAvg[1]+= curPoses[j]->t[1];
        tAvg[2]+= curPoses[j]->t[2];
      }

      tAvg[0]/=(double)curSize;
      tAvg[1]/=(double)curSize;
      tAvg[2]/=(double)curSize;

      qAvg[0]/=(double)curSize;
      qAvg[1]/=(double)curSize;
      qAvg[2]/=(double)curSize;
      qAvg[3]/=(double)curSize;

      curPoses[0]->updatePoseQuat(qAvg, tAvg);
      curPoses[0]->numVotes=curCluster->numVotes;

      finalPoses[i]=curPoses[0]->clone();
    }
  }

  poseClusters.clear();
}

void PPF3DDetector::match(const Mat& pc, std::vector<Pose3DPtr>& results, const double relativeSceneSampleStep, const double relativeSceneDistance)
{
  if (!trained)
  {
    throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot match without training", __FUNCTION__, __FILE__, __LINE__);
  }

  CV_Assert(pc.type() == CV_32F || pc.type() == CV_32FC1);
  CV_Assert(relativeSceneSampleStep<=1 && relativeSceneSampleStep>0);

  scene_sample_step = (int)(1.0/relativeSceneSampleStep);

  //int numNeighbors = 10;
  int numAngles = (int) (floor (2 * M_PI / angle_step));
  float distanceStep = (float)distance_step;
  unsigned int n = num_ref_points;
  std::vector<Pose3DPtr> poseList;
  int sceneSamplingStep = scene_sample_step;

  // compute bbox
  float xRange[2], yRange[2], zRange[2];
  computeBboxStd(pc, xRange, yRange, zRange);

  // sample the point cloud
  /*float dx = xRange[1] - xRange[0];
  float dy = yRange[1] - yRange[0];
  float dz = zRange[1] - zRange[0];
  float diameter = sqrt ( dx * dx + dy * dy + dz * dz );
  float distanceSampleStep = diameter * RelativeSceneDistance;*/
  Mat sampled = samplePCByQuantization(pc, xRange, yRange, zRange, (float)relativeSceneDistance, 0);

  // allocate the accumulator : Moved this to the inside of the loop
  /*#if !defined (_OPENMP)
     unsigned int* accumulator = (unsigned int*)calloc(numAngles*n, sizeof(unsigned int));
  #endif*/

  poseList.reserve((sampled.rows/sceneSamplingStep)+4);

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < sampled.rows; i += sceneSamplingStep)
  {
    unsigned int refIndMax = 0, alphaIndMax = 0;
    unsigned int maxVotes = 0;

    float* f1 = (float*)(&sampled.data[i * sampled.step]);
    const double p1[4] = {f1[0], f1[1], f1[2], 0};
    const double n1[4] = {f1[3], f1[4], f1[5], 0};
    double *row2, *row3, tsg[3]={0}, Rsg[9]={0}, RInv[9]={0};

    unsigned int* accumulator = (unsigned int*)calloc(numAngles*n, sizeof(unsigned int));
    computeTransformRT(p1, n1, Rsg, tsg);
    row2=&Rsg[3];
    row3=&Rsg[6];

    // Tolga Birdal's notice:
    // As a later update, we might want to look into a local neighborhood only
    // To do this, simply search the local neighborhood by radius look up
    // and collect the neighbors to compute the relative pose

    for (int j = 0; j < sampled.rows; j ++)
    {
      if (i!=j)
      {
        float* f2 = (float*)(&sampled.data[j * sampled.step]);
        const double p2[4] = {f2[0], f2[1], f2[2], 0};
        const double n2[4] = {f2[3], f2[4], f2[5], 0};
        double p2t[4], alpha_scene;

        double f[4]={0};
        computePPFFeatures(p1, n1, p2, n2, f);
        KeyType hashValue = hashPPF(f, angle_step, distanceStep);

        // we don't need to call this here, as we already estimate the tsg from scene reference point
        // double alpha = computeAlpha(p1, n1, p2);
        p2t[1] = tsg[1] + row2[0] * p2[0] + row2[1] * p2[1] + row2[2] * p2[2];
        p2t[2] = tsg[2] + row3[0] * p2[0] + row3[1] * p2[1] + row3[2] * p2[2];

        alpha_scene=atan2(-p2t[2], p2t[1]);

        if ( alpha_scene != alpha_scene)
        {
          continue;
        }

        if (sin(alpha_scene)*p2t[2]<0.0)
          alpha_scene=-alpha_scene;

        alpha_scene=-alpha_scene;

        hashnode_i* node = hashtableGetBucketHashed(hash_table, (hashValue));

        while (node)
        {
          THash* tData = (THash*) node->data;
          int corrI = (int)tData->i;
          int ppfInd = (int)tData->ppfInd;
          float* ppfCorrScene = (float*)(&ppf.data[ppfInd]);
          double alpha_model = (double)ppfCorrScene[PPF_LENGTH-1];
          double alpha = alpha_model - alpha_scene;

          /*  Tolga Birdal's note: Map alpha to the indices:
                  atan2 generates results in (-pi pi]
                  That's why alpha should be in range [-2pi 2pi]
                  So the quantization would be :
                  numAngles * (alpha+2pi)/(4pi)
                  */

          //printf("%f\n", alpha);
          int alpha_index = (int)(numAngles*(alpha + 2*M_PI) / (4*M_PI));

          unsigned int accIndex = corrI * numAngles + alpha_index;

          accumulator[accIndex]++;
          node = node->next;
        }
      }
    }

    // Maximize the accumulator
    for (unsigned int k = 0; k < n; k++)
    {
      for (int j = 0; j < numAngles; j++)
      {
        const unsigned int accInd = k*numAngles + j;
        const unsigned int accVal = accumulator[ accInd ];
        if (accVal > maxVotes)
        {
          maxVotes = accVal;
          refIndMax = k;
          alphaIndMax = j;
        }

#if !defined (_OPENMP)
        accumulator[accInd ] = 0;
#endif
      }
    }

    // invert Tsg : Luckily rotation is orthogonal: Inverse = Transpose.
    // We are not required to invert.
    double tInv[3], tmg[3], Rmg[9];
    matrixTranspose33(Rsg, RInv);
    matrixProduct331(RInv, tsg, tInv);

    double TsgInv[16] = { RInv[0], RInv[1], RInv[2], -tInv[0],
                          RInv[3], RInv[4], RInv[5], -tInv[1],
                          RInv[6], RInv[7], RInv[8], -tInv[2],
                          0, 0, 0, 1
                        };

    // TODO : Compute pose
    const float* fMax = (float*)(&sampled_pc.data[refIndMax * sampled_pc.step]);
    const double pMax[4] = {fMax[0], fMax[1], fMax[2], 1};
    const double nMax[4] = {fMax[3], fMax[4], fMax[5], 1};

    computeTransformRT(pMax, nMax, Rmg, tmg);
    row2=&Rsg[3];
    row3=&Rsg[6];

    double Tmg[16] = { Rmg[0], Rmg[1], Rmg[2], tmg[0],
                       Rmg[3], Rmg[4], Rmg[5], tmg[1],
                       Rmg[6], Rmg[7], Rmg[8], tmg[2],
                       0, 0, 0, 1
                     };

    // convert alpha_index to alpha
    int alpha_index = alphaIndMax;
    double alpha = (alpha_index*(4*M_PI))/numAngles-2*M_PI;

    // Equation 2:
    double Talpha[16]={0};
    getUnitXRotation_44(alpha, Talpha);

    double Temp[16]={0};
    double rawPose[16]={0};
    matrixProduct44(Talpha, Tmg, Temp);
    matrixProduct44(TsgInv, Temp, rawPose);

    Pose3DPtr pose(new Pose3D(alpha, refIndMax, maxVotes));
    pose->updatePose(rawPose);
    poseList.push_back(pose);

#if defined (_OPENMP)
    free(accumulator);
#endif
  }

  // TODO : Make the parameters relative if not arguments.
  //double MinMatchScore = 0.5;

  int numPosesAdded = sampled.rows/sceneSamplingStep;

  clusterPoses(poseList, numPosesAdded, results);
}

} // namespace ppf_match_3d

} // namespace cv
