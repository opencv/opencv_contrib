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
/*static int hashPPFSimple(const Vec4d& f, const double AngleStep, const double DistanceStep)
{
  Vec4i key(
      (int)(f[0] / AngleStep),
      (int)(f[1] / AngleStep),
      (int)(f[2] / AngleStep),
      (int)(f[3] / DistanceStep));

  int hashKey = d.val[0] | (d.val[1] << 8) | (d.val[2] << 16) | (d.val[3] << 24);
  return hashKey;
}*/

// quantize ppf and hash it for proper indexing
static KeyType hashPPF(const Vec4d& f, const double AngleStep, const double DistanceStep)
{
  Vec4i key(
      (int)(f[0] / AngleStep),
      (int)(f[1] / AngleStep),
      (int)(f[2] / AngleStep),
      (int)(f[3] / DistanceStep));
  KeyType hashKey = 0;

  murmurHash(key.val, 4*sizeof(int), 42, &hashKey);
  return hashKey;
}

/*static size_t hashMurmur(uint key)
{
  size_t hashKey=0;
  hashMurmurx86((void*)&key, 4, 42, &hashKey);
  return hashKey;
}*/

static double computeAlpha(const Vec3d& p1, const Vec3d& n1, const Vec3d& p2)
{
  Vec3d Tmg, mpt;
  Matx33d R;
  double alpha;

  computeTransformRT(p1, n1, R, Tmg);
  mpt = Tmg + R * p2;
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

  hash_table = NULL;
  hash_nodes = NULL;

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

  hash_table = NULL;
  hash_nodes = NULL;

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
void PPF3DDetector::computePPFFeatures(const Vec3d& p1, const Vec3d& n1,
                                       const Vec3d& p2, const Vec3d& n2,
                                       Vec4d& f)
{
  Vec3d d(p2 - p1);
  f[3] = cv::norm(d);
  if (f[3] <= EPS)
    return;
  d *= 1.0 / f[3];

  f[0] = TAngle3Normalized(n1, d);
  f[1] = TAngle3Normalized(n2, d);
  f[2] = TAngle3Normalized(n1, n2);
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
  Vec2f xRange, yRange, zRange;
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
    const Vec3f p1(sampled.ptr<float>(i));
    const Vec3f n1(sampled.ptr<float>(i) + 3);

    //printf("///////////////////// NEW REFERENCE ////////////////////////\n");
    for (int j=0; j<numRefPoints; j++)
    {
      // cannnot compute the ppf with myself
      if (i!=j)
      {
        const Vec3f p2(sampled.ptr<float>(j));
        const Vec3f n2(sampled.ptr<float>(j) + 3);

        Vec4d f = Vec4d::all(0);
        computePPFFeatures(p1, n1, p2, n2, f);
        KeyType hashValue = hashPPF(f, angle_step_radians, distanceStep);
        double alpha = computeAlpha(p1, n1, p2);
        uint ppfInd = i*numRefPoints+j;

        THash* hashNode = &hash_nodes[i*numRefPoints+j];
        hashNode->id = hashValue;
        hashNode->i = i;
        hashNode->ppfInd = ppfInd;

        hashtableInsertHashed(hashTable, hashValue, (void*)hashNode);

        Mat(f).reshape(1, 1).convertTo(ppf.row(ppfInd).colRange(0, 4), CV_32F);
        ppf.ptr<float>(ppfInd)[4] = (float)alpha;
      }
    }
  }

  angle_step = angle_step_radians;
  distance_step = distanceStep;
  hash_table = hashTable;
  num_ref_points = numRefPoints;
  sampled_pc = sampled;
  trained = true;
}



///////////////////////// MATCHING ////////////////////////////////////////


bool PPF3DDetector::matchPose(const Pose3D& sourcePose, const Pose3D& targetPose)
{
  // translational difference
  Vec3d dv = targetPose.t - sourcePose.t;
  double dNorm = cv::norm(dv);

  const double phi = fabs ( sourcePose.angle - targetPose.angle );

  return (phi<this->rotation_threshold && dNorm < this->position_threshold);
}

void PPF3DDetector::clusterPoses(std::vector<Pose3DPtr>& poseList, int numPoses, std::vector<Pose3DPtr> &finalPoses)
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
      Vec4d qAvg = Vec4d::all(0);
      Vec3d tAvg = Vec3d::all(0);

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      int curSize = (int)curPoses.size();
      size_t numTotalVotes = 0;

      for (int j=0; j<curSize; j++)
        numTotalVotes += curPoses[j]->numVotes;

      double wSum=0;

      for (int j=0; j<curSize; j++)
      {
        const double w = (double)curPoses[j]->numVotes / (double)numTotalVotes;

        qAvg += w * curPoses[j]->q;
        tAvg += w * curPoses[j]->t;
        wSum += w;
      }

      tAvg *= 1.0 / wSum;
      qAvg *= 1.0 / wSum;

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
      Vec4d qAvg = Vec4d::all(0);
      Vec3d tAvg = Vec3d::all(0);

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      const int curSize = (int)curPoses.size();

      for (int j=0; j<curSize; j++)
      {
        qAvg += curPoses[j]->q;
        tAvg += curPoses[j]->t;
      }

      tAvg *= 1.0 / curSize;
      qAvg *= 1.0 / curSize;

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
  uint n = num_ref_points;
  std::vector<Pose3DPtr> poseList;
  int sceneSamplingStep = scene_sample_step;

  // compute bbox
  Vec2f xRange, yRange, zRange;
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
     uint* accumulator = (uint*)calloc(numAngles*n, sizeof(uint));
  #endif*/

  poseList.reserve((sampled.rows/sceneSamplingStep)+4);

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < sampled.rows; i += sceneSamplingStep)
  {
    uint refIndMax = 0, alphaIndMax = 0;
    uint maxVotes = 0;

    const Vec3f p1(sampled.ptr<float>(i));
    const Vec3f n1(sampled.ptr<float>(i) + 3);
    Vec3d tsg = Vec3d::all(0);
    Matx33d Rsg = Matx33d::all(0), RInv = Matx33d::all(0);

    uint* accumulator = (uint*)calloc(numAngles*n, sizeof(uint));
    computeTransformRT(p1, n1, Rsg, tsg);

    // Tolga Birdal's notice:
    // As a later update, we might want to look into a local neighborhood only
    // To do this, simply search the local neighborhood by radius look up
    // and collect the neighbors to compute the relative pose

    for (int j = 0; j < sampled.rows; j ++)
    {
      if (i!=j)
      {
        const Vec3f p2(sampled.ptr<float>(j));
        const Vec3f n2(sampled.ptr<float>(j) + 3);
        Vec3d p2t;
        double alpha_scene;

        Vec4d f = Vec4d::all(0);
        computePPFFeatures(p1, n1, p2, n2, f);
        KeyType hashValue = hashPPF(f, angle_step, distanceStep);

        p2t = tsg + Rsg * Vec3d(p2);

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
          float* ppfCorrScene = ppf.ptr<float>(ppfInd);
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

          uint accIndex = corrI * numAngles + alpha_index;

          accumulator[accIndex]++;
          node = node->next;
        }
      }
    }

    // Maximize the accumulator
    for (uint k = 0; k < n; k++)
    {
      for (int j = 0; j < numAngles; j++)
      {
        const uint accInd = k*numAngles + j;
        const uint accVal = accumulator[ accInd ];
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
    Vec3d tInv, tmg;
    Matx33d Rmg;
    RInv = Rsg.t();
    tInv = -RInv * tsg;

    Matx44d TsgInv;
    rtToPose(RInv, tInv, TsgInv);

    // TODO : Compute pose
    const Vec3f pMax(sampled_pc.ptr<float>(refIndMax));
    const Vec3f nMax(sampled_pc.ptr<float>(refIndMax) + 3);

    computeTransformRT(pMax, nMax, Rmg, tmg);

    Matx44d Tmg;
    rtToPose(Rmg, tmg, Tmg);

    // convert alpha_index to alpha
    int alpha_index = alphaIndMax;
    double alpha = (alpha_index*(4*M_PI))/numAngles-2*M_PI;

    // Equation 2:
    Matx44d Talpha;
    Matx33d R;
    Vec3d t = Vec3d::all(0);
    getUnitXRotation(alpha, R);
    rtToPose(R, t, Talpha);

    Matx44d rawPose = TsgInv * (Talpha * Tmg);

    Pose3DPtr pose(new Pose3D(alpha, refIndMax, maxVotes));
    pose->updatePose(rawPose);
    #if defined (_OPENMP)
    #pragma omp critical
    #endif
    {
      poseList.push_back(pose);
    }

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
