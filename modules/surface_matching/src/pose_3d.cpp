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

void Pose3D::updatePose(Matx44d& NewPose)
{
  Matx33d R;

  pose = NewPose;
  poseToRT(pose, R, t);

  // compute the angle
  const double trace = cv::trace(R);

  if (fabs(trace - 3) <= EPS)
  {
    angle = 0;
  }
  else
    if (fabs(trace + 1) <= EPS)
    {
      angle = M_PI;
    }
    else
    {
      angle = ( acos((trace - 1)/2) );
    }

  // compute the quaternion
  dcmToQuat(R, q);
}

void Pose3D::updatePose(Matx33d& NewR, Vec3d& NewT)
{
  rtToPose(NewR, NewT, pose);

  // compute the angle
  const double trace = cv::trace(NewR);

  if (fabs(trace - 3) <= EPS)
  {
    angle = 0;
  }
  else
    if (fabs(trace + 1) <= EPS)
    {
      angle = M_PI;
    }
    else
    {
      angle = ( acos((trace - 1)/2) );
    }

  // compute the quaternion
  dcmToQuat(NewR, q);
}

void Pose3D::updatePoseQuat(Vec4d& Q, Vec3d& NewT)
{
  Matx33d NewR;

  quatToDCM(Q, NewR);
  q = Q;

  rtToPose(NewR, NewT, pose);

  // compute the angle
  const double trace = cv::trace(NewR);

  if (fabs(trace - 3) <= EPS)
  {
    angle = 0;
  }
  else
  {
    if (fabs(trace + 1) <= EPS)
    {
      angle = M_PI;
    }
    else
    {
      angle = ( acos((trace - 1)/2) );
    }
  }
}


void Pose3D::appendPose(Matx44d& IncrementalPose)
{
  Matx33d R;
  Matx44d PoseFull = IncrementalPose * this->pose;

  poseToRT(PoseFull, R, t);

  // compute the angle
  const double trace = cv::trace(R);

  if (fabs(trace - 3) <= EPS)
  {
    angle = 0;
  }
  else
    if (fabs(trace + 1) <= EPS)
    {
      angle = M_PI;
    }
    else
    {
      angle = ( acos((trace - 1)/2) );
    }

  // compute the quaternion
  dcmToQuat(R, q);

  pose = PoseFull;
}

Pose3DPtr Pose3D::clone()
{
  Ptr<Pose3D> new_pose(new Pose3D(alpha, modelIndex, numVotes));

  new_pose->pose = this->pose;
  new_pose->q = q;
  new_pose->t = t;
  new_pose->angle = angle;

  return new_pose;
}

void Pose3D::printPose()
{
  printf("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n", (uint)this->modelIndex, (uint)this->numVotes, this->residual);
  std::cout << this->pose << std::endl;
}

int Pose3D::writePose(FILE* f)
{
  int POSE_MAGIC = 7673;
  fwrite(&POSE_MAGIC, sizeof(int), 1, f);
  fwrite(&angle, sizeof(double), 1, f);
  fwrite(&numVotes, sizeof(int), 1, f);
  fwrite(&modelIndex, sizeof(int), 1, f);
  fwrite(pose.val, sizeof(double)*16, 1, f);
  fwrite(t.val, sizeof(double)*3, 1, f);
  fwrite(q.val, sizeof(double)*4, 1, f);
  fwrite(&residual, sizeof(double), 1, f);
  return 0;
}

int Pose3D::readPose(FILE* f)
{
  int POSE_MAGIC = 7673, magic;

  size_t status = fread(&magic, sizeof(int), 1, f);
  if (status && magic == POSE_MAGIC)
  {
    status = fread(&angle, sizeof(double), 1, f);
    status = fread(&numVotes, sizeof(int), 1, f);
    status = fread(&modelIndex, sizeof(int), 1, f);
    status = fread(pose.val, sizeof(double)*16, 1, f);
    status = fread(t.val, sizeof(double)*3, 1, f);
    status = fread(q.val, sizeof(double)*4, 1, f);
    status = fread(&residual, sizeof(double), 1, f);
    return 0;
  }

  return -1;
}

int Pose3D::writePose(const std::string& FileName)
{
  FILE* f = fopen(FileName.c_str(), "wb");

  if (!f)
    return -1;

  int status = writePose(f);

  fclose(f);
  return status;
}

int Pose3D::readPose(const std::string& FileName)
{
  FILE* f = fopen(FileName.c_str(), "rb");

  if (!f)
    return -1;

  int status = readPose(f);

  fclose(f);
  return status;
}


void PoseCluster3D::addPose(Pose3DPtr newPose)
{
  poseList.push_back(newPose);
  this->numVotes += newPose->numVotes;
};

int PoseCluster3D::writePoseCluster(FILE* f)
{
  int POSE_CLUSTER_MAGIC_IO = 8462597;
  fwrite(&POSE_CLUSTER_MAGIC_IO, sizeof(int), 1, f);
  fwrite(&id, sizeof(int), 1, f);
  fwrite(&numVotes, sizeof(int), 1, f);

  int numPoses = (int)poseList.size();
  fwrite(&numPoses, sizeof(int), 1, f);

  for (int i=0; i<numPoses; i++)
    poseList[i]->writePose(f);

  return 0;
}

int PoseCluster3D::readPoseCluster(FILE* f)
{
  // The magic values are only used to check the files
  int POSE_CLUSTER_MAGIC_IO = 8462597;
  int magic=0, numPoses=0;
  size_t status;
  status = fread(&magic, sizeof(int), 1, f);

  if (!status || magic!=POSE_CLUSTER_MAGIC_IO)
    return -1;

  status = fread(&id, sizeof(int), 1, f);
  status = fread(&numVotes, sizeof(int), 1, f);
  status = fread(&numPoses, sizeof(int), 1, f);
  fclose(f);

  poseList.clear();
  poseList.resize(numPoses);
  for (size_t i=0; i<poseList.size(); i++)
  {
    poseList[i] = Pose3DPtr(new Pose3D());
    poseList[i]->readPose(f);
  }

  return 0;
}

int PoseCluster3D::writePoseCluster(const std::string& FileName)
{
  FILE* f = fopen(FileName.c_str(), "wb");

  if (!f)
    return -1;

  int status = writePoseCluster(f);

  fclose(f);
  return status;
}

int PoseCluster3D::readPoseCluster(const std::string& FileName)
{
  FILE* f = fopen(FileName.c_str(), "rb");

  if (!f)
    return -1;

  int status = readPoseCluster(f);

  fclose(f);
  return status;
}

} // namespace ppf_match_3d

} // namespace cv
