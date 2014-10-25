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

void Pose3D::updatePose(double NewPose[16])
{
  double R[9];

  for (int i=0; i<16; i++)
    pose[i]=NewPose[i];

  R[0] = pose[0];
  R[1] = pose[1];
  R[2] = pose[2];
  R[3] = pose[4];
  R[4] = pose[5];
  R[5] = pose[6];
  R[6] = pose[8];
  R[7] = pose[9];
  R[8] = pose[10];

  t[0]=pose[3];
  t[1]=pose[7];
  t[2]=pose[11];

  // compute the angle
  const double trace = R[0] + R[4] + R[8];

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

void Pose3D::updatePose(double NewR[9], double NewT[3])
{
  pose[0]=NewR[0];
  pose[1]=NewR[1];
  pose[2]=NewR[2];
  pose[3]=NewT[0];
  pose[4]=NewR[3];
  pose[5]=NewR[4];
  pose[6]=NewR[5];
  pose[7]=NewT[1];
  pose[8]=NewR[6];
  pose[9]=NewR[7];
  pose[10]=NewR[8];
  pose[11]=NewT[2];
  pose[12]=0;
  pose[13]=0;
  pose[14]=0;
  pose[15]=1;

  // compute the angle
  const double trace = NewR[0] + NewR[4] + NewR[8];

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

void Pose3D::updatePoseQuat(double Q[4], double NewT[3])
{
  double NewR[9];

  quatToDCM(Q, NewR);
  q[0]=Q[0];
  q[1]=Q[1];
  q[2]=Q[2];
  q[3]=Q[3];

  pose[0]=NewR[0];
  pose[1]=NewR[1];
  pose[2]=NewR[2];
  pose[3]=NewT[0];
  pose[4]=NewR[3];
  pose[5]=NewR[4];
  pose[6]=NewR[5];
  pose[7]=NewT[1];
  pose[8]=NewR[6];
  pose[9]=NewR[7];
  pose[10]=NewR[8];
  pose[11]=NewT[2];
  pose[12]=0;
  pose[13]=0;
  pose[14]=0;
  pose[15]=1;

  // compute the angle
  const double trace = NewR[0] + NewR[4] + NewR[8];

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


void Pose3D::appendPose(double IncrementalPose[16])
{
  double R[9], PoseFull[16]={0};

  matrixProduct44(IncrementalPose, this->pose, PoseFull);

  R[0] = PoseFull[0];
  R[1] = PoseFull[1];
  R[2] = PoseFull[2];
  R[3] = PoseFull[4];
  R[4] = PoseFull[5];
  R[5] = PoseFull[6];
  R[6] = PoseFull[8];
  R[7] = PoseFull[9];
  R[8] = PoseFull[10];

  t[0]=PoseFull[3];
  t[1]=PoseFull[7];
  t[2]=PoseFull[11];

  // compute the angle
  const double trace = R[0] + R[4] + R[8];

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

  for (int i=0; i<16; i++)
    pose[i]=PoseFull[i];
}

Pose3DPtr Pose3D::clone()
{
  Ptr<Pose3D> new_pose(new Pose3D(alpha, modelIndex, numVotes));
  for (int i=0; i<16; i++)
    new_pose->pose[i]= this->pose[i];

  new_pose->q[0]=q[0];
  new_pose->q[1]=q[1];
  new_pose->q[2]=q[2];
  new_pose->q[3]=q[3];

  new_pose->t[0]=t[0];
  new_pose->t[1]=t[1];
  new_pose->t[2]=t[2];

  new_pose->angle=angle;

  return new_pose;
}

void Pose3D::printPose()
{
  printf("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n", this->modelIndex, this->numVotes, this->residual);
  for (int j=0; j<4; j++)
  {
    for (int k=0; k<4; k++)
    {
      printf("%f ", this->pose[j*4+k]);
    }
    printf("\n");
  }
  printf("\n");
}

int Pose3D::writePose(FILE* f)
{
  int POSE_MAGIC = 7673;
  fwrite(&POSE_MAGIC, sizeof(int), 1, f);
  fwrite(&angle, sizeof(double), 1, f);
  fwrite(&numVotes, sizeof(int), 1, f);
  fwrite(&modelIndex, sizeof(int), 1, f);
  fwrite(pose, sizeof(double)*16, 1, f);
  fwrite(t, sizeof(double)*3, 1, f);
  fwrite(q, sizeof(double)*4, 1, f);
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
    status = fread(pose, sizeof(double)*16, 1, f);
    status = fread(t, sizeof(double)*3, 1, f);
    status = fread(q, sizeof(double)*4, 1, f);
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
