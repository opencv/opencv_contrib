/*
Copyright (c) 2008 University of North Carolina at Chapel Hill

This file is part of SSBA (Simple Sparse Bundle Adjustment).

SSBA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

SSBA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with SSBA. If not, see <http://www.gnu.org/licenses/>.
*/

// Bundle adjustment application for datasets captured with the same camera (common intrinsics).

#include "Math/v3d_linear.h"
#include "Math/v3d_linear_utils.h"
#include "Geometry/v3d_metricbundle.h"

#include <cstring>
#include <iostream>
#include <fstream>
#include <map>

using namespace V3D;
using namespace std;

namespace
{

   inline void
   showErrorStatistics(double const f0,
                       StdDistortionFunction const& distortion,
                       vector<CameraMatrix> const& cams,
                       vector<Vector3d> const& Xs,
                       vector<Vector2d> const& measurements,
                       vector<int> const& correspondingView,
                       vector<int> const& correspondingPoint)
   {
      int const K = measurements.size();

      double meanReprojectionError = 0.0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];
         Vector2d p = cams[i].projectPoint(distortion, Xs[j]);

         double reprojectionError = norm_L2(f0 * (p - measurements[k]));
         meanReprojectionError += reprojectionError;
      }
      cout << "mean reprojection error (in pixels): " << meanReprojectionError/K << endl;
   }

} // end namespace <>

int
main(int argc, char * argv[])
{
   if (argc != 3)
   {
      cerr << "Usage: " << argv[0] << " <sparse reconstruction file> <mode>" << endl;
      cout << "<mode> is one of metric, focal, prinipal, radial, tangental." << endl;
      return -1;
   }

   ifstream is(argv[1]);
   if (!is)
   {
      cerr << "Cannot open " << argv[1] << endl;
      return -2;
   }

   int mode = 0;
   if (strcmp(argv[2], "metric") == 0)
      mode = FULL_BUNDLE_METRIC;
   else if (strcmp(argv[2], "focal") == 0)
      mode = FULL_BUNDLE_FOCAL_LENGTH;
   else if (strcmp(argv[2], "principal") == 0)
      mode = FULL_BUNDLE_FOCAL_LENGTH_PP;
   else if (strcmp(argv[2], "radial") == 0)
      mode = FULL_BUNDLE_RADIAL;
   else if (strcmp(argv[2], "tangential") == 0)
      mode = FULL_BUNDLE_RADIAL_TANGENTIAL;
   else
   {
      cerr << "Unknown bundle mode: " << argv[2] << endl;
      return -2;
   }

   int N, M, K;
   is >> M >> N >> K;
   cout << "N (cams) = " << N << " M (points) = " << M << " K (measurements) = " << K << endl;

   Matrix3x3d KMat;
   StdDistortionFunction distortion;

   makeIdentityMatrix(KMat);
   is >> KMat[0][0] >> KMat[0][1] >> KMat[0][2] >> KMat[1][1] >> KMat[1][2]
      >> distortion.k1 >> distortion.k2 >> distortion.p1 >> distortion.p2;

   double const f0 = KMat[0][0];
   cout << "intrinsic before bundle = "; displayMatrix(KMat);
   Matrix3x3d Knorm = KMat;
   // Normalize the intrinsic to have unit focal length.
   scaleMatrixIP(1.0/f0, Knorm);
   Knorm[2][2] = 1.0;

   vector<int> pointIdFwdMap(M);
   map<int, int> pointIdBwdMap;

   vector<Vector3d > Xs(M);
   for (int j = 0; j < M; ++j)
   {
      int pointId;
      is >> pointId >> Xs[j][0] >> Xs[j][1] >> Xs[j][2];
      pointIdFwdMap[j] = pointId;
      pointIdBwdMap.insert(make_pair(pointId, j));
   }
   cout << "Read the 3D points." << endl;

   vector<int> camIdFwdMap(N);
   map<int, int> camIdBwdMap;

   vector<CameraMatrix> cams(N);
   for (int i = 0; i < N; ++i)
   {
      int camId;
      Matrix3x3d R;
      Vector3d T;

      is >> camId;
      is >> R[0][0] >> R[0][1] >> R[0][2] >> T[0];
      is >> R[1][0] >> R[1][1] >> R[1][2] >> T[1];
      is >> R[2][0] >> R[2][1] >> R[2][2] >> T[2];

      camIdFwdMap[i] = camId;
      camIdBwdMap.insert(make_pair(camId, i));

      cams[i].setIntrinsic(Knorm);
#if 1
      cams[i].setRotation(R);
      cams[i].setTranslation(T);
#else
      cams[i].setRotation(transposedMatrix(R));
      cams[i].setTranslation(transposedMatrix(R) * (-1.0 * T));
#endif
   }
   cout << "Read the cameras." << endl;

   vector<Vector2d > measurements;
   vector<int> correspondingView;
   vector<int> correspondingPoint;

   measurements.reserve(K);
   correspondingView.reserve(K);
   correspondingPoint.reserve(K);

   for (int k = 0; k < K; ++k)
   {
      int view, point;
      Vector3d p;

      is >> view >> point;
      is >> p[0] >> p[1] >> p[2];

      if (camIdBwdMap.find(view) != camIdBwdMap.end() &&
          pointIdBwdMap.find(point) != pointIdBwdMap.end())
      {
         // Normalize the measurements to match the unit focal length.
         scaleVectorIP(1.0/f0, p);
         measurements.push_back(makeVector2(p[0], p[1]));
         correspondingView.push_back(camIdBwdMap[view]);
         correspondingPoint.push_back(pointIdBwdMap[point]);
      }
   } // end for (k)

   K = measurements.size();

   cout << "Read " << K << " valid 2D measurements." << endl;

   showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

   V3D::optimizerVerbosenessLevel = 1;
   double const inlierThreshold = 2.0 / f0;

   Matrix3x3d K0 = cams[0].getIntrinsic();
   cout << "K0 = "; displayMatrix(K0);

   CommonInternalsMetricBundleOptimizer opt(mode, inlierThreshold, K0, distortion, cams, Xs,
                                            measurements, correspondingView, correspondingPoint);
   opt.maxIterations = 50;
   opt.minimize();
   cout << "optimizer status = " << opt.status << endl;
   cout << "refined K = "; displayMatrix(K0);
   cout << "distortion = " << distortion.k1 << " " << distortion.k2 << " "
        << distortion.p1 << " " << distortion.p2 << endl;

   for (int i = 0; i < N; ++i) cams[i].setIntrinsic(K0);

   Matrix3x3d Knew = K0;
   scaleMatrixIP(f0, Knew);
   Knew[2][2] = 1.0;
   cout << "Knew = "; displayMatrix(Knew);

   showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

   ofstream os("refined.txt");
   os << Knew[0][0] << " " << Knew[0][1] << " " << Knew[0][2] << " " << Knew[1][1] << " " << Knew[1][2] << " "
      << distortion.k1 << " " << distortion.k2 << " "
      << distortion.p1 << " " << distortion.p2 << " " << endl;
   os << M << " " << N << " " << K << endl;

   for (int j = 0; j < M; ++j)
   {
      os << pointIdFwdMap[j] << " " << Xs[j][0] << " " << Xs[j][1] << " " << Xs[j][2] << endl;
   }

   for (int i = 0; i < N; ++i)
   {
      os << camIdFwdMap[i] << " ";
      Matrix3x4d const RT = cams[i].getOrientation();
      os << RT[0][0] << " " << RT[0][1] << " " << RT[0][2] << " " << RT[0][3] << " ";
      os << RT[1][0] << " " << RT[1][1] << " " << RT[1][2] << " " << RT[1][3] << " ";
      os << RT[2][0] << " " << RT[2][1] << " " << RT[2][2] << " " << RT[2][3] << endl;
   }

   return 0;
}
