#ifndef OPENCV_DEHAZE_HPP
#define OPENCV_DEHAZE_HPP


#include "opencv2/core.hpp"


namespace cv {


CV_EXPORTS_W void computeDarkChannel(
   InputArray src,
   OutputArray dark,
   int patchSize = 15
);


CV_EXPORTS_W void estimateAtmosphericLight(
   InputArray src,
   InputArray dark,
   Vec3d& A
);


CV_EXPORTS_W void computeTransmission(
   InputArray src,
   const Vec3d& A,
   OutputArray transmission,
   int patchSize = 15,
   double omega = 0.95
);


CV_EXPORTS_W void guidedFilter(
   InputArray guide,
   InputArray src,
   OutputArray dst,
   int radius = 60,
   double eps = 0.0001
);


CV_EXPORTS_W void detectSkyRegion(
   InputArray src,
   OutputArray skyMask,
   double threshold = 0.9
);




CV_EXPORTS_W void recoverSceneRadiance(
   InputArray src,
   InputArray transmission,
   const Vec3d& A,
   OutputArray dst,
   double t0 = 0.1
);


CV_EXPORTS_W void dehazeImage(
   InputArray src,
   OutputArray dst
);


} // namespace cv


#endif