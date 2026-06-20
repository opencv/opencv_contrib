#include "precomp.hpp"


namespace cv {


void computeDarkChannel(InputArray _src, OutputArray _dark, int patchSize)
{
   Mat src = _src.getMat();
   CV_Assert(src.type() == CV_8UC3);


   _dark.create(src.size(), CV_8UC1);
   Mat dark = _dark.getMat();


   // For each pixel, take the minimum across B, G, R channels
   for (int i = 0; i < src.rows; i++) {
       const Vec3b* row = src.ptr<Vec3b>(i);
       uchar* out = dark.ptr<uchar>(i);
       for (int j = 0; j < src.cols; j++) {
           out[j] = std::min({row[j][0], row[j][1], row[j][2]});
       }
   }


   // Apply erosion (local minimum filter) over the patch
   Mat kernel = getStructuringElement(
       MORPH_RECT, Size(patchSize, patchSize)
   );
   erode(dark, dark, kernel);
}

void estimateAtmosphericLight(InputArray _src, InputArray _dark, Vec3d& A)
{
   Mat src  = _src.getMat();
   Mat dark = _dark.getMat();


   int total      = dark.rows * dark.cols;
   int numPixels  = std::max(1, total / 1000); // top 0.1%


   // Flatten dark channel and sort indices descending
   Mat darkFlat = dark.reshape(1, total);
   Mat idx;
   sortIdx(darkFlat, idx, SORT_EVERY_COLUMN + SORT_DESCENDING);


   // Average the RGB values of those top pixels
   Mat srcFlat = src.reshape(3, total);
   Vec3d sum(0, 0, 0);
   for (int i = 0; i < numPixels; i++) {
       int id = idx.at<int>(i, 0);
       Vec3b pixel = srcFlat.at<Vec3b>(id);
       sum += Vec3d(pixel[0], pixel[1], pixel[2]);
   }
   A = sum / static_cast<double>(numPixels);
}


void computeTransmission(InputArray _src, const Vec3d& A,
                         OutputArray _t, int patchSize, double omega)
{
   Mat src = _src.getMat();
   CV_Assert(src.type() == CV_8UC3);


   // Normalize each channel by atmospheric light
   Mat normalized(src.size(), CV_8UC3);
   for (int i = 0; i < src.rows; i++) {
       const Vec3b* in  = src.ptr<Vec3b>(i);
       Vec3b*       out = normalized.ptr<Vec3b>(i);
       for (int j = 0; j < src.cols; j++) {
           for (int c = 0; c < 3; c++) {
               out[j][c] = saturate_cast<uchar>(
                   in[j][c] / A[c] * 255.0
               );
           }
       }
   }


   // Dark channel of normalized image
   Mat darkNorm;
   computeDarkChannel(normalized, darkNorm, patchSize);


   // t(x) = 1 - omega * darkNorm(x)
   darkNorm.convertTo(_t, CV_64F, -omega / 255.0, 1.0);
}


void detectSkyRegion(InputArray _src, OutputArray _skyMask, double threshold)
{
   Mat src = _src.getMat();


   // Convert to HSV — sky is high value, low saturation
   Mat hsv;
   cvtColor(src, hsv, COLOR_BGR2HSV);


   std::vector<Mat> channels;
   split(hsv, channels);


   Mat saturation = channels[1];
   Mat value      = channels[2];  


   // Sky = bright (high V) and not colorful (low S)
   Mat brightMask, desaturatedMask;
   cv::threshold(value,      brightMask,      230, 255, THRESH_BINARY);
   cv::threshold(saturation, desaturatedMask,  30, 255, THRESH_BINARY_INV);


   // Combine both conditions
   Mat skyMask;
   bitwise_and(brightMask, desaturatedMask, skyMask);


   // Clean up noise with morphological ops
   Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
   morphologyEx(skyMask, skyMask, MORPH_CLOSE, kernel);
   morphologyEx(skyMask, skyMask, MORPH_OPEN,  kernel);


   skyMask.copyTo(_skyMask);
}


void guidedFilter(InputArray _guide, InputArray _src,
                 OutputArray _dst, int radius, double eps)
{
   Mat guide = _guide.getMat();
   Mat src   = _src.getMat();


   // Convert to float
   Mat I, p;
   guide.convertTo(I, CV_64F, 1.0 / 255.0);
   src.convertTo(p, CV_64F);


   Size win(2 * radius + 1, 2 * radius + 1);


   // Mean of guide, src, guide*src, guide*guide
   Mat mean_I, mean_p, mean_Ip, mean_II;
   boxFilter(I,   mean_I,  CV_64F, win);
   boxFilter(p,   mean_p,  CV_64F, win);
   boxFilter(I.mul(p), mean_Ip, CV_64F, win);
   boxFilter(I.mul(I), mean_II, CV_64F, win);


   // Covariance and variance
   Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
   Mat var_I  = mean_II - mean_I.mul(mean_I);


   // Linear coefficients a and b
   Mat a = cov_Ip / (var_I + eps);
   Mat b = mean_p - a.mul(mean_I);


   // Mean of a and b
   Mat mean_a, mean_b;
   boxFilter(a, mean_a, CV_64F, win);
   boxFilter(b, mean_b, CV_64F, win);


   // Final output
   Mat result = mean_a.mul(I) + mean_b;
   result.copyTo(_dst);
}

void recoverSceneRadiance(InputArray _src, InputArray _t,
                          const Vec3d& A, OutputArray _dst, double t0)
{
   Mat src = _src.getMat();
   Mat t   = _t.getMat();


   _dst.create(src.size(), CV_8UC3);
   Mat dst = _dst.getMat();


   // J(x) = (I(x) - A) / max(t(x), t0) + A
   for (int i = 0; i < src.rows; i++) {
       const Vec3b* srcRow = src.ptr<Vec3b>(i);
       const double* tRow  = t.ptr<double>(i);
       Vec3b* dstRow       = dst.ptr<Vec3b>(i);


       for (int j = 0; j < src.cols; j++) {
           double tVal = std::max(tRow[j], t0);
           for (int c = 0; c < 3; c++) {
               double val = (srcRow[j][c] - A[c]) / tVal + A[c];
               dstRow[j][c] = saturate_cast<uchar>(val);
           }
       }
   }
}

void dehazeImage(InputArray _src, OutputArray _dst)
{
   Mat src = _src.getMat();
   CV_Assert(src.type() == CV_8UC3);


   // 1. Dark channel
   Mat dark;
   computeDarkChannel(src, dark, 15);


   // 2. Atmospheric light
   Vec3d A;
   estimateAtmosphericLight(src, dark, A);


   // 3. Transmission map
   Mat transmission;
   computeTransmission(src, A, transmission, 15, 0.95);


   // 4. Refine transmission map with guided filter
   Mat gray;
   cvtColor(src, gray, COLOR_BGR2GRAY);
   Mat refinedTransmission;
   guidedFilter(gray, transmission, refinedTransmission, 60, 0.0001);


   // 5. Detect sky region and protect it
   Mat skyMask;
   detectSkyRegion(src, skyMask);

   int skyLine = src.rows / 8;


   for (int i = 0; i < refinedTransmission.rows; i++) {
       double* tRow = refinedTransmission.ptr<double>(i);
       const uchar* sRow = skyMask.ptr<uchar>(i);
       for (int j = 0; j < refinedTransmission.cols; j++) {
           if (i < skyLine && sRow[j] > 0) {
               // Pure sky — leave untouched
               tRow[j] = 0.95;
           } else if (i < skyLine * 2 && sRow[j] > 0) {
               // Transition zone — partial dehazing
               double blend = (double)(i - skyLine) / skyLine;
               tRow[j] = 0.95 * (1.0 - blend) + tRow[j] * blend;
           }
           // Everything else — full dehazing as normal
       }
   }


   // 6. Recover scene radiance using refined transmission
   recoverSceneRadiance(src, refinedTransmission, A, _dst, 0.1);
}


} // namespace cv