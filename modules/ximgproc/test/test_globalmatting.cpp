// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test
{
  namespace
  {
    const std::string INPUT_DIR       = "cv/ximgproc";
    const std::string IMAGE_FILENAME  = "input/doll.png";
    const std::string TRIMAP_FILENAME = "trimap/doll.png";


    void runModel()
    {

      Ptr<GlobalMatting> gm    = makePtr<GlobalMatting>();
      std::string img_path     = cvtest::findDataFile(INPUT_DIR + "/" + IMAGE_FILENAME);
      std::string trimap_path  = cvtest::findDataFile(INPUT_DIR + "/" + TRIMAP_FILENAME);

      Mat img     = cv::imread(img_path,cv::IMREAD_COLOR);
      Mat trimap  = cv::imread(trimap_path,cv::IMREAD_GRAYSCALE);
      ASSERT_FALSE(img.empty()) << "The Image could not be loaded: "<< img_path;
      ASSERT_FALSE(trimap.empty()) << "The trimap could not be loaded: "<< trimap_path;

      ASSERT_EQ(img.cols,trimap.cols);
      ASSERT_EQ(img.rows,trimap.rows);
      Mat foreground,alpha;
      int niter = 9;
      gm->getMat(img,trimap,foreground,alpha,niter);

      ASSERT_FALSE(foreground.empty()) << " Could not extract the foreground ";
      ASSERT_FALSE(alpha.empty()) << " Could not generate alpha matte ";

      ASSERT_EQ(alpha.cols,img.cols);
      ASSERT_EQ(alpha.rows,img.rows);

    }


    TEST(CV_GlobalMattingTest,accuracy)
    {
      runModel();
    }

  }
}
