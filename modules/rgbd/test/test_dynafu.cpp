// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "test_precomp.hpp"

#ifdef HAVE_OPENGL
namespace opencv_test { namespace {

static std::vector<std::string> readDepth(std::string fileList)
{
    std::vector<std::string> v;

    std::fstream file(fileList);
    if(!file.is_open())
        throw std::runtime_error("Failed to read depth list");

    std::string dir;
    size_t slashIdx = fileList.rfind('/');
    slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
    dir = fileList.substr(0, slashIdx);

    while(!file.eof())
    {
        std::string s, imgPath;
        std::getline(file, s);
        if(s.empty() || s[0] == '#') continue;
        std::stringstream ss;
        ss << s;
        double thumb;
        ss >> thumb >> imgPath;
        v.push_back(dir+'/'+imgPath);
    }

    return v;
}

static const bool display = false;

void flyTest(bool hiDense, bool inequal)
{
    Ptr<kinfu::Params> params;
    if(hiDense)
        params = kinfu::Params::defaultParams();
    else
        params = kinfu::Params::coarseParams();

    if(inequal)
    {
        params->volumeDims[0] += 32;
        params->volumeDims[1] -= 32;
    }

    std::vector<String> depths = readDepth(cvtest::TS::ptr()->get_data_path() + "dynafu/depth.txt");
    CV_Assert(!depths.empty());

    Ptr<dynafu::DynaFu> df = dynafu::DynaFu::create(params);

    // Check for first 10 frames
    CV_Assert(depths.size() >= 10);
    Mat currentDepth, prevDepth;
    for(size_t i = 0; i < 10; i++)
    {
        currentDepth = cv::imread(depths[i], IMREAD_ANYDEPTH);

        ASSERT_TRUE(df->update(currentDepth));

        Mat renderedDepth;
        df->renderSurface(renderedDepth, noArray(), noArray());

        if(i > 0)
        {
            // Check if estimated depth aligns with actual depth in the previous frame
            Mat depthCvt8, renderCvt8;
            convertScaleAbs(prevDepth, depthCvt8, 0.25*256. / params->depthFactor);
            convertScaleAbs(renderedDepth, renderCvt8, 0.33*255, -0.5*0.33*255);

            Mat diff;
            absdiff(depthCvt8, renderCvt8, diff);

            Scalar_<float> mu, sigma;
            meanStdDev(diff, mu, sigma);
            std::cout << "Mean: " << mu[0] << ", Std dev: " << sigma[0] << std::endl;
        }

        if(display)
        {
            imshow("depth", currentDepth*(1.f/params->depthFactor/4.f));
            Mat rendered;
            df->render(rendered);
            imshow("render", rendered);
            waitKey(10);
        }

        currentDepth.copyTo(prevDepth);
    }
}

/*
#ifdef OPENCV_ENABLE_NONFREE
TEST( DynamicFusion, lowDense )
#else
TEST(DynamicFusion, DISABLED_lowDense)
#endif
{
    flyTest(false, false);
}

#ifdef OPENCV_ENABLE_NONFREE
TEST( DynamicFusion, inequal )
#else
TEST(DynamicFusion, DISABLED_inequal)
#endif
{
    flyTest(false, true);
}
*/

// To enable DynamicFusion tests, uncomment the above lines and delete the following lines
TEST(DynamicFusion, DISABLED)
{
    CV_UNUSED(flyTest);
}

}} // namespace

#endif
