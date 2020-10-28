// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

TEST(CV_ccmColor, test_srgb)
{
    Color color = Color((Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5)), sRGB);
    Color color_rgb = color.to(sRGB);
    Color color_rgbl = color.to(sRGBL);
    Color color_xyz = color.to(XYZ_D65_2);
    Color color_lab = color.to(Lab_D65_2);
    Color color_xyz_d50 = color.to(XYZ_D50_2);
    Color color_lab_d50 = color.to(Lab_D50_2);

    Mat color_rgb_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5));
    Mat color_rgbl_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.07323896, 0.03310477, 0.21404114));
    Mat color_xyz_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.080666, 0.054699, 0.208766));
    Mat color_lab_M = (Mat_<Vec3d>(1, 1) << Vec3d(28.0337, 29.9289, -39.4065));
    Mat color_xyz_d50_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.075310, 0.053003, 0.157097));
    Mat color_lab_d50_M = (Mat_<Vec3d>(1, 1) << Vec3d(27.5736, 25.9112, -39.9261));

    ASSERT_MAT_NEAR(color_rgb.colors, color_rgb_M, 1e-4);
    ASSERT_MAT_NEAR(color_rgbl.colors, color_rgbl_M, 1e-4);
    ASSERT_MAT_NEAR(color_xyz.colors, color_xyz_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab.colors, color_lab_M, 1e-2);
    ASSERT_MAT_NEAR(color_xyz_d50.colors, color_xyz_d50_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab_d50.colors, color_lab_d50_M, 1e-2);
}

TEST(CV_ccmColor, test_adobergbl)
{
    Color color = Color((Mat_<Vec3d>(2, 1) << Vec3d(0.3, 0.2, 0.5), Vec3d(0.7, 0.1, 0.4)), AdobeRGBL);
    Color color_rgb = color.to(AdobeRGB);
    Color color_rgbl = color.to(AdobeRGBL);
    Color color_xyz = color.to(XYZ_D65_2);
    Color color_lab = color.to(Lab_D65_2);
    Color color_xyz_d50 = color.to(XYZ_D50_2);
    Color color_lab_d50 = color.to(Lab_D50_2);

    Mat color_rgb_M = (Mat_<Vec3d>(2, 1) << Vec3d(0.578533, 0.481157, 0.729740), Vec3d(0.850335, 0.351119, 0.659353));
    Mat color_rgbl_M = (Mat_<Vec3d>(2, 1) << Vec3d(0.3, 0.2, 0.5), Vec3d(0.7, 0.1, 0.4));
    Mat color_xyz_M = (Mat_<Vec3d>(2, 1) << Vec3d(0.304223, 0.252320, 0.517802), Vec3d(0.497541, 0.301008, 0.422436));
    Mat color_lab_M = (Mat_<Vec3d>(2, 1) << Vec3d(57.3008, 26.0707, -29.7295), Vec3d(61.7411, 67.8735, -11.8328));
    Mat color_xyz_d50_M = (Mat_<Vec3d>(2, 1) << Vec3d(0.298587, 0.250078, 0.390442), Vec3d(0.507043, 0.305640, 0.317661));
    Mat color_lab_d50_M = (Mat_<Vec3d>(2, 1) << Vec3d(57.0831, 23.2605, -29.8401), Vec3d(62.1379, 66.7756, -10.7684));

    ASSERT_MAT_NEAR(color_rgb.colors, color_rgb_M, 1e-4);
    ASSERT_MAT_NEAR(color_rgbl.colors, color_rgbl_M, 1e-4);
    ASSERT_MAT_NEAR(color_xyz.colors, color_xyz_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab.colors, color_lab_M, 1e-2);
    ASSERT_MAT_NEAR(color_xyz_d50.colors, color_xyz_d50_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab_d50.colors, color_lab_d50_M, 1e-2);
}

TEST(CV_ccmColor, test_xyz)
{
    Color color = Color((Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5)), XYZ_D65_2);
    Color color_rgb = color.to(ProPhotoRGB, VON_KRIES);
    Color color_rgbl = color.to(ProPhotoRGBL, VON_KRIES);
    Color color_xyz = color.to(XYZ_D65_2, VON_KRIES);
    Color color_lab = color.to(Lab_D65_2, VON_KRIES);
    Color color_xyz_d50 = color.to(XYZ_D50_2, VON_KRIES);
    Color color_lab_d50 = color.to(Lab_D50_2, VON_KRIES);

    Mat color_rgb_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.530513, 0.351224, 0.648975));
    Mat color_rgbl_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.319487, 0.152073, 0.459209));
    Mat color_xyz_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5));
    Mat color_lab_M = (Mat_<Vec3d>(1, 1) << Vec3d(51.8372, 48.0307, -37.3395));
    Mat color_xyz_d50_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.289804, 0.200321, 0.378944));
    Mat color_lab_d50_M = (Mat_<Vec3d>(1, 1) << Vec3d(51.8735, 42.3654, -37.2770));

    ASSERT_MAT_NEAR(color_rgb.colors, color_rgb_M, 1e-4);
    ASSERT_MAT_NEAR(color_rgbl.colors, color_rgbl_M, 1e-4);
    ASSERT_MAT_NEAR(color_xyz.colors, color_xyz_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab.colors, color_lab_M, 1e-2);
    ASSERT_MAT_NEAR(color_xyz_d50.colors, color_xyz_d50_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab_d50.colors, color_lab_d50_M, 1e-2);
}

TEST(CV_ccmColor, test_lab)
{
    Color color = Color((Mat_<Vec3d>(1, 1) << Vec3d(30., 20., 10.)), Lab_D50_2);
    Color color_rgb = color.to(AppleRGB, IDENTITY);
    Color color_rgbl = color.to(AppleRGBL, IDENTITY);
    Color color_xyz = color.to(XYZ_D65_2, IDENTITY);
    Color color_lab = color.to(Lab_D65_2, IDENTITY);
    Color color_xyz_d50 = color.to(XYZ_D50_2, IDENTITY);
    Color color_lab_d50 = color.to(Lab_D50_2, IDENTITY);

    Mat color_rgb_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.323999, 0.167314, 0.165874));
    Mat color_rgbl_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.131516, 0.040028, 0.039410));
    Mat color_xyz_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.079076, 0.062359, 0.045318));
    Mat color_lab_M = (Mat_<Vec3d>(1, 1) << Vec3d(30.0001, 19.9998, 9.9999));
    Mat color_xyz_d50_M = (Mat_<Vec3d>(1, 1) << Vec3d(0.080220, 0.062359, 0.034345));
    Mat color_lab_d50_M = (Mat_<Vec3d>(1, 1) << Vec3d(30., 20., 10.));

    ASSERT_MAT_NEAR(color_rgb.colors, color_rgb_M, 1e-4);
    ASSERT_MAT_NEAR(color_rgbl.colors, color_rgbl_M, 1e-4);
    ASSERT_MAT_NEAR(color_xyz.colors, color_xyz_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab.colors, color_lab_M, 1e-2);
    ASSERT_MAT_NEAR(color_xyz_d50.colors, color_xyz_d50_M, 1e-4);
    ASSERT_MAT_NEAR(color_lab_d50.colors, color_lab_d50_M, 1e-2);
}

TEST(CV_ccmColor, test_grays)
{
    const double ColorChecker2005_LAB_D50_2 [24][3] =
   { {37.986, 13.555, 14.059},
    {65.711, 18.13, 17.81},
    {49.927, -4.88, -21.925},
    {43.139, -13.095, 21.905},
    {55.112, 8.844, -25.399},
    {70.719, -33.397, -0.199},
    {62.661, 36.067, 57.096},
    {40.02, 10.41, -45.964},
    {51.124, 48.239, 16.248},
    {30.325, 22.976, -21.587},
    {72.532, -23.709, 57.255},
    {71.941, 19.363, 67.857},
    {28.778, 14.179, -50.297},
    {55.261, -38.342, 31.37},
    {42.101, 53.378, 28.19},
    {81.733, 4.039, 79.819},
    {51.935, 49.986, -14.574},
    {51.038, -28.631, -28.638},
    {96.539, -0.425, 1.186},
    {81.257, -0.638, -0.335},
    {66.766, -0.734, -0.504},
    {50.867, -0.153, -0.27},
    {35.656, -0.421, -1.231},
    {20.461, -0.079, -0.973}};

const double ColorChecker2005_LAB_D65_2[24][3]=
    {{37.542, 12.018, 13.33},
    {65.2, 14.821, 17.545},
    {50.366, -1.573, -21.431},
    {43.125, -14.63, 22.12},
    {55.343, 11.449, -25.289},
    {71.36, -32.718, 1.636},
    {61.365, 32.885, 55.155},
    {40.712, 16.908, -45.085},
    {49.86, 45.934, 13.876},
    {30.15, 24.915, -22.606},
    {72.438, -27.464, 58.469},
    {70.916, 15.583, 66.543},
    {29.624, 21.425, -49.031},
    {55.643, -40.76, 33.274},
    {40.554, 49.972, 25.46},
    {80.982, -1.037, 80.03},
    {51.006, 49.876, -16.93},
    {52.121, -24.61, -26.176},
    {96.536, -0.694, 1.354},
    {81.274, -0.61, -0.24},
    {66.787, -0.647, -0.429},
    {50.872, -0.059, -0.247},
    {35.68, -0.22, -1.205},
    {20.475, 0.049, -0.972}};
    Mat ColorChecker2005_LAB_D50_2_test = GetColor::get_ColorChecker(*ColorChecker2005_LAB_D50_2,24);
    Mat ColorChecker2005_LAB_D65_2_test = GetColor::get_ColorChecker(*ColorChecker2005_LAB_D65_2,24);
    Color color_d50_2(ColorChecker2005_LAB_D50_2_test, Lab_D50_2);

    Color color_d65_2(ColorChecker2005_LAB_D65_2_test, Lab_D65_2);

    Mat grays = (Mat_<uchar>(24, 1) <<
                false, false, false, false, false, false,
                false, false, false, false, false, false,
                false, false, false, false, false, false,
                true, true, true, true, true, true);
    color_d50_2.getGray();
    color_d65_2.getGray();

    ASSERT_MAT_NEAR(color_d50_2.grays > 0, grays > 0, 0.0);
    ASSERT_MAT_NEAR(color_d65_2.grays > 0, grays > 0, 0.0);
}

TEST(CV_ccmColor, test_gray_luminant)
{
    Color color1 = Color((Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5)), sRGB);
    Mat toGray1 = (Mat_<double>(1, 1) <<0.054699);
    Mat toLuminant1 = (Mat_<double>(1, 1) <<28.0337);
    ASSERT_MAT_NEAR(color1.toGray(color1.cs.io), toGray1, 1e-4);
    ASSERT_MAT_NEAR(color1.toLuminant(color1.cs.io), toLuminant1, 1e-4);

    Color color2 = Color((Mat_<Vec3d>(2, 1) << Vec3d(0.3, 0.2, 0.5), Vec3d(0.7, 0.1, 0.4)), sRGB);
    Mat toGray2 = (Mat_<double>(2, 1) <<0.054699, 0.112033);
    Mat toLuminant2 = (Mat_<double>(2, 1) <<28.0337, 39.9207);
    ASSERT_MAT_NEAR(color2.toGray(color2.cs.io), toGray2, 1e-4);
    ASSERT_MAT_NEAR(color2.toLuminant(color2.cs.io), toLuminant2, 1e-4);
}

TEST(CV_ccmColor, test_diff)
{
    Color color1 = Color((Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5)), sRGB);
    Color color2 = Color((Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5)), XYZ_D50_2);

    Mat diff_CIE2000 = (Mat_<double>(1, 1) <<22.58031);
    Mat diff_CIE94_GRAPHIC_ARTS = (Mat_<double>(1, 1) <<25.701214);
    Mat diff_CIE76 = (Mat_<double>(1, 1) <<34.586351);
    Mat diff_CMC_1TO1 = (Mat_<double>(1, 1) <<33.199419);
    Mat diff_RGB = (Mat_<double>(1, 1) <<0.51057);
    Mat diff_RGBL = (Mat_<double>(1, 1) <<0.556741);

    ASSERT_MAT_NEAR(color1.diff(color2, D65_2, CIE2000), diff_CIE2000, 1e-2);
    ASSERT_MAT_NEAR(color1.diff(color2, D65_2, CIE94_GRAPHIC_ARTS), diff_CIE94_GRAPHIC_ARTS, 1e-2);
    ASSERT_MAT_NEAR(color1.diff(color2, D65_2, CIE76), diff_CIE76, 1e-2);
    ASSERT_MAT_NEAR(color1.diff(color2, D65_2, CMC_1TO1), diff_CMC_1TO1, 1e-2);
    ASSERT_MAT_NEAR(color1.diff(color2, D65_2, RGB), diff_RGB, 1e-4);
    ASSERT_MAT_NEAR(color1.diff(color2, D65_2, RGBL), diff_RGBL, 1e-4);
}

} // namespace
} // namespace opencv_test