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
    Color color = Color((Mat_<double>(1, 3) <<0.3, 0.2, 0.5), sRGB)；
    Color color_rgb = color.to(sRGB);
    Color color_rgbl = color.to(sRGBL);
    Color color_xyz = color.to(XYZ_D65_2);
    Color color_lab = color.to(Lab_D65_2);
    Color color_xyz_d50 = color.to(XYZ_D50_2);
    Color color_lab_d50 = color.to(Lab_D50_2);

    ASSERT_EQ(color_rgb.colors, (Mat_<double>(1, 3) <<0.3, 0.2, 0.5));
    ASSERT_EQ(color_rgbl.colors, (Mat_<double>(1, 3) <<0.07323896, 0.03310477, 0.21404114));
    ASSERT_EQ(color_xyz.colors, (Mat_<double>(1, 3) <<0.080666, 0.054699, 0.208766));
    ASSERT_EQ(color_lab.colors, (Mat_<double>(1, 3) <<28.0337, 29.9289, -39.4065));
    ASSERT_EQ(color_xyz_d50.colors, (Mat_<double>(1, 3) <<0.075310, 0.053003, 0.157097));
    ASSERT_EQ(color_lab_d50.colors, (Mat_<double>(1, 3) <<27.5736, 25.9112, -39.9261));
}

TEST(CV_ccmColor, test_adobergbl)
{
    Color color = Color((Mat_<double>(2, 3) <<0.3, 0.2, 0.5, 0.7, 0.1, 0.4), AdobeRGBL)；
    Color color_rgb = color.to(AdobeRGB);
    Color color_rgbl = color.to(AdobeRGBL);
    Color color_xyz = color.to(XYZ_D65_2);
    Color color_lab = color.to(Lab_D65_2);
    Color color_xyz_d50 = color.to(XYZ_D50_2);
    Color color_lab_d50 = color.to(Lab_D50_2);

    ASSERT_EQ(color_rgb.colors, (Mat_<double>(2, 3) <<0.578533, 0.481157, 0.729740, 0.850335, 0.351119, 0.659353));
    ASSERT_EQ(color_rgbl.colors, (Mat_<double>(2, 3) <<0.3, 0.2, 0.5, 0.7, 0.1, 0.4));
    ASSERT_EQ(color_xyz.colors, (Mat_<double>(2, 3) <<0.304223, 0.252320, 0.517802, 0.497541, 0.301008, 0.422436));
    ASSERT_EQ(color_lab.colors, (Mat_<double>(2, 3) <<57.3008, 26.0707, -29.7295, 61.7411, 67.8735, -11.8328));
    ASSERT_EQ(color_xyz_d50.colors, (Mat_<double>(2, 3) <<0.298587, 0.250078, 0.390442, 0.507043, 0.305640, 0.317661));
    ASSERT_EQ(color_lab_d50.colors, (Mat_<double>(2, 3) <<57.0831, 23.2605, -29.8401, 62.1379, 66.7756, -10.7684));
}

TEST(CV_ccmColor, test_xyz)
{
    Color color = Color((Mat_<double>(1, 3) <<0.3, 0.2, 0.5), XYZ_D65_2)；
    Color color_rgb = color.to(ProPhotoRGB, VON_KRIES);
    Color color_rgbl = color.to(ProPhotoRGB, VON_KRIES);
    Color color_xyz = color.to(XYZ_D65_2, VON_KRIES);
    Color color_lab = color.to(Lab_D65_2, VON_KRIES);
    Color color_xyz_d50 = color.to(XYZ_D50_2, VON_KRIES);
    Color color_lab_d50 = color.to(Lab_D50_2, VON_KRIES);

    ASSERT_EQ(color_rgb.colors, (Mat_<double>(1, 3) <<0.530513, 0.351224, 0.648975));
    ASSERT_EQ(color_rgbl.colors, (Mat_<double>(1, 3) <<0.319487, 0.152073, 0.459209));
    ASSERT_EQ(color_xyz.colors, (Mat_<double>(1, 3) <<0.3, 0.2, 0.5));
    ASSERT_EQ(color_lab.colors, (Mat_<double>(1, 3) <<51.8372, 48.0307, -37.3395));
    ASSERT_EQ(color_xyz_d50.colors, (Mat_<double>(1, 3) <<0.289804, 0.200321, 0.378944));
    ASSERT_EQ(color_lab_d50.colors, (Mat_<double>(1, 3) <<51.8735, 42.3654, -37.2770));
}

TEST(CV_ccmColor, test_lab)
{
    Color color = Color((Mat_<double>(1, 3) <<30., 20., 10.), Lab_D50_2)；
    Color color_rgb = color.to(AppleRGB, IDENTITY);
    Color color_rgbl = color.to(AppleRGBL, IDENTITY);
    Color color_xyz = color.to(XYZ_D65_2, IDENTITY);
    Color color_lab = color.to(Lab_D65_2, IDENTITY);
    Color color_xyz_d50 = color.to(XYZ_D50_2, IDENTITY);
    Color color_lab_d50 = color.to(Lab_D50_2, IDENTITY);

    ASSERT_EQ(color_rgb.colors, (Mat_<double>(1, 3) <<0.323999, 0.167314, 0.165874));
    ASSERT_EQ(color_rgbl.colors, (Mat_<double>(1, 3) <<0.131516, 0.040028, 0.039410));
    ASSERT_EQ(color_xyz.colors, (Mat_<double>(1, 3) <<0.079076, 0.062359, 0.045318));
    ASSERT_EQ(color_lab.colors, (Mat_<double>(1, 3) <<30.0001, 19.9998, 9.9999));
    ASSERT_EQ(color_xyz_d50.colors, (Mat_<double>(1, 3) <<0.080220, 0.062359, 0.034345));
    ASSERT_EQ(color_lab_d50.colors, (Mat_<double>(1, 3) <<30., 20., 10.));
}

TEST(CV_ccmColor, test_grays)
{
    Mat grays = (Mat_<bool>(24, 1) <<
                False, False, False, False, False, False,
                False, False, False, False, False, False,
                False, False, False, False, False, False,
                True, True, True, True, True, True);
    Macbeth_D50_2.getGray();
    Macbeth_D65_2.getGray();

    ASSERT_EQ(Macbeth_D50_2.grays, grays);
    ASSERT_EQ(Macbeth_D65_2.grays, grays);
}

TEST(CV_ccmColor, test_gray_luminant)
{
    Color color = Color((Mat_<double>(1, 3) <<0.3, 0.2, 0.5), sRGB);
    ASSERT_EQ(color.toGray(color.cs.io), (Mat_<double>(1, 1) <<0.054699));
    ASSERT_EQ(color.toLuminant(color.cs.io), (Mat_<double>(1, 1) <<28.0337));

    Color color = Color((Mat_<double>(2, 3) <<0.3, 0.2, 0.5, 0.7, 0.1, 0.4), sRGB);
    ASSERT_EQ(color.toGray(color.cs.io), (Mat_<double>(1, 2) <<0.054699, 0.112033));
    ASSERT_EQ(color.toLuminant(color.cs.io), (Mat_<double>(1, 2) <<28.0337, 39.9207));
}

TEST(CV_ccmColor, test_diff)
{
    Color color1 = Color((Mat_<double>(1, 3) <<0.3, 0.2, 0.5), sRGB);
    Color color2 = Color((Mat_<double>(1, 3) <<0.3, 0.2, 0.5), XYZ_D50_2);

    ASSERT_EQ(color1.diff(color2, method=CIE2000, io=D65_2), (Mat_<double>(1, 1) <<22.58031));
    ASSERT_EQ(color1.diff(color2, method=CIE94_GRAPHIC_ARTS, io=D65_2), (Mat_<double>(1, 1) <<25.701214));
    ASSERT_EQ(color1.diff(color2, method=CIE76, io=D65_2), (Mat_<double>(1, 1) <<34.586351));
    ASSERT_EQ(color1.diff(color2, method=CMC_1TO1, io=D65_2), (Mat_<double>(1, 1) <<33.199419));
    ASSERT_EQ(color1.diff(color2, method=RGB, io=D65_2), (Mat_<double>(1, 1) <<0.51057));
    ASSERT_EQ(color1.diff(color2, method=RGBL, io=D65_2), (Mat_<double>(1, 1) <<0.556741));
}

} // namespace
} // namespace opencv_test