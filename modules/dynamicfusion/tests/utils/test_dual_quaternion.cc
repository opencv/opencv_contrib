#include <gtest/gtest.h>
#include <dual_quaternion.hpp>


using namespace kfusion::utils;
TEST(DualQuaternionTest, DualQuaternionConstructor)
{
    DualQuaternion<float> dualQuaternion(1, 2, 3, 1, 2, 3);
    Quaternion<float> translation = dualQuaternion.getTranslation();
    Quaternion<float> rotation = dualQuaternion.getRotation();

    EXPECT_NEAR(translation.w_, 0, 0.001);
    EXPECT_NEAR(translation.x_, 1, 0.1);
    EXPECT_NEAR(translation.y_, 2, 0.1);
    EXPECT_NEAR(translation.z_, 3, 0.1);

    EXPECT_NEAR(rotation.w_, 0.435953, 0.01);
    EXPECT_NEAR(rotation.x_, -0.718287, 0.01);
    EXPECT_NEAR(rotation.y_, 0.310622, 0.01);
    EXPECT_NEAR(rotation.z_, 0.454649, 0.01);
}

TEST(DualQuaternionTest, get6DOF)
{
    DualQuaternion<float> dualQuaternion(1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f);
    float x, y, z, roll, pitch, yaw;
    dualQuaternion.get6DOF(x, y, z, roll, pitch, yaw);
    EXPECT_FLOAT_EQ(x, 1.0f);
    EXPECT_FLOAT_EQ(y, 2.0f);
    EXPECT_FLOAT_EQ(z, 3.0f);
    EXPECT_FLOAT_EQ(roll, 1.0f);
    EXPECT_FLOAT_EQ(pitch, 2.0f);
    EXPECT_FLOAT_EQ(yaw, 3.0f);
}

TEST(DualQuaternionTest, isAssociative)
{
    DualQuaternion<float> dualQuaternion(1, 2, 3, 1, 2, 3);
    DualQuaternion<float> dualQuaternion1(3, 4, 5, 3, 4, 5);
    dualQuaternion.encodeRotation(1,2,3);
    dualQuaternion1.encodeRotation(3,4,5);
    cv::Vec3f test11(1,0,0);
    cv::Vec3f test12(1,0,0);
    cv::Vec3f test2(1,0,0);
    auto cumul = dualQuaternion1 + dualQuaternion;
    cumul.normalize();
    dualQuaternion.transform(test11);
    dualQuaternion1.transform(test12);
    cumul.transform(test2);
    auto result = test11 + test12;

    EXPECT_NE(test2, result);
}
TEST(DualQuaternionTest, canSplitOperations)
{
    DualQuaternion<float> dualQuaternion(1, 2, 3, 1, 2, 3);
    DualQuaternion<float> dualQuaternion1(3, 4, 5, 3, 4, 5);
    dualQuaternion.encodeRotation(1,2,3);
    dualQuaternion1.encodeRotation(3,4,5);
    cv::Vec3f test1(1,0,0);
    cv::Vec3f test2(1,0,0);
    cv::Vec3f t1, t2;

    auto cumul = dualQuaternion1 + dualQuaternion;
    auto quat1 = dualQuaternion.getRotation() + dualQuaternion1.getRotation();
    dualQuaternion.getTranslation(t1);
    dualQuaternion1.getTranslation(t2);
    auto trans = t1 + t2;


    cumul.normalize();
    cumul.transform(test1);
//    auto result = test1 + test12;
    quat1.rotate(test2);
    auto result = test2 + trans;
    std::cout<<test1<<std::endl;
    std::cout<<test2<<std::endl;
    EXPECT_NE(test1, result);
}