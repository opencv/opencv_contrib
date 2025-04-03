#include <gtest/gtest.h>
#include <quaternion.hpp>
#include <math.h>
#include <kfusion/types.hpp>
using namespace kfusion::utils;
TEST(QuaternionTest, encodeRotation)
{
    Quaternion<float> quaternion;
    quaternion.encodeRotation(M_PI_4, 0, 0, 1);

    ASSERT_FLOAT_EQ(0.9238795, quaternion.w_);
    ASSERT_FLOAT_EQ(0, quaternion.x_);
    ASSERT_FLOAT_EQ(0, quaternion.y_);
    ASSERT_FLOAT_EQ(0.38268346, quaternion.z_);
}

TEST(QuaternionTest, rotate)
{
    Quaternion<float> quaternion(0,0,1,1);
    float vector[] = {0,0,1};
    quaternion.rotate(vector[0], vector[1], vector[2]);
    EXPECT_EQ(vector[0], 0);
    EXPECT_EQ(vector[1], 2);
    EXPECT_EQ(vector[2], 0);
}

TEST(QuaternionTest, quat_product)
{
    Quaternion<float> quaternion(1,1,2,2);
    Quaternion<float> quaternion1(0,0,1,1);
    Quaternion<float> product = quaternion * quaternion1;
    EXPECT_EQ(product.w_, -4);
    EXPECT_EQ(product.x_, 0);
    EXPECT_EQ(product.y_, 0);
    EXPECT_EQ(product.z_, 2);
}

TEST(QuaternionTest, power)
{
    Quaternion<float> quaternion(1,1,2,2);
    EXPECT_EQ(quaternion.power(2), quaternion*quaternion);
}

TEST(QuaternionTest, dotProduct)
{
    Quaternion<float> quaternion(1,1,2,2);
    Quaternion<float> quaternion1(0,0,1,1);
    EXPECT_EQ(quaternion.dotProduct(quaternion1), 4);
}

TEST(QuaternionTest, normalize)
{
    Quaternion<float> quaternion(10,10,10,10);
    quaternion.normalize();
    EXPECT_EQ(quaternion, Quaternion<float>(0.5, 0.5, 0.5, 0.5));
}

TEST(QuaternionTest, slerp)
{
    Quaternion<float> quaternion(1,1,2,2);
    Quaternion<float> quaternion1(0,0,1,1);
    Quaternion<float> result = quaternion.slerp(quaternion1, 0.5);

    ASSERT_FLOAT_EQ(result.w_, 0.16245984);
    ASSERT_FLOAT_EQ(result.x_, 0.16245984);
    ASSERT_FLOAT_EQ(result.y_, 0.688191);
    ASSERT_FLOAT_EQ(result.z_, 0.688191);
}

TEST(QuaternionTest, rodrigues)
{
    Quaternion<float> quaternion;
    quaternion.encodeRotation(3,1,1,1);
    float x, y, z;
    quaternion.getRodrigues(x,y,z);
    std::cout<<x<<" "<<y<<" "<<z<<std::endl;
    ASSERT_FLOAT_EQ(2, x);
    ASSERT_FLOAT_EQ(4, y);
    ASSERT_FLOAT_EQ(4, z);
}

TEST(QuaternionTest, normal)
{
    kfusion::Vec3f normal(0,1,0);
    Quaternion<float> quaternion(normal);

    std::cout<<"Quaternion:" << quaternion;
}
