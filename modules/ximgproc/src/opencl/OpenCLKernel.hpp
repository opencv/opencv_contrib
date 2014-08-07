/*
 * Copyright (C) 2013 René Jeschke <rene_jeschke@yahoo.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPENCLKERNEL_HPP_
#define OPENCLKERNEL_HPP_

#include <cstddef>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef size_t uintptr_t;
typedef ptrdiff_t intptr_t;

#define CHAR_BIT         8
#define SCHAR_MAX        127
#define SCHAR_MIN        (-127-1)
#define CHAR_MAX         SCHAR_MAX
#define CHAR_MIN         SCHAR_MIN
#define UCHAR_MAX        255
#define SHRT_MAX         32767
#define SHRT_MIN         (-32767-1)
#define USHRT_MAX        65535
#define INT_MAX          2147483647
#define INT_MIN          (-2147483647-1)
#define UINT_MAX         0xffffffffU
#define LONG_MAX         ((long) 0x7FFFFFFFFFFFFFFFLL)
#define LONG_MIN         ((long) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define ULONG_MAX        ((ulong) 0xFFFFFFFFFFFFFFFFULL)

#define FLT_DIG          6
#define FLT_MANT_DIG     24
#define FLT_MAX_10_EXP   +38
#define FLT_MAX_EXP      +128
#define FLT_MIN_10_EXP   -37
#define FLT_MIN_EXP      -125
#define FLT_RADIX        2
#define FLT_MAX          340282346638528859811704183484516925440.0f
#define FLT_MIN          1.175494350822287507969e-38f
#define FLT_EPSILON      0x1.0p-23f

#define DBL_DIG          15
#define DBL_MANT_DIG     53
#define DBL_MAX_10_EXP   +308
#define DBL_MAX_EXP      +1024
#define DBL_MIN_10_EXP   -307
#define DBL_MIN_EXP      -1021
#define DBL_RADIX        2
#define DBL_MAX          179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0
#define DBL_MIN          2.225073858507201383090e-308
#define DBL_EPSILON      2.220446049250313080847e-16

#define HUGE_VALF        ((float) 1e50)
#define HUGE_VAL         ((double) 1e500)
#define MAXFLOAT         FLT_MAX
#define INFINITY         HUGE_VALF
#define NAN              (INFINITY - INFINITY)

#define M_E              2.718281828459045090796
#define M_LOG2E          1.442695040888963387005
#define M_LOG10E         0.434294481903251816668
#define M_LN2            0.693147180559945286227
#define M_LN10           2.302585092994045901094
#define M_PI             3.141592653589793115998
#define M_PI_2           1.570796326794896557999
#define M_PI_4           0.785398163397448278999
#define M_1_PI           0.318309886183790691216
#define M_2_PI           0.636619772367581382433
#define M_2_SQRTPI       1.128379167095512558561
#define M_SQRT2          1.414213562373095145475
#define M_SQRT1_2        0.707106781186547572737

#define M_E_F            2.71828174591064f
#define M_LOG2E_F        1.44269502162933f
#define M_LOG10E_F       0.43429449200630f
#define M_LN2_F          0.69314718246460f
#define M_LN10_F         2.30258512496948f
#define M_PI_F           3.14159274101257f
#define M_PI_2_F         1.57079637050629f
#define M_PI_4_F         0.78539818525314f
#define M_1_PI_F         0.31830987334251f
#define M_2_PI_F         0.63661974668503f
#define M_2_SQRTPI_F     1.12837922573090f
#define M_SQRT2_F        1.41421353816986f
#define M_SQRT1_2_F      0.70710676908493f

#define CLK_LOCAL_MEM_FENCE 0
#define CLK_GLOBAL_MEM_FENCE 1

#define CLK_SNORM_INT8 0
#define CLK_SNORM_INT16 1
#define CLK_UNORM_INT8 2
#define CLK_UNORM_INT16 3
#define CLK_UNORM_SHORT_565 4
#define CLK_UNORM_SHORT_555 5
#define CLK_UNORM_SHORT_101010 6
#define CLK_SIGNED_INT8 7
#define CLK_SIGNED_INT16 8
#define CLK_SIGNED_INT32 9
#define CLK_UNSIGNED_INT8 10
#define CLK_UNSIGNED_INT16 11
#define CLK_UNSIGNED_INT32 12
#define CLK_HALF_FLOAT 13
#define CLK_FLOAT 14

#define CLK_A 0
#define CLK_R 1
#define CLK_Rx 2
#define CLK_RG 3
#define CLK_RGx 4
#define CLK_RA 5
#define CLK_RGB 6
#define CLK_RGBx 7
#define CLK_RGBA 8
#define CLK_ARGB 9
#define CLK_BGRA 10
#define CLK_INTENSITY 11
#define CLK_LUMINANCE 12

#define CLK_NORMALIZED_COORDS_TRUE 0
#define CLK_NORMALIZED_COORDS_FALSE 1
#define CLK_ADDRESS_MIRRORED_REPEAT 2
#define CLK_ADDRESS_REPEAT 3
#define CLK_ADDRESS_CLAMP_TO_EDGE 4
#define CLK_ADDRESS_CLAMP 5
#define CLK_ADDRESS_NONE 6
#define CLK_FILTER_NEAREST 7
#define CLK_FILTER_LINEAR 8

#define __global
#define __local
#define __constant const
#define __private
#define __kernel
#define __read_only const
#define __read_write
#define __write_only

#define global
#define local
#define constant const
//#define private
#define kernel
#define read_only const
#define read_write
#define write_only

class char2
{
public:
    char2(const char a);
    char x, y, lo, hi, even, odd;
    char2 xx, yx, xy, yy;
};

class uchar2
{
public:
    uchar2(const uchar a);
    uchar x, y, lo, hi, even, odd;
    uchar2 xx, yx, xy, yy;
};

class short2
{
public:
    short2(const short a);
    short x, y, lo, hi, even, odd;
    short2 xx, yx, xy, yy;
};

class ushort2
{
public:
    ushort2(const ushort a);
    ushort x, y, lo, hi, even, odd;
    ushort2 xx, yx, xy, yy;
};

class int2
{
public:
    int2(const int a);
    int x, y, lo, hi, even, odd;
    int2 xx, yx, xy, yy;
};

class uint2
{
public:
    uint2(const uint a);
    uint x, y, lo, hi, even, odd;
    uint2 xx, yx, xy, yy;
};

class long2
{
public:
    long2(const long a);
    long x, y, lo, hi, even, odd;
    long2 xx, yx, xy, yy;
};

class ulong2
{
public:
    ulong2(const ulong a);
    ulong x, y, lo, hi, even, odd;
    ulong2 xx, yx, xy, yy;
};

class float2
{
public:
    float2(const float a);
    float x, y, lo, hi, even, odd;
    float2 xx, yx, xy, yy;
};

class double2
{
public:
    double2(const double a);
    double x, y, lo, hi, even, odd;
    double2 xx, yx, xy, yy;
};

class char3
{
public:
    char3(const char a);
    char x, y, z;
    char2 lo, hi, even, odd;
    char2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    char3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class uchar3
{
public:
    uchar3(const uchar a);
    uchar x, y, z;
    uchar2 lo, hi, even, odd;
    uchar2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    uchar3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class short3
{
public:
    short3(const short a);
    short x, y, z;
    short2 lo, hi, even, odd;
    short2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    short3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class ushort3
{
public:
    ushort3(const ushort a);
    ushort x, y, z;
    ushort2 lo, hi, even, odd;
    ushort2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    ushort3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class int3
{
public:
    int3(const int a);
    int x, y, z;
    int2 lo, hi, even, odd;
    int2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    int3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class uint3
{
public:
    uint3(const uint a);
    uint x, y, z;
    uint2 lo, hi, even, odd;
    uint2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    uint3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class long3
{
public:
    long3(const long a);
    long x, y, z;
    long2 lo, hi, even, odd;
    long2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    long3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class ulong3
{
public:
    ulong3(const ulong a);
    ulong x, y, z;
    ulong2 lo, hi, even, odd;
    ulong2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    ulong3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class float3
{
public:
    float3(const float a);
    float x, y, z;
    float2 lo, hi, even, odd;
    float2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    float3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class double3
{
public:
    double3(const double a);
    double x, y, z;
    double2 lo, hi, even, odd;
    double2 xx, yx, zx, xy, yy, zy, xz, yz, zz;
    double3 xxx, yxx, zxx, xyx, yyx, zyx, xzx, yzx, zzx, xxy, yxy, zxy, xyy, yyy, zyy, xzy, yzy, zzy, xxz, yxz, zxz, xyz, yyz, zyz, xzz, yzz, zzz;
};

class char4
{
public:
    char4(const char a);
    char x, y, z, w;
    char2 lo, hi, even, odd;
    char2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    char3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    char4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class uchar4
{
public:
    uchar4(const uchar a);
    uchar x, y, z, w;
    uchar2 lo, hi, even, odd;
    uchar2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    uchar3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    uchar4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class short4
{
public:
    short4(const short a);
    short x, y, z, w;
    short2 lo, hi, even, odd;
    short2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    short3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    short4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class ushort4
{
public:
    ushort4(const ushort a);
    ushort x, y, z, w;
    ushort2 lo, hi, even, odd;
    ushort2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    ushort3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    ushort4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class int4
{
public:
    int4(const int a);
    int x, y, z, w;
    int2 lo, hi, even, odd;
    int2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    int3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    int4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class uint4
{
public:
    uint4(const uint a);
    uint x, y, z, w;
    uint2 lo, hi, even, odd;
    uint2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    uint3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    uint4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class long4
{
public:
    long4(const long a);
    long x, y, z, w;
    long2 lo, hi, even, odd;
    long2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    long3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    long4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class ulong4
{
public:
    ulong4(const ulong a);
    ulong x, y, z, w;
    ulong2 lo, hi, even, odd;
    ulong2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    ulong3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    ulong4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class float4
{
public:
    float4(const float a);
    float x, y, z, w;
    float2 lo, hi, even, odd;
    float2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    float3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    float4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class double4
{
public:
    double4(const double a);
    double x, y, z, w;
    double2 lo, hi, even, odd;
    double2 xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww;
    double3 xxx, yxx, zxx, wxx, xyx, yyx, zyx, wyx, xzx, yzx, zzx, wzx, xwx, ywx, zwx, wwx, xxy, yxy, zxy, wxy, xyy, yyy, zyy, wyy, xzy, yzy, zzy, wzy, xwy, ywy, zwy, wwy, xxz, yxz, zxz, wxz, xyz, yyz, zyz, wyz, xzz, yzz, zzz, wzz, xwz, ywz, zwz, wwz, xxw, yxw, zxw, wxw, xyw, yyw, zyw, wyw, xzw, yzw, zzw, wzw, xww, yww, zww, www;
    double4 xxxx, yxxx, zxxx, wxxx, xyxx, yyxx, zyxx, wyxx, xzxx, yzxx, zzxx, wzxx, xwxx, ywxx, zwxx, wwxx, xxyx, yxyx, zxyx, wxyx, xyyx, yyyx, zyyx, wyyx, xzyx, yzyx, zzyx, wzyx, xwyx, ywyx, zwyx, wwyx, xxzx, yxzx, zxzx, wxzx, xyzx, yyzx, zyzx, wyzx, xzzx, yzzx, zzzx, wzzx, xwzx, ywzx, zwzx, wwzx, xxwx, yxwx, zxwx, wxwx, xywx, yywx, zywx, wywx, xzwx, yzwx, zzwx, wzwx, xwwx, ywwx, zwwx, wwwx, xxxy, yxxy, zxxy, wxxy, xyxy, yyxy, zyxy, wyxy, xzxy, yzxy, zzxy, wzxy, xwxy, ywxy, zwxy, wwxy, xxyy, yxyy, zxyy, wxyy, xyyy, yyyy, zyyy, wyyy, xzyy, yzyy, zzyy, wzyy, xwyy, ywyy, zwyy, wwyy, xxzy, yxzy, zxzy, wxzy, xyzy, yyzy, zyzy, wyzy, xzzy, yzzy, zzzy, wzzy, xwzy, ywzy, zwzy, wwzy, xxwy, yxwy, zxwy, wxwy, xywy, yywy, zywy, wywy, xzwy, yzwy, zzwy, wzwy, xwwy, ywwy, zwwy, wwwy, xxxz, yxxz, zxxz, wxxz, xyxz, yyxz, zyxz, wyxz, xzxz, yzxz, zzxz, wzxz, xwxz, ywxz, zwxz, wwxz, xxyz, yxyz, zxyz, wxyz, xyyz, yyyz, zyyz, wyyz, xzyz, yzyz, zzyz, wzyz, xwyz, ywyz, zwyz, wwyz, xxzz, yxzz, zxzz, wxzz, xyzz, yyzz, zyzz, wyzz, xzzz, yzzz, zzzz, wzzz, xwzz, ywzz, zwzz, wwzz, xxwz, yxwz, zxwz, wxwz, xywz, yywz, zywz, wywz, xzwz, yzwz, zzwz, wzwz, xwwz, ywwz, zwwz, wwwz, xxxw, yxxw, zxxw, wxxw, xyxw, yyxw, zyxw, wyxw, xzxw, yzxw, zzxw, wzxw, xwxw, ywxw, zwxw, wwxw, xxyw, yxyw, zxyw, wxyw, xyyw, yyyw, zyyw, wyyw, xzyw, yzyw, zzyw, wzyw, xwyw, ywyw, zwyw, wwyw, xxzw, yxzw, zxzw, wxzw, xyzw, yyzw, zyzw, wyzw, xzzw, yzzw, zzzw, wzzw, xwzw, ywzw, zwzw, wwzw, xxww, yxww, zxww, wxww, xyww, yyww, zyww, wyww, xzww, yzww, zzww, wzww, xwww, ywww, zwww, wwww;
};

class char8
{
public:
    char8(const char a);
    char x, y, z, w;
    char s0, s1, s2, s3, s4, s5, s6, s7;
    char2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    char3 xyz;
    char4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class uchar8
{
public:
    uchar8(const uchar a);
    uchar x, y, z, w;
    uchar s0, s1, s2, s3, s4, s5, s6, s7;
    uchar2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    uchar3 xyz;
    uchar4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class short8
{
public:
    short8(const short a);
    short x, y, z, w;
    short s0, s1, s2, s3, s4, s5, s6, s7;
    short2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    short3 xyz;
    short4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class ushort8
{
public:
    ushort8(const ushort a);
    ushort x, y, z, w;
    ushort s0, s1, s2, s3, s4, s5, s6, s7;
    ushort2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    ushort3 xyz;
    ushort4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class int8
{
public:
    int8(const int a);
    int x, y, z, w;
    int s0, s1, s2, s3, s4, s5, s6, s7;
    int2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    int3 xyz;
    int4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class uint8
{
public:
    uint8(const uint a);
    uint x, y, z, w;
    uint s0, s1, s2, s3, s4, s5, s6, s7;
    uint2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    uint3 xyz;
    uint4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class long8
{
public:
    long8(const long a);
    long x, y, z, w;
    long s0, s1, s2, s3, s4, s5, s6, s7;
    long2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    long3 xyz;
    long4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class ulong8
{
public:
    ulong8(const ulong a);
    ulong x, y, z, w;
    ulong s0, s1, s2, s3, s4, s5, s6, s7;
    ulong2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    ulong3 xyz;
    ulong4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class float8
{
public:
    float8(const float a);
    float x, y, z, w;
    float s0, s1, s2, s3, s4, s5, s6, s7;
    float2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    float3 xyz;
    float4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class double8
{
public:
    double8(const double a);
    double x, y, z, w;
    double s0, s1, s2, s3, s4, s5, s6, s7;
    double2 xy, zw, s01, s23, s45, s67, s04, s15, s26, s37;
    double3 xyz;
    double4 xyzw, lo, hi, even, odd, s0123, s4567;
};

class char16
{
public:
    char16(const char a);
    char x, y, z, w;
    char s0, s1, s2, s3, s4, s5, s6, s7;
    char s8, s9, sA, sB, sC, sD, sE, sF;
    char2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    char3 xyz;
    char4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    char8 lo, hi, even, odd;
};

class uchar16
{
public:
    uchar16(const uchar a);
    uchar x, y, z, w;
    uchar s0, s1, s2, s3, s4, s5, s6, s7;
    uchar s8, s9, sA, sB, sC, sD, sE, sF;
    uchar2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    uchar3 xyz;
    uchar4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    uchar8 lo, hi, even, odd;
};

class short16
{
public:
    short16(const short a);
    short x, y, z, w;
    short s0, s1, s2, s3, s4, s5, s6, s7;
    short s8, s9, sA, sB, sC, sD, sE, sF;
    short2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    short3 xyz;
    short4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    short8 lo, hi, even, odd;
};

class ushort16
{
public:
    ushort16(const ushort a);
    ushort x, y, z, w;
    ushort s0, s1, s2, s3, s4, s5, s6, s7;
    ushort s8, s9, sA, sB, sC, sD, sE, sF;
    ushort2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    ushort3 xyz;
    ushort4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    ushort8 lo, hi, even, odd;
};

class int16
{
public:
    int16(const int a);
    int x, y, z, w;
    int s0, s1, s2, s3, s4, s5, s6, s7;
    int s8, s9, sA, sB, sC, sD, sE, sF;
    int2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    int3 xyz;
    int4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    int8 lo, hi, even, odd;
};

class uint16
{
public:
    uint16(const uint a);
    uint x, y, z, w;
    uint s0, s1, s2, s3, s4, s5, s6, s7;
    uint s8, s9, sA, sB, sC, sD, sE, sF;
    uint2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    uint3 xyz;
    uint4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    uint8 lo, hi, even, odd;
};

class long16
{
public:
    long16(const long a);
    long x, y, z, w;
    long s0, s1, s2, s3, s4, s5, s6, s7;
    long s8, s9, sA, sB, sC, sD, sE, sF;
    long2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    long3 xyz;
    long4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    long8 lo, hi, even, odd;
};

class ulong16
{
public:
    ulong16(const ulong a);
    ulong x, y, z, w;
    ulong s0, s1, s2, s3, s4, s5, s6, s7;
    ulong s8, s9, sA, sB, sC, sD, sE, sF;
    ulong2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    ulong3 xyz;
    ulong4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    ulong8 lo, hi, even, odd;
};

class float16
{
public:
    float16(const float a);
    float x, y, z, w;
    float s0, s1, s2, s3, s4, s5, s6, s7;
    float s8, s9, sA, sB, sC, sD, sE, sF;
    float2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    float3 xyz;
    float4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    float8 lo, hi, even, odd;
};

class double16
{
public:
    double16(const double a);
    double x, y, z, w;
    double s0, s1, s2, s3, s4, s5, s6, s7;
    double s8, s9, sA, sB, sC, sD, sE, sF;
    double2 xy, zw, s01, s23, s45, s67, s89, sAB, sCD, sEF;
    double3 xyz;
    double4 xyzw, s0123, s4567, s89AB, sCDEF, s048C, s159D, s26AE, s37BF;
    double8 lo, hi, even, odd;
};

class sampler_t
{
public:
    sampler_t& operator=(const int value);
};

class event_t {};
class image1d_t {};
class image1d_array_t {};
class image1d_buffer_t {};
class image2d_t {};
class image2d_array_t {};
class image3d_t {};

size_t get_global_id(uint dim);
size_t get_global_size(uint dim);
size_t get_global_offset(uint dim);
size_t get_group_id(uint dim);
size_t get_local_id(uint dim);
size_t get_local_size(uint dim);
size_t get_num_groups(uint dim);
uint get_work_dim();

void barrier(int flags);
void mem_fence(int flags);
void read_mem_fence(int flags);
void write_mem_fence(int flags);

void wait_group_events(int num_events, event_t* event_list);

int printf(const char* format, ...); 

void prefetch(const char* ptr, size_t elements);
void prefetch(const char2* ptr, size_t elements);
void prefetch(const char3* ptr, size_t elements);
void prefetch(const char4* ptr, size_t elements);
void prefetch(const char8* ptr, size_t elements);
void prefetch(const char16* ptr, size_t elements);

void prefetch(const uchar* ptr, size_t elements);
void prefetch(const uchar2* ptr, size_t elements);
void prefetch(const uchar3* ptr, size_t elements);
void prefetch(const uchar4* ptr, size_t elements);
void prefetch(const uchar8* ptr, size_t elements);
void prefetch(const uchar16* ptr, size_t elements);

void prefetch(const short* ptr, size_t elements);
void prefetch(const short2* ptr, size_t elements);
void prefetch(const short3* ptr, size_t elements);
void prefetch(const short4* ptr, size_t elements);
void prefetch(const short8* ptr, size_t elements);
void prefetch(const short16* ptr, size_t elements);

void prefetch(const ushort* ptr, size_t elements);
void prefetch(const ushort2* ptr, size_t elements);
void prefetch(const ushort3* ptr, size_t elements);
void prefetch(const ushort4* ptr, size_t elements);
void prefetch(const ushort8* ptr, size_t elements);
void prefetch(const ushort16* ptr, size_t elements);

void prefetch(const int* ptr, size_t elements);
void prefetch(const int2* ptr, size_t elements);
void prefetch(const int3* ptr, size_t elements);
void prefetch(const int4* ptr, size_t elements);
void prefetch(const int8* ptr, size_t elements);
void prefetch(const int16* ptr, size_t elements);

void prefetch(const uint* ptr, size_t elements);
void prefetch(const uint2* ptr, size_t elements);
void prefetch(const uint3* ptr, size_t elements);
void prefetch(const uint4* ptr, size_t elements);
void prefetch(const uint8* ptr, size_t elements);
void prefetch(const uint16* ptr, size_t elements);

void prefetch(const long* ptr, size_t elements);
void prefetch(const long2* ptr, size_t elements);
void prefetch(const long3* ptr, size_t elements);
void prefetch(const long4* ptr, size_t elements);
void prefetch(const long8* ptr, size_t elements);
void prefetch(const long16* ptr, size_t elements);

void prefetch(const ulong* ptr, size_t elements);
void prefetch(const ulong2* ptr, size_t elements);
void prefetch(const ulong3* ptr, size_t elements);
void prefetch(const ulong4* ptr, size_t elements);
void prefetch(const ulong8* ptr, size_t elements);
void prefetch(const ulong16* ptr, size_t elements);

void prefetch(const float* ptr, size_t elements);
void prefetch(const float2* ptr, size_t elements);
void prefetch(const float3* ptr, size_t elements);
void prefetch(const float4* ptr, size_t elements);
void prefetch(const float8* ptr, size_t elements);
void prefetch(const float16* ptr, size_t elements);

void prefetch(const double* ptr, size_t elements);
void prefetch(const double2* ptr, size_t elements);
void prefetch(const double3* ptr, size_t elements);
void prefetch(const double4* ptr, size_t elements);
void prefetch(const double8* ptr, size_t elements);
void prefetch(const double16* ptr, size_t elements);

event_t async_work_group_copy(char* dst, const char* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(char* dst, const char* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(char2* dst, const char2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(char2* dst, const char2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(char3* dst, const char3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(char3* dst, const char3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(char4* dst, const char4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(char4* dst, const char4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(char8* dst, const char8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(char8* dst, const char8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(char16* dst, const char16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(char16* dst, const char16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(uchar* dst, const uchar* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uchar* dst, const uchar* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uchar2* dst, const uchar2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uchar2* dst, const uchar2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uchar3* dst, const uchar3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uchar3* dst, const uchar3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uchar4* dst, const uchar4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uchar4* dst, const uchar4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uchar8* dst, const uchar8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uchar8* dst, const uchar8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uchar16* dst, const uchar16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uchar16* dst, const uchar16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(short* dst, const short* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(short* dst, const short* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(short2* dst, const short2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(short2* dst, const short2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(short3* dst, const short3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(short3* dst, const short3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(short4* dst, const short4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(short4* dst, const short4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(short8* dst, const short8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(short8* dst, const short8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(short16* dst, const short16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(short16* dst, const short16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(ushort* dst, const ushort* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ushort* dst, const ushort* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ushort2* dst, const ushort2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ushort2* dst, const ushort2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ushort3* dst, const ushort3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ushort3* dst, const ushort3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ushort4* dst, const ushort4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ushort4* dst, const ushort4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ushort8* dst, const ushort8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ushort8* dst, const ushort8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ushort16* dst, const ushort16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ushort16* dst, const ushort16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(int* dst, const int* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(int* dst, const int* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(int2* dst, const int2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(int2* dst, const int2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(int3* dst, const int3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(int3* dst, const int3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(int4* dst, const int4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(int4* dst, const int4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(int8* dst, const int8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(int8* dst, const int8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(int16* dst, const int16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(int16* dst, const int16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(uint* dst, const uint* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uint* dst, const uint* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uint2* dst, const uint2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uint2* dst, const uint2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uint3* dst, const uint3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uint3* dst, const uint3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uint4* dst, const uint4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uint4* dst, const uint4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uint8* dst, const uint8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uint8* dst, const uint8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(uint16* dst, const uint16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(uint16* dst, const uint16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(long* dst, const long* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(long* dst, const long* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(long2* dst, const long2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(long2* dst, const long2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(long3* dst, const long3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(long3* dst, const long3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(long4* dst, const long4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(long4* dst, const long4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(long8* dst, const long8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(long8* dst, const long8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(long16* dst, const long16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(long16* dst, const long16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(ulong* dst, const ulong* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ulong* dst, const ulong* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ulong2* dst, const ulong2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ulong2* dst, const ulong2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ulong3* dst, const ulong3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ulong3* dst, const ulong3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ulong4* dst, const ulong4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ulong4* dst, const ulong4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ulong8* dst, const ulong8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ulong8* dst, const ulong8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(ulong16* dst, const ulong16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(ulong16* dst, const ulong16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(float* dst, const float* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(float* dst, const float* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(float2* dst, const float2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(float2* dst, const float2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(float3* dst, const float3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(float3* dst, const float3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(float4* dst, const float4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(float4* dst, const float4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(float8* dst, const float8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(float8* dst, const float8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(float16* dst, const float16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(float16* dst, const float16* src, size_t num_gentypes, size_t src_stride, event_t event);

event_t async_work_group_copy(double* dst, const double* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(double* dst, const double* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(double2* dst, const double2* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(double2* dst, const double2* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(double3* dst, const double3* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(double3* dst, const double3* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(double4* dst, const double4* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(double4* dst, const double4* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(double8* dst, const double8* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(double8* dst, const double8* src, size_t num_gentypes, size_t src_stride, event_t event);
event_t async_work_group_copy(double16* dst, const double16* src, size_t num_gentypes, event_t event);
event_t async_work_group_strided_copy(double16* dst, const double16* src, size_t num_gentypes, size_t src_stride, event_t event);

float4 read_imagef(image1d_t image, sampler_t sampler, int coord);
float4 read_imagef(image1d_t image, sampler_t sampler, float coord);
float4 read_imagef(image1d_t image, int coord);
float4 read_imagef(image1d_buffer_t image, int coord);
float4 read_imagef(image1d_array_t image, int2 coord);
float4 read_imagef(image1d_array_t image, sampler_t sampler, float4 coord);

float4 read_imagef(image2d_t image, sampler_t sampler, int2 coord);
float4 read_imagef(image2d_t image, sampler_t sampler, float2 coord);
float4 read_imagef(image2d_t image, int2 coord);
float4 read_imagef(image2d_array_t image, int4 coord);
float4 read_imagef(image2d_array_t image, sampler_t sampler, int4 coord);
float4 read_imagef(image2d_array_t image, sampler_t sampler, float4 coord);

float4 read_imagef(image3d_t image, sampler_t sampler, int4 coord);
float4 read_imagef(image3d_t image, sampler_t sampler, float4 coord);
float4 read_imagef(image3d_t image, int4 coord);

void write_imagef(image1d_t image, int coord, float4 color);
void write_imagef(image1d_buffer_t image, int coord, float4 color);
void write_imagef(image1d_array_t image, int2 coord, float4 color);
void write_imagei(image1d_t image, int coord, int4 color);
void write_imagei(image1d_buffer_t image, int coord, int4 color);
void write_imagei(image1d_array_t image, int2 coord, int4 color);
void write_imageui(image1d_t image, int coord, uint4 color);
void write_imageui(image1d_buffer_t image, int coord, uint4 color);
void write_imageui(image1d_array_t image, int2 coord, uint4 color);

void write_imagef(image2d_t image, int2 coord, float4 color);
void write_imagef(image2d_array_t image, int4 coord, float4 color);
void write_imagei(image2d_t image, int2 coord, int4 color);
void write_imagei(image2d_array_t image, int4 coord, int4 color);
void write_imageui(image2d_t image, int2 coord, uint4 color);
void write_imageui(image2d_array_t image, int4 coord, uint4 color);

void write_imagef(image3d_t image, int4 coord, float4 color);
void write_imagei(image3d_t image, int4 coord, int4 color);
void write_imageui(image3d_t image, int4 coord, uint4 color);

int get_image_width(image1d_t image);
int get_image_width(image1d_buffer_t image);
int get_image_width(image2d_t image);
int get_image_width(image3d_t image);
int get_image_width(image1d_array_t image);
int get_image_width(image2d_array_t image);

int get_image_height(image2d_t image);
int get_image_height(image3d_t image);
int get_image_height(image2d_array_t image);

int get_image_depth(image3d_t image);

int get_image_channel_data_type(image1d_t image);
int get_image_channel_data_type(image1d_buffer_t image);
int get_image_channel_data_type(image2d_t image);
int get_image_channel_data_type(image3d_t image);
int get_image_channel_data_type(image1d_array_t image);
int get_image_channel_data_type(image2d_array_t image);

int get_image_channel_order(image1d_t image);
int get_image_channel_order(image1d_buffer_t image);
int get_image_channel_order(image2d_t image);
int get_image_channel_order(image3d_t image);
int get_image_channel_order(image1d_array_t image);
int get_image_channel_order(image2d_array_t image);

int2 get_image_dim(image2d_t image);
int2 get_image_dim(image2d_array_t image);
int4 get_image_dim(image3d_t image);

char2 vload2(size_t offset, char* ptr);
void vstore2(const char2& data, size_t offset, char* ptr);
char3 vload3(size_t offset, char* ptr);
void vstore3(const char3& data, size_t offset, char* ptr);
char4 vload4(size_t offset, char* ptr);
void vstore4(const char4& data, size_t offset, char* ptr);
char8 vload8(size_t offset, char* ptr);
void vstore8(const char8& data, size_t offset, char* ptr);
char16 vload16(size_t offset, char* ptr);
void vstore16(const char16& data, size_t offset, char* ptr);

uchar2 vload2(size_t offset, uchar* ptr);
void vstore2(const uchar2& data, size_t offset, uchar* ptr);
uchar3 vload3(size_t offset, uchar* ptr);
void vstore3(const uchar3& data, size_t offset, uchar* ptr);
uchar4 vload4(size_t offset, uchar* ptr);
void vstore4(const uchar4& data, size_t offset, uchar* ptr);
uchar8 vload8(size_t offset, uchar* ptr);
void vstore8(const uchar8& data, size_t offset, uchar* ptr);
uchar16 vload16(size_t offset, uchar* ptr);
void vstore16(const uchar16& data, size_t offset, uchar* ptr);

short2 vload2(size_t offset, short* ptr);
void vstore2(const short2& data, size_t offset, short* ptr);
short3 vload3(size_t offset, short* ptr);
void vstore3(const short3& data, size_t offset, short* ptr);
short4 vload4(size_t offset, short* ptr);
void vstore4(const short4& data, size_t offset, short* ptr);
short8 vload8(size_t offset, short* ptr);
void vstore8(const short8& data, size_t offset, short* ptr);
short16 vload16(size_t offset, short* ptr);
void vstore16(const short16& data, size_t offset, short* ptr);

ushort2 vload2(size_t offset, ushort* ptr);
void vstore2(const ushort2& data, size_t offset, ushort* ptr);
ushort3 vload3(size_t offset, ushort* ptr);
void vstore3(const ushort3& data, size_t offset, ushort* ptr);
ushort4 vload4(size_t offset, ushort* ptr);
void vstore4(const ushort4& data, size_t offset, ushort* ptr);
ushort8 vload8(size_t offset, ushort* ptr);
void vstore8(const ushort8& data, size_t offset, ushort* ptr);
ushort16 vload16(size_t offset, ushort* ptr);
void vstore16(const ushort16& data, size_t offset, ushort* ptr);

int2 vload2(size_t offset, int* ptr);
void vstore2(const int2& data, size_t offset, int* ptr);
int3 vload3(size_t offset, int* ptr);
void vstore3(const int3& data, size_t offset, int* ptr);
int4 vload4(size_t offset, int* ptr);
void vstore4(const int4& data, size_t offset, int* ptr);
int8 vload8(size_t offset, int* ptr);
void vstore8(const int8& data, size_t offset, int* ptr);
int16 vload16(size_t offset, int* ptr);
void vstore16(const int16& data, size_t offset, int* ptr);

uint2 vload2(size_t offset, uint* ptr);
void vstore2(const uint2& data, size_t offset, uint* ptr);
uint3 vload3(size_t offset, uint* ptr);
void vstore3(const uint3& data, size_t offset, uint* ptr);
uint4 vload4(size_t offset, uint* ptr);
void vstore4(const uint4& data, size_t offset, uint* ptr);
uint8 vload8(size_t offset, uint* ptr);
void vstore8(const uint8& data, size_t offset, uint* ptr);
uint16 vload16(size_t offset, uint* ptr);
void vstore16(const uint16& data, size_t offset, uint* ptr);

long2 vload2(size_t offset, long* ptr);
void vstore2(const long2& data, size_t offset, long* ptr);
long3 vload3(size_t offset, long* ptr);
void vstore3(const long3& data, size_t offset, long* ptr);
long4 vload4(size_t offset, long* ptr);
void vstore4(const long4& data, size_t offset, long* ptr);
long8 vload8(size_t offset, long* ptr);
void vstore8(const long8& data, size_t offset, long* ptr);
long16 vload16(size_t offset, long* ptr);
void vstore16(const long16& data, size_t offset, long* ptr);

ulong2 vload2(size_t offset, ulong* ptr);
void vstore2(const ulong2& data, size_t offset, ulong* ptr);
ulong3 vload3(size_t offset, ulong* ptr);
void vstore3(const ulong3& data, size_t offset, ulong* ptr);
ulong4 vload4(size_t offset, ulong* ptr);
void vstore4(const ulong4& data, size_t offset, ulong* ptr);
ulong8 vload8(size_t offset, ulong* ptr);
void vstore8(const ulong8& data, size_t offset, ulong* ptr);
ulong16 vload16(size_t offset, ulong* ptr);
void vstore16(const ulong16& data, size_t offset, ulong* ptr);

float2 vload2(size_t offset, float* ptr);
void vstore2(const float2& data, size_t offset, float* ptr);
float3 vload3(size_t offset, float* ptr);
void vstore3(const float3& data, size_t offset, float* ptr);
float4 vload4(size_t offset, float* ptr);
void vstore4(const float4& data, size_t offset, float* ptr);
float8 vload8(size_t offset, float* ptr);
void vstore8(const float8& data, size_t offset, float* ptr);
float16 vload16(size_t offset, float* ptr);
void vstore16(const float16& data, size_t offset, float* ptr);

double2 vload2(size_t offset, double* ptr);
void vstore2(const double2& data, size_t offset, double* ptr);
double3 vload3(size_t offset, double* ptr);
void vstore3(const double3& data, size_t offset, double* ptr);
double4 vload4(size_t offset, double* ptr);
void vstore4(const double4& data, size_t offset, double* ptr);
double8 vload8(size_t offset, double* ptr);
void vstore8(const double8& data, size_t offset, double* ptr);
double16 vload16(size_t offset, double* ptr);
void vstore16(const double16& data, size_t offset, double* ptr);

float3 cross(const float3& p0, const float3& p1);
float4 cross(const float4& p0, const float4& p1);
double3 cross(const double3& p0, const double3& p1);
double4 cross(const double4& p0, const double4& p1);

float dot(const float& p0, const float& p1);
float distance(const float& p0, const float& p1);
float length(const float& p);
float normalize(const float& p);
float fast_distance(const float& p0, const float& p1);
float fast_length(const float& p);
float fast_normalize(const float& p);

float dot(const float2& p0, const float2& p1);
float distance(const float2& p0, const float2& p1);
float length(const float2& p);
float2 normalize(const float2& p);
float fast_distance(const float2& p0, const float2& p1);
float fast_length(const float2& p);
float2 fast_normalize(const float2& p);

float dot(const float3& p0, const float3& p1);
float distance(const float3& p0, const float3& p1);
float length(const float3& p);
float3 normalize(const float3& p);
float fast_distance(const float3& p0, const float3& p1);
float fast_length(const float3& p);
float3 fast_normalize(const float3& p);

float dot(const float4& p0, const float4& p1);
float distance(const float4& p0, const float4& p1);
float length(const float4& p);
float4 normalize(const float4& p);
float fast_distance(const float4& p0, const float4& p1);
float fast_length(const float4& p);
float4 fast_normalize(const float4& p);

float dot(const float8& p0, const float8& p1);
float distance(const float8& p0, const float8& p1);
float length(const float8& p);
float8 normalize(const float8& p);
float fast_distance(const float8& p0, const float8& p1);
float fast_length(const float8& p);
float8 fast_normalize(const float8& p);

float dot(const float16& p0, const float16& p1);
float distance(const float16& p0, const float16& p1);
float length(const float16& p);
float16 normalize(const float16& p);
float fast_distance(const float16& p0, const float16& p1);
float fast_length(const float16& p);
float16 fast_normalize(const float16& p);

double dot(const double& p0, const double& p1);
double distance(const double& p0, const double& p1);
double length(const double& p);
double normalize(const double& p);
double fast_distance(const double& p0, const double& p1);
double fast_length(const double& p);
double fast_normalize(const double& p);

double dot(const double2& p0, const double2& p1);
double distance(const double2& p0, const double2& p1);
double length(const double2& p);
double2 normalize(const double2& p);
double fast_distance(const double2& p0, const double2& p1);
double fast_length(const double2& p);
double2 fast_normalize(const double2& p);

double dot(const double3& p0, const double3& p1);
double distance(const double3& p0, const double3& p1);
double length(const double3& p);
double3 normalize(const double3& p);
double fast_distance(const double3& p0, const double3& p1);
double fast_length(const double3& p);
double3 fast_normalize(const double3& p);

double dot(const double4& p0, const double4& p1);
double distance(const double4& p0, const double4& p1);
double length(const double4& p);
double4 normalize(const double4& p);
double fast_distance(const double4& p0, const double4& p1);
double fast_length(const double4& p);
double4 fast_normalize(const double4& p);

double dot(const double8& p0, const double8& p1);
double distance(const double8& p0, const double8& p1);
double length(const double8& p);
double8 normalize(const double8& p);
double fast_distance(const double8& p0, const double8& p1);
double fast_length(const double8& p);
double8 fast_normalize(const double8& p);

double dot(const double16& p0, const double16& p1);
double distance(const double16& p0, const double16& p1);
double length(const double16& p);
double16 normalize(const double16& p);
double fast_distance(const double16& p0, const double16& p1);
double fast_length(const double16& p);
double16 fast_normalize(const double16& p);

uchar abs(const char& x);
uchar abs_diff(const char& x, const char& y);
uchar2 abs(const char2& x);
uchar2 abs_diff(const char2& x, const char2& y);
uchar3 abs(const char3& x);
uchar3 abs_diff(const char3& x, const char3& y);
uchar4 abs(const char4& x);
uchar4 abs_diff(const char4& x, const char4& y);
uchar8 abs(const char8& x);
uchar8 abs_diff(const char8& x, const char8& y);
uchar16 abs(const char16& x);
uchar16 abs_diff(const char16& x, const char16& y);

uchar abs(const uchar& x);
uchar abs_diff(const uchar& x, const uchar& y);
uchar2 abs(const uchar2& x);
uchar2 abs_diff(const uchar2& x, const uchar2& y);
uchar3 abs(const uchar3& x);
uchar3 abs_diff(const uchar3& x, const uchar3& y);
uchar4 abs(const uchar4& x);
uchar4 abs_diff(const uchar4& x, const uchar4& y);
uchar8 abs(const uchar8& x);
uchar8 abs_diff(const uchar8& x, const uchar8& y);
uchar16 abs(const uchar16& x);
uchar16 abs_diff(const uchar16& x, const uchar16& y);

ushort abs(const short& x);
ushort abs_diff(const short& x, const short& y);
ushort2 abs(const short2& x);
ushort2 abs_diff(const short2& x, const short2& y);
ushort3 abs(const short3& x);
ushort3 abs_diff(const short3& x, const short3& y);
ushort4 abs(const short4& x);
ushort4 abs_diff(const short4& x, const short4& y);
ushort8 abs(const short8& x);
ushort8 abs_diff(const short8& x, const short8& y);
ushort16 abs(const short16& x);
ushort16 abs_diff(const short16& x, const short16& y);

ushort abs(const ushort& x);
ushort abs_diff(const ushort& x, const ushort& y);
ushort2 abs(const ushort2& x);
ushort2 abs_diff(const ushort2& x, const ushort2& y);
ushort3 abs(const ushort3& x);
ushort3 abs_diff(const ushort3& x, const ushort3& y);
ushort4 abs(const ushort4& x);
ushort4 abs_diff(const ushort4& x, const ushort4& y);
ushort8 abs(const ushort8& x);
ushort8 abs_diff(const ushort8& x, const ushort8& y);
ushort16 abs(const ushort16& x);
ushort16 abs_diff(const ushort16& x, const ushort16& y);

uint abs(const int& x);
uint abs_diff(const int& x, const int& y);
uint2 abs(const int2& x);
uint2 abs_diff(const int2& x, const int2& y);
uint3 abs(const int3& x);
uint3 abs_diff(const int3& x, const int3& y);
uint4 abs(const int4& x);
uint4 abs_diff(const int4& x, const int4& y);
uint8 abs(const int8& x);
uint8 abs_diff(const int8& x, const int8& y);
uint16 abs(const int16& x);
uint16 abs_diff(const int16& x, const int16& y);

uint abs(const uint& x);
uint abs_diff(const uint& x, const uint& y);
uint2 abs(const uint2& x);
uint2 abs_diff(const uint2& x, const uint2& y);
uint3 abs(const uint3& x);
uint3 abs_diff(const uint3& x, const uint3& y);
uint4 abs(const uint4& x);
uint4 abs_diff(const uint4& x, const uint4& y);
uint8 abs(const uint8& x);
uint8 abs_diff(const uint8& x, const uint8& y);
uint16 abs(const uint16& x);
uint16 abs_diff(const uint16& x, const uint16& y);

ulong abs(const long& x);
ulong abs_diff(const long& x, const long& y);
ulong2 abs(const long2& x);
ulong2 abs_diff(const long2& x, const long2& y);
ulong3 abs(const long3& x);
ulong3 abs_diff(const long3& x, const long3& y);
ulong4 abs(const long4& x);
ulong4 abs_diff(const long4& x, const long4& y);
ulong8 abs(const long8& x);
ulong8 abs_diff(const long8& x, const long8& y);
ulong16 abs(const long16& x);
ulong16 abs_diff(const long16& x, const long16& y);

ulong abs(const ulong& x);
ulong abs_diff(const ulong& x, const ulong& y);
ulong2 abs(const ulong2& x);
ulong2 abs_diff(const ulong2& x, const ulong2& y);
ulong3 abs(const ulong3& x);
ulong3 abs_diff(const ulong3& x, const ulong3& y);
ulong4 abs(const ulong4& x);
ulong4 abs_diff(const ulong4& x, const ulong4& y);
ulong8 abs(const ulong8& x);
ulong8 abs_diff(const ulong8& x, const ulong8& y);
ulong16 abs(const ulong16& x);
ulong16 abs_diff(const ulong16& x, const ulong16& y);

char clamp(const char& x, const char& minval, const char& maxval);
char clz(const char& x);
char hadd(const char& x, const char& y);
char rhadd(const char& x, const char& y);
char mad24(const char& x, const char& y, const char& z);
char mad_hi(const char& a, const char& b, const char& c);
char mad_sat(const char& a, const char& b, const char& c);
char max(const char& x, const char& y);
char min(const char& x, const char& y);
char mul24(const char& x, const char& y);
char mul_hi(const char& x, const char& y);
char popcount(const char& x);
char rotate(const char& v, const char& i);
char sub_sat(const char& x, const char& y);

char2 clamp(const char2& x, const char2& minval, const char2& maxval);
char2 clz(const char2& x);
char2 hadd(const char2& x, const char2& y);
char2 rhadd(const char2& x, const char2& y);
char2 mad24(const char2& x, const char2& y, const char2& z);
char2 mad_hi(const char2& a, const char2& b, const char2& c);
char2 mad_sat(const char2& a, const char2& b, const char2& c);
char2 max(const char2& x, const char2& y);
char2 min(const char2& x, const char2& y);
char2 mul24(const char2& x, const char2& y);
char2 mul_hi(const char2& x, const char2& y);
char2 popcount(const char2& x);
char2 rotate(const char2& v, const char2& i);
char2 sub_sat(const char2& x, const char2& y);

char3 clamp(const char3& x, const char3& minval, const char3& maxval);
char3 clz(const char3& x);
char3 hadd(const char3& x, const char3& y);
char3 rhadd(const char3& x, const char3& y);
char3 mad24(const char3& x, const char3& y, const char3& z);
char3 mad_hi(const char3& a, const char3& b, const char3& c);
char3 mad_sat(const char3& a, const char3& b, const char3& c);
char3 max(const char3& x, const char3& y);
char3 min(const char3& x, const char3& y);
char3 mul24(const char3& x, const char3& y);
char3 mul_hi(const char3& x, const char3& y);
char3 popcount(const char3& x);
char3 rotate(const char3& v, const char3& i);
char3 sub_sat(const char3& x, const char3& y);

char4 clamp(const char4& x, const char4& minval, const char4& maxval);
char4 clz(const char4& x);
char4 hadd(const char4& x, const char4& y);
char4 rhadd(const char4& x, const char4& y);
char4 mad24(const char4& x, const char4& y, const char4& z);
char4 mad_hi(const char4& a, const char4& b, const char4& c);
char4 mad_sat(const char4& a, const char4& b, const char4& c);
char4 max(const char4& x, const char4& y);
char4 min(const char4& x, const char4& y);
char4 mul24(const char4& x, const char4& y);
char4 mul_hi(const char4& x, const char4& y);
char4 popcount(const char4& x);
char4 rotate(const char4& v, const char4& i);
char4 sub_sat(const char4& x, const char4& y);

char8 clamp(const char8& x, const char8& minval, const char8& maxval);
char8 clz(const char8& x);
char8 hadd(const char8& x, const char8& y);
char8 rhadd(const char8& x, const char8& y);
char8 mad24(const char8& x, const char8& y, const char8& z);
char8 mad_hi(const char8& a, const char8& b, const char8& c);
char8 mad_sat(const char8& a, const char8& b, const char8& c);
char8 max(const char8& x, const char8& y);
char8 min(const char8& x, const char8& y);
char8 mul24(const char8& x, const char8& y);
char8 mul_hi(const char8& x, const char8& y);
char8 popcount(const char8& x);
char8 rotate(const char8& v, const char8& i);
char8 sub_sat(const char8& x, const char8& y);

char16 clamp(const char16& x, const char16& minval, const char16& maxval);
char16 clz(const char16& x);
char16 hadd(const char16& x, const char16& y);
char16 rhadd(const char16& x, const char16& y);
char16 mad24(const char16& x, const char16& y, const char16& z);
char16 mad_hi(const char16& a, const char16& b, const char16& c);
char16 mad_sat(const char16& a, const char16& b, const char16& c);
char16 max(const char16& x, const char16& y);
char16 min(const char16& x, const char16& y);
char16 mul24(const char16& x, const char16& y);
char16 mul_hi(const char16& x, const char16& y);
char16 popcount(const char16& x);
char16 rotate(const char16& v, const char16& i);
char16 sub_sat(const char16& x, const char16& y);

uchar clamp(const uchar& x, const uchar& minval, const uchar& maxval);
uchar clz(const uchar& x);
uchar hadd(const uchar& x, const uchar& y);
uchar rhadd(const uchar& x, const uchar& y);
uchar mad24(const uchar& x, const uchar& y, const uchar& z);
uchar mad_hi(const uchar& a, const uchar& b, const uchar& c);
uchar mad_sat(const uchar& a, const uchar& b, const uchar& c);
uchar max(const uchar& x, const uchar& y);
uchar min(const uchar& x, const uchar& y);
uchar mul24(const uchar& x, const uchar& y);
uchar mul_hi(const uchar& x, const uchar& y);
uchar popcount(const uchar& x);
uchar rotate(const uchar& v, const uchar& i);
uchar sub_sat(const uchar& x, const uchar& y);

uchar2 clamp(const uchar2& x, const uchar2& minval, const uchar2& maxval);
uchar2 clz(const uchar2& x);
uchar2 hadd(const uchar2& x, const uchar2& y);
uchar2 rhadd(const uchar2& x, const uchar2& y);
uchar2 mad24(const uchar2& x, const uchar2& y, const uchar2& z);
uchar2 mad_hi(const uchar2& a, const uchar2& b, const uchar2& c);
uchar2 mad_sat(const uchar2& a, const uchar2& b, const uchar2& c);
uchar2 max(const uchar2& x, const uchar2& y);
uchar2 min(const uchar2& x, const uchar2& y);
uchar2 mul24(const uchar2& x, const uchar2& y);
uchar2 mul_hi(const uchar2& x, const uchar2& y);
uchar2 popcount(const uchar2& x);
uchar2 rotate(const uchar2& v, const uchar2& i);
uchar2 sub_sat(const uchar2& x, const uchar2& y);

uchar3 clamp(const uchar3& x, const uchar3& minval, const uchar3& maxval);
uchar3 clz(const uchar3& x);
uchar3 hadd(const uchar3& x, const uchar3& y);
uchar3 rhadd(const uchar3& x, const uchar3& y);
uchar3 mad24(const uchar3& x, const uchar3& y, const uchar3& z);
uchar3 mad_hi(const uchar3& a, const uchar3& b, const uchar3& c);
uchar3 mad_sat(const uchar3& a, const uchar3& b, const uchar3& c);
uchar3 max(const uchar3& x, const uchar3& y);
uchar3 min(const uchar3& x, const uchar3& y);
uchar3 mul24(const uchar3& x, const uchar3& y);
uchar3 mul_hi(const uchar3& x, const uchar3& y);
uchar3 popcount(const uchar3& x);
uchar3 rotate(const uchar3& v, const uchar3& i);
uchar3 sub_sat(const uchar3& x, const uchar3& y);

uchar4 clamp(const uchar4& x, const uchar4& minval, const uchar4& maxval);
uchar4 clz(const uchar4& x);
uchar4 hadd(const uchar4& x, const uchar4& y);
uchar4 rhadd(const uchar4& x, const uchar4& y);
uchar4 mad24(const uchar4& x, const uchar4& y, const uchar4& z);
uchar4 mad_hi(const uchar4& a, const uchar4& b, const uchar4& c);
uchar4 mad_sat(const uchar4& a, const uchar4& b, const uchar4& c);
uchar4 max(const uchar4& x, const uchar4& y);
uchar4 min(const uchar4& x, const uchar4& y);
uchar4 mul24(const uchar4& x, const uchar4& y);
uchar4 mul_hi(const uchar4& x, const uchar4& y);
uchar4 popcount(const uchar4& x);
uchar4 rotate(const uchar4& v, const uchar4& i);
uchar4 sub_sat(const uchar4& x, const uchar4& y);

uchar8 clamp(const uchar8& x, const uchar8& minval, const uchar8& maxval);
uchar8 clz(const uchar8& x);
uchar8 hadd(const uchar8& x, const uchar8& y);
uchar8 rhadd(const uchar8& x, const uchar8& y);
uchar8 mad24(const uchar8& x, const uchar8& y, const uchar8& z);
uchar8 mad_hi(const uchar8& a, const uchar8& b, const uchar8& c);
uchar8 mad_sat(const uchar8& a, const uchar8& b, const uchar8& c);
uchar8 max(const uchar8& x, const uchar8& y);
uchar8 min(const uchar8& x, const uchar8& y);
uchar8 mul24(const uchar8& x, const uchar8& y);
uchar8 mul_hi(const uchar8& x, const uchar8& y);
uchar8 popcount(const uchar8& x);
uchar8 rotate(const uchar8& v, const uchar8& i);
uchar8 sub_sat(const uchar8& x, const uchar8& y);

uchar16 clamp(const uchar16& x, const uchar16& minval, const uchar16& maxval);
uchar16 clz(const uchar16& x);
uchar16 hadd(const uchar16& x, const uchar16& y);
uchar16 rhadd(const uchar16& x, const uchar16& y);
uchar16 mad24(const uchar16& x, const uchar16& y, const uchar16& z);
uchar16 mad_hi(const uchar16& a, const uchar16& b, const uchar16& c);
uchar16 mad_sat(const uchar16& a, const uchar16& b, const uchar16& c);
uchar16 max(const uchar16& x, const uchar16& y);
uchar16 min(const uchar16& x, const uchar16& y);
uchar16 mul24(const uchar16& x, const uchar16& y);
uchar16 mul_hi(const uchar16& x, const uchar16& y);
uchar16 popcount(const uchar16& x);
uchar16 rotate(const uchar16& v, const uchar16& i);
uchar16 sub_sat(const uchar16& x, const uchar16& y);

short clamp(const short& x, const short& minval, const short& maxval);
short clz(const short& x);
short hadd(const short& x, const short& y);
short rhadd(const short& x, const short& y);
short mad24(const short& x, const short& y, const short& z);
short mad_hi(const short& a, const short& b, const short& c);
short mad_sat(const short& a, const short& b, const short& c);
short max(const short& x, const short& y);
short min(const short& x, const short& y);
short mul24(const short& x, const short& y);
short mul_hi(const short& x, const short& y);
short popcount(const short& x);
short rotate(const short& v, const short& i);
short sub_sat(const short& x, const short& y);

short2 clamp(const short2& x, const short2& minval, const short2& maxval);
short2 clz(const short2& x);
short2 hadd(const short2& x, const short2& y);
short2 rhadd(const short2& x, const short2& y);
short2 mad24(const short2& x, const short2& y, const short2& z);
short2 mad_hi(const short2& a, const short2& b, const short2& c);
short2 mad_sat(const short2& a, const short2& b, const short2& c);
short2 max(const short2& x, const short2& y);
short2 min(const short2& x, const short2& y);
short2 mul24(const short2& x, const short2& y);
short2 mul_hi(const short2& x, const short2& y);
short2 popcount(const short2& x);
short2 rotate(const short2& v, const short2& i);
short2 sub_sat(const short2& x, const short2& y);

short3 clamp(const short3& x, const short3& minval, const short3& maxval);
short3 clz(const short3& x);
short3 hadd(const short3& x, const short3& y);
short3 rhadd(const short3& x, const short3& y);
short3 mad24(const short3& x, const short3& y, const short3& z);
short3 mad_hi(const short3& a, const short3& b, const short3& c);
short3 mad_sat(const short3& a, const short3& b, const short3& c);
short3 max(const short3& x, const short3& y);
short3 min(const short3& x, const short3& y);
short3 mul24(const short3& x, const short3& y);
short3 mul_hi(const short3& x, const short3& y);
short3 popcount(const short3& x);
short3 rotate(const short3& v, const short3& i);
short3 sub_sat(const short3& x, const short3& y);

short4 clamp(const short4& x, const short4& minval, const short4& maxval);
short4 clz(const short4& x);
short4 hadd(const short4& x, const short4& y);
short4 rhadd(const short4& x, const short4& y);
short4 mad24(const short4& x, const short4& y, const short4& z);
short4 mad_hi(const short4& a, const short4& b, const short4& c);
short4 mad_sat(const short4& a, const short4& b, const short4& c);
short4 max(const short4& x, const short4& y);
short4 min(const short4& x, const short4& y);
short4 mul24(const short4& x, const short4& y);
short4 mul_hi(const short4& x, const short4& y);
short4 popcount(const short4& x);
short4 rotate(const short4& v, const short4& i);
short4 sub_sat(const short4& x, const short4& y);

short8 clamp(const short8& x, const short8& minval, const short8& maxval);
short8 clz(const short8& x);
short8 hadd(const short8& x, const short8& y);
short8 rhadd(const short8& x, const short8& y);
short8 mad24(const short8& x, const short8& y, const short8& z);
short8 mad_hi(const short8& a, const short8& b, const short8& c);
short8 mad_sat(const short8& a, const short8& b, const short8& c);
short8 max(const short8& x, const short8& y);
short8 min(const short8& x, const short8& y);
short8 mul24(const short8& x, const short8& y);
short8 mul_hi(const short8& x, const short8& y);
short8 popcount(const short8& x);
short8 rotate(const short8& v, const short8& i);
short8 sub_sat(const short8& x, const short8& y);

short16 clamp(const short16& x, const short16& minval, const short16& maxval);
short16 clz(const short16& x);
short16 hadd(const short16& x, const short16& y);
short16 rhadd(const short16& x, const short16& y);
short16 mad24(const short16& x, const short16& y, const short16& z);
short16 mad_hi(const short16& a, const short16& b, const short16& c);
short16 mad_sat(const short16& a, const short16& b, const short16& c);
short16 max(const short16& x, const short16& y);
short16 min(const short16& x, const short16& y);
short16 mul24(const short16& x, const short16& y);
short16 mul_hi(const short16& x, const short16& y);
short16 popcount(const short16& x);
short16 rotate(const short16& v, const short16& i);
short16 sub_sat(const short16& x, const short16& y);

ushort clamp(const ushort& x, const ushort& minval, const ushort& maxval);
ushort clz(const ushort& x);
ushort hadd(const ushort& x, const ushort& y);
ushort rhadd(const ushort& x, const ushort& y);
ushort mad24(const ushort& x, const ushort& y, const ushort& z);
ushort mad_hi(const ushort& a, const ushort& b, const ushort& c);
ushort mad_sat(const ushort& a, const ushort& b, const ushort& c);
ushort max(const ushort& x, const ushort& y);
ushort min(const ushort& x, const ushort& y);
ushort mul24(const ushort& x, const ushort& y);
ushort mul_hi(const ushort& x, const ushort& y);
ushort popcount(const ushort& x);
ushort rotate(const ushort& v, const ushort& i);
ushort sub_sat(const ushort& x, const ushort& y);

ushort2 clamp(const ushort2& x, const ushort2& minval, const ushort2& maxval);
ushort2 clz(const ushort2& x);
ushort2 hadd(const ushort2& x, const ushort2& y);
ushort2 rhadd(const ushort2& x, const ushort2& y);
ushort2 mad24(const ushort2& x, const ushort2& y, const ushort2& z);
ushort2 mad_hi(const ushort2& a, const ushort2& b, const ushort2& c);
ushort2 mad_sat(const ushort2& a, const ushort2& b, const ushort2& c);
ushort2 max(const ushort2& x, const ushort2& y);
ushort2 min(const ushort2& x, const ushort2& y);
ushort2 mul24(const ushort2& x, const ushort2& y);
ushort2 mul_hi(const ushort2& x, const ushort2& y);
ushort2 popcount(const ushort2& x);
ushort2 rotate(const ushort2& v, const ushort2& i);
ushort2 sub_sat(const ushort2& x, const ushort2& y);

ushort3 clamp(const ushort3& x, const ushort3& minval, const ushort3& maxval);
ushort3 clz(const ushort3& x);
ushort3 hadd(const ushort3& x, const ushort3& y);
ushort3 rhadd(const ushort3& x, const ushort3& y);
ushort3 mad24(const ushort3& x, const ushort3& y, const ushort3& z);
ushort3 mad_hi(const ushort3& a, const ushort3& b, const ushort3& c);
ushort3 mad_sat(const ushort3& a, const ushort3& b, const ushort3& c);
ushort3 max(const ushort3& x, const ushort3& y);
ushort3 min(const ushort3& x, const ushort3& y);
ushort3 mul24(const ushort3& x, const ushort3& y);
ushort3 mul_hi(const ushort3& x, const ushort3& y);
ushort3 popcount(const ushort3& x);
ushort3 rotate(const ushort3& v, const ushort3& i);
ushort3 sub_sat(const ushort3& x, const ushort3& y);

ushort4 clamp(const ushort4& x, const ushort4& minval, const ushort4& maxval);
ushort4 clz(const ushort4& x);
ushort4 hadd(const ushort4& x, const ushort4& y);
ushort4 rhadd(const ushort4& x, const ushort4& y);
ushort4 mad24(const ushort4& x, const ushort4& y, const ushort4& z);
ushort4 mad_hi(const ushort4& a, const ushort4& b, const ushort4& c);
ushort4 mad_sat(const ushort4& a, const ushort4& b, const ushort4& c);
ushort4 max(const ushort4& x, const ushort4& y);
ushort4 min(const ushort4& x, const ushort4& y);
ushort4 mul24(const ushort4& x, const ushort4& y);
ushort4 mul_hi(const ushort4& x, const ushort4& y);
ushort4 popcount(const ushort4& x);
ushort4 rotate(const ushort4& v, const ushort4& i);
ushort4 sub_sat(const ushort4& x, const ushort4& y);

ushort8 clamp(const ushort8& x, const ushort8& minval, const ushort8& maxval);
ushort8 clz(const ushort8& x);
ushort8 hadd(const ushort8& x, const ushort8& y);
ushort8 rhadd(const ushort8& x, const ushort8& y);
ushort8 mad24(const ushort8& x, const ushort8& y, const ushort8& z);
ushort8 mad_hi(const ushort8& a, const ushort8& b, const ushort8& c);
ushort8 mad_sat(const ushort8& a, const ushort8& b, const ushort8& c);
ushort8 max(const ushort8& x, const ushort8& y);
ushort8 min(const ushort8& x, const ushort8& y);
ushort8 mul24(const ushort8& x, const ushort8& y);
ushort8 mul_hi(const ushort8& x, const ushort8& y);
ushort8 popcount(const ushort8& x);
ushort8 rotate(const ushort8& v, const ushort8& i);
ushort8 sub_sat(const ushort8& x, const ushort8& y);

ushort16 clamp(const ushort16& x, const ushort16& minval, const ushort16& maxval);
ushort16 clz(const ushort16& x);
ushort16 hadd(const ushort16& x, const ushort16& y);
ushort16 rhadd(const ushort16& x, const ushort16& y);
ushort16 mad24(const ushort16& x, const ushort16& y, const ushort16& z);
ushort16 mad_hi(const ushort16& a, const ushort16& b, const ushort16& c);
ushort16 mad_sat(const ushort16& a, const ushort16& b, const ushort16& c);
ushort16 max(const ushort16& x, const ushort16& y);
ushort16 min(const ushort16& x, const ushort16& y);
ushort16 mul24(const ushort16& x, const ushort16& y);
ushort16 mul_hi(const ushort16& x, const ushort16& y);
ushort16 popcount(const ushort16& x);
ushort16 rotate(const ushort16& v, const ushort16& i);
ushort16 sub_sat(const ushort16& x, const ushort16& y);

int clamp(const int& x, const int& minval, const int& maxval);
int clz(const int& x);
int hadd(const int& x, const int& y);
int rhadd(const int& x, const int& y);
int mad24(const int& x, const int& y, const int& z);
int mad_hi(const int& a, const int& b, const int& c);
int mad_sat(const int& a, const int& b, const int& c);
int max(const int& x, const int& y);
int min(const int& x, const int& y);
int mul24(const int& x, const int& y);
int mul_hi(const int& x, const int& y);
int popcount(const int& x);
int rotate(const int& v, const int& i);
int sub_sat(const int& x, const int& y);

int2 clamp(const int2& x, const int2& minval, const int2& maxval);
int2 clz(const int2& x);
int2 hadd(const int2& x, const int2& y);
int2 rhadd(const int2& x, const int2& y);
int2 mad24(const int2& x, const int2& y, const int2& z);
int2 mad_hi(const int2& a, const int2& b, const int2& c);
int2 mad_sat(const int2& a, const int2& b, const int2& c);
int2 max(const int2& x, const int2& y);
int2 min(const int2& x, const int2& y);
int2 mul24(const int2& x, const int2& y);
int2 mul_hi(const int2& x, const int2& y);
int2 popcount(const int2& x);
int2 rotate(const int2& v, const int2& i);
int2 sub_sat(const int2& x, const int2& y);

int3 clamp(const int3& x, const int3& minval, const int3& maxval);
int3 clz(const int3& x);
int3 hadd(const int3& x, const int3& y);
int3 rhadd(const int3& x, const int3& y);
int3 mad24(const int3& x, const int3& y, const int3& z);
int3 mad_hi(const int3& a, const int3& b, const int3& c);
int3 mad_sat(const int3& a, const int3& b, const int3& c);
int3 max(const int3& x, const int3& y);
int3 min(const int3& x, const int3& y);
int3 mul24(const int3& x, const int3& y);
int3 mul_hi(const int3& x, const int3& y);
int3 popcount(const int3& x);
int3 rotate(const int3& v, const int3& i);
int3 sub_sat(const int3& x, const int3& y);

int4 clamp(const int4& x, const int4& minval, const int4& maxval);
int4 clz(const int4& x);
int4 hadd(const int4& x, const int4& y);
int4 rhadd(const int4& x, const int4& y);
int4 mad24(const int4& x, const int4& y, const int4& z);
int4 mad_hi(const int4& a, const int4& b, const int4& c);
int4 mad_sat(const int4& a, const int4& b, const int4& c);
int4 max(const int4& x, const int4& y);
int4 min(const int4& x, const int4& y);
int4 mul24(const int4& x, const int4& y);
int4 mul_hi(const int4& x, const int4& y);
int4 popcount(const int4& x);
int4 rotate(const int4& v, const int4& i);
int4 sub_sat(const int4& x, const int4& y);

int8 clamp(const int8& x, const int8& minval, const int8& maxval);
int8 clz(const int8& x);
int8 hadd(const int8& x, const int8& y);
int8 rhadd(const int8& x, const int8& y);
int8 mad24(const int8& x, const int8& y, const int8& z);
int8 mad_hi(const int8& a, const int8& b, const int8& c);
int8 mad_sat(const int8& a, const int8& b, const int8& c);
int8 max(const int8& x, const int8& y);
int8 min(const int8& x, const int8& y);
int8 mul24(const int8& x, const int8& y);
int8 mul_hi(const int8& x, const int8& y);
int8 popcount(const int8& x);
int8 rotate(const int8& v, const int8& i);
int8 sub_sat(const int8& x, const int8& y);

int16 clamp(const int16& x, const int16& minval, const int16& maxval);
int16 clz(const int16& x);
int16 hadd(const int16& x, const int16& y);
int16 rhadd(const int16& x, const int16& y);
int16 mad24(const int16& x, const int16& y, const int16& z);
int16 mad_hi(const int16& a, const int16& b, const int16& c);
int16 mad_sat(const int16& a, const int16& b, const int16& c);
int16 max(const int16& x, const int16& y);
int16 min(const int16& x, const int16& y);
int16 mul24(const int16& x, const int16& y);
int16 mul_hi(const int16& x, const int16& y);
int16 popcount(const int16& x);
int16 rotate(const int16& v, const int16& i);
int16 sub_sat(const int16& x, const int16& y);

uint clamp(const uint& x, const uint& minval, const uint& maxval);
uint clz(const uint& x);
uint hadd(const uint& x, const uint& y);
uint rhadd(const uint& x, const uint& y);
uint mad24(const uint& x, const uint& y, const uint& z);
uint mad_hi(const uint& a, const uint& b, const uint& c);
uint mad_sat(const uint& a, const uint& b, const uint& c);
uint max(const uint& x, const uint& y);
uint min(const uint& x, const uint& y);
uint mul24(const uint& x, const uint& y);
uint mul_hi(const uint& x, const uint& y);
uint popcount(const uint& x);
uint rotate(const uint& v, const uint& i);
uint sub_sat(const uint& x, const uint& y);

uint2 clamp(const uint2& x, const uint2& minval, const uint2& maxval);
uint2 clz(const uint2& x);
uint2 hadd(const uint2& x, const uint2& y);
uint2 rhadd(const uint2& x, const uint2& y);
uint2 mad24(const uint2& x, const uint2& y, const uint2& z);
uint2 mad_hi(const uint2& a, const uint2& b, const uint2& c);
uint2 mad_sat(const uint2& a, const uint2& b, const uint2& c);
uint2 max(const uint2& x, const uint2& y);
uint2 min(const uint2& x, const uint2& y);
uint2 mul24(const uint2& x, const uint2& y);
uint2 mul_hi(const uint2& x, const uint2& y);
uint2 popcount(const uint2& x);
uint2 rotate(const uint2& v, const uint2& i);
uint2 sub_sat(const uint2& x, const uint2& y);

uint3 clamp(const uint3& x, const uint3& minval, const uint3& maxval);
uint3 clz(const uint3& x);
uint3 hadd(const uint3& x, const uint3& y);
uint3 rhadd(const uint3& x, const uint3& y);
uint3 mad24(const uint3& x, const uint3& y, const uint3& z);
uint3 mad_hi(const uint3& a, const uint3& b, const uint3& c);
uint3 mad_sat(const uint3& a, const uint3& b, const uint3& c);
uint3 max(const uint3& x, const uint3& y);
uint3 min(const uint3& x, const uint3& y);
uint3 mul24(const uint3& x, const uint3& y);
uint3 mul_hi(const uint3& x, const uint3& y);
uint3 popcount(const uint3& x);
uint3 rotate(const uint3& v, const uint3& i);
uint3 sub_sat(const uint3& x, const uint3& y);

uint4 clamp(const uint4& x, const uint4& minval, const uint4& maxval);
uint4 clz(const uint4& x);
uint4 hadd(const uint4& x, const uint4& y);
uint4 rhadd(const uint4& x, const uint4& y);
uint4 mad24(const uint4& x, const uint4& y, const uint4& z);
uint4 mad_hi(const uint4& a, const uint4& b, const uint4& c);
uint4 mad_sat(const uint4& a, const uint4& b, const uint4& c);
uint4 max(const uint4& x, const uint4& y);
uint4 min(const uint4& x, const uint4& y);
uint4 mul24(const uint4& x, const uint4& y);
uint4 mul_hi(const uint4& x, const uint4& y);
uint4 popcount(const uint4& x);
uint4 rotate(const uint4& v, const uint4& i);
uint4 sub_sat(const uint4& x, const uint4& y);

uint8 clamp(const uint8& x, const uint8& minval, const uint8& maxval);
uint8 clz(const uint8& x);
uint8 hadd(const uint8& x, const uint8& y);
uint8 rhadd(const uint8& x, const uint8& y);
uint8 mad24(const uint8& x, const uint8& y, const uint8& z);
uint8 mad_hi(const uint8& a, const uint8& b, const uint8& c);
uint8 mad_sat(const uint8& a, const uint8& b, const uint8& c);
uint8 max(const uint8& x, const uint8& y);
uint8 min(const uint8& x, const uint8& y);
uint8 mul24(const uint8& x, const uint8& y);
uint8 mul_hi(const uint8& x, const uint8& y);
uint8 popcount(const uint8& x);
uint8 rotate(const uint8& v, const uint8& i);
uint8 sub_sat(const uint8& x, const uint8& y);

uint16 clamp(const uint16& x, const uint16& minval, const uint16& maxval);
uint16 clz(const uint16& x);
uint16 hadd(const uint16& x, const uint16& y);
uint16 rhadd(const uint16& x, const uint16& y);
uint16 mad24(const uint16& x, const uint16& y, const uint16& z);
uint16 mad_hi(const uint16& a, const uint16& b, const uint16& c);
uint16 mad_sat(const uint16& a, const uint16& b, const uint16& c);
uint16 max(const uint16& x, const uint16& y);
uint16 min(const uint16& x, const uint16& y);
uint16 mul24(const uint16& x, const uint16& y);
uint16 mul_hi(const uint16& x, const uint16& y);
uint16 popcount(const uint16& x);
uint16 rotate(const uint16& v, const uint16& i);
uint16 sub_sat(const uint16& x, const uint16& y);

long clamp(const long& x, const long& minval, const long& maxval);
long clz(const long& x);
long hadd(const long& x, const long& y);
long rhadd(const long& x, const long& y);
long mad24(const long& x, const long& y, const long& z);
long mad_hi(const long& a, const long& b, const long& c);
long mad_sat(const long& a, const long& b, const long& c);
long max(const long& x, const long& y);
long min(const long& x, const long& y);
long mul24(const long& x, const long& y);
long mul_hi(const long& x, const long& y);
long popcount(const long& x);
long rotate(const long& v, const long& i);
long sub_sat(const long& x, const long& y);

long2 clamp(const long2& x, const long2& minval, const long2& maxval);
long2 clz(const long2& x);
long2 hadd(const long2& x, const long2& y);
long2 rhadd(const long2& x, const long2& y);
long2 mad24(const long2& x, const long2& y, const long2& z);
long2 mad_hi(const long2& a, const long2& b, const long2& c);
long2 mad_sat(const long2& a, const long2& b, const long2& c);
long2 max(const long2& x, const long2& y);
long2 min(const long2& x, const long2& y);
long2 mul24(const long2& x, const long2& y);
long2 mul_hi(const long2& x, const long2& y);
long2 popcount(const long2& x);
long2 rotate(const long2& v, const long2& i);
long2 sub_sat(const long2& x, const long2& y);

long3 clamp(const long3& x, const long3& minval, const long3& maxval);
long3 clz(const long3& x);
long3 hadd(const long3& x, const long3& y);
long3 rhadd(const long3& x, const long3& y);
long3 mad24(const long3& x, const long3& y, const long3& z);
long3 mad_hi(const long3& a, const long3& b, const long3& c);
long3 mad_sat(const long3& a, const long3& b, const long3& c);
long3 max(const long3& x, const long3& y);
long3 min(const long3& x, const long3& y);
long3 mul24(const long3& x, const long3& y);
long3 mul_hi(const long3& x, const long3& y);
long3 popcount(const long3& x);
long3 rotate(const long3& v, const long3& i);
long3 sub_sat(const long3& x, const long3& y);

long4 clamp(const long4& x, const long4& minval, const long4& maxval);
long4 clz(const long4& x);
long4 hadd(const long4& x, const long4& y);
long4 rhadd(const long4& x, const long4& y);
long4 mad24(const long4& x, const long4& y, const long4& z);
long4 mad_hi(const long4& a, const long4& b, const long4& c);
long4 mad_sat(const long4& a, const long4& b, const long4& c);
long4 max(const long4& x, const long4& y);
long4 min(const long4& x, const long4& y);
long4 mul24(const long4& x, const long4& y);
long4 mul_hi(const long4& x, const long4& y);
long4 popcount(const long4& x);
long4 rotate(const long4& v, const long4& i);
long4 sub_sat(const long4& x, const long4& y);

long8 clamp(const long8& x, const long8& minval, const long8& maxval);
long8 clz(const long8& x);
long8 hadd(const long8& x, const long8& y);
long8 rhadd(const long8& x, const long8& y);
long8 mad24(const long8& x, const long8& y, const long8& z);
long8 mad_hi(const long8& a, const long8& b, const long8& c);
long8 mad_sat(const long8& a, const long8& b, const long8& c);
long8 max(const long8& x, const long8& y);
long8 min(const long8& x, const long8& y);
long8 mul24(const long8& x, const long8& y);
long8 mul_hi(const long8& x, const long8& y);
long8 popcount(const long8& x);
long8 rotate(const long8& v, const long8& i);
long8 sub_sat(const long8& x, const long8& y);

long16 clamp(const long16& x, const long16& minval, const long16& maxval);
long16 clz(const long16& x);
long16 hadd(const long16& x, const long16& y);
long16 rhadd(const long16& x, const long16& y);
long16 mad24(const long16& x, const long16& y, const long16& z);
long16 mad_hi(const long16& a, const long16& b, const long16& c);
long16 mad_sat(const long16& a, const long16& b, const long16& c);
long16 max(const long16& x, const long16& y);
long16 min(const long16& x, const long16& y);
long16 mul24(const long16& x, const long16& y);
long16 mul_hi(const long16& x, const long16& y);
long16 popcount(const long16& x);
long16 rotate(const long16& v, const long16& i);
long16 sub_sat(const long16& x, const long16& y);

ulong clamp(const ulong& x, const ulong& minval, const ulong& maxval);
ulong clz(const ulong& x);
ulong hadd(const ulong& x, const ulong& y);
ulong rhadd(const ulong& x, const ulong& y);
ulong mad24(const ulong& x, const ulong& y, const ulong& z);
ulong mad_hi(const ulong& a, const ulong& b, const ulong& c);
ulong mad_sat(const ulong& a, const ulong& b, const ulong& c);
ulong max(const ulong& x, const ulong& y);
ulong min(const ulong& x, const ulong& y);
ulong mul24(const ulong& x, const ulong& y);
ulong mul_hi(const ulong& x, const ulong& y);
ulong popcount(const ulong& x);
ulong rotate(const ulong& v, const ulong& i);
ulong sub_sat(const ulong& x, const ulong& y);

ulong2 clamp(const ulong2& x, const ulong2& minval, const ulong2& maxval);
ulong2 clz(const ulong2& x);
ulong2 hadd(const ulong2& x, const ulong2& y);
ulong2 rhadd(const ulong2& x, const ulong2& y);
ulong2 mad24(const ulong2& x, const ulong2& y, const ulong2& z);
ulong2 mad_hi(const ulong2& a, const ulong2& b, const ulong2& c);
ulong2 mad_sat(const ulong2& a, const ulong2& b, const ulong2& c);
ulong2 max(const ulong2& x, const ulong2& y);
ulong2 min(const ulong2& x, const ulong2& y);
ulong2 mul24(const ulong2& x, const ulong2& y);
ulong2 mul_hi(const ulong2& x, const ulong2& y);
ulong2 popcount(const ulong2& x);
ulong2 rotate(const ulong2& v, const ulong2& i);
ulong2 sub_sat(const ulong2& x, const ulong2& y);

ulong3 clamp(const ulong3& x, const ulong3& minval, const ulong3& maxval);
ulong3 clz(const ulong3& x);
ulong3 hadd(const ulong3& x, const ulong3& y);
ulong3 rhadd(const ulong3& x, const ulong3& y);
ulong3 mad24(const ulong3& x, const ulong3& y, const ulong3& z);
ulong3 mad_hi(const ulong3& a, const ulong3& b, const ulong3& c);
ulong3 mad_sat(const ulong3& a, const ulong3& b, const ulong3& c);
ulong3 max(const ulong3& x, const ulong3& y);
ulong3 min(const ulong3& x, const ulong3& y);
ulong3 mul24(const ulong3& x, const ulong3& y);
ulong3 mul_hi(const ulong3& x, const ulong3& y);
ulong3 popcount(const ulong3& x);
ulong3 rotate(const ulong3& v, const ulong3& i);
ulong3 sub_sat(const ulong3& x, const ulong3& y);

ulong4 clamp(const ulong4& x, const ulong4& minval, const ulong4& maxval);
ulong4 clz(const ulong4& x);
ulong4 hadd(const ulong4& x, const ulong4& y);
ulong4 rhadd(const ulong4& x, const ulong4& y);
ulong4 mad24(const ulong4& x, const ulong4& y, const ulong4& z);
ulong4 mad_hi(const ulong4& a, const ulong4& b, const ulong4& c);
ulong4 mad_sat(const ulong4& a, const ulong4& b, const ulong4& c);
ulong4 max(const ulong4& x, const ulong4& y);
ulong4 min(const ulong4& x, const ulong4& y);
ulong4 mul24(const ulong4& x, const ulong4& y);
ulong4 mul_hi(const ulong4& x, const ulong4& y);
ulong4 popcount(const ulong4& x);
ulong4 rotate(const ulong4& v, const ulong4& i);
ulong4 sub_sat(const ulong4& x, const ulong4& y);

ulong8 clamp(const ulong8& x, const ulong8& minval, const ulong8& maxval);
ulong8 clz(const ulong8& x);
ulong8 hadd(const ulong8& x, const ulong8& y);
ulong8 rhadd(const ulong8& x, const ulong8& y);
ulong8 mad24(const ulong8& x, const ulong8& y, const ulong8& z);
ulong8 mad_hi(const ulong8& a, const ulong8& b, const ulong8& c);
ulong8 mad_sat(const ulong8& a, const ulong8& b, const ulong8& c);
ulong8 max(const ulong8& x, const ulong8& y);
ulong8 min(const ulong8& x, const ulong8& y);
ulong8 mul24(const ulong8& x, const ulong8& y);
ulong8 mul_hi(const ulong8& x, const ulong8& y);
ulong8 popcount(const ulong8& x);
ulong8 rotate(const ulong8& v, const ulong8& i);
ulong8 sub_sat(const ulong8& x, const ulong8& y);

ulong16 clamp(const ulong16& x, const ulong16& minval, const ulong16& maxval);
ulong16 clz(const ulong16& x);
ulong16 hadd(const ulong16& x, const ulong16& y);
ulong16 rhadd(const ulong16& x, const ulong16& y);
ulong16 mad24(const ulong16& x, const ulong16& y, const ulong16& z);
ulong16 mad_hi(const ulong16& a, const ulong16& b, const ulong16& c);
ulong16 mad_sat(const ulong16& a, const ulong16& b, const ulong16& c);
ulong16 max(const ulong16& x, const ulong16& y);
ulong16 min(const ulong16& x, const ulong16& y);
ulong16 mul24(const ulong16& x, const ulong16& y);
ulong16 mul_hi(const ulong16& x, const ulong16& y);
ulong16 popcount(const ulong16& x);
ulong16 rotate(const ulong16& v, const ulong16& i);
ulong16 sub_sat(const ulong16& x, const ulong16& y);

short upsample(const char& hi, const uchar& lo);
ushort upsample(const uchar& hi, const uchar& lo);
int upsample(const short& hi, const ushort& lo);
uint upsample(const ushort& hi, const ushort& lo);
long upsample(const int& hi, const uint& lo);
ulong upsample(const uint& hi, const uint& lo);

short2 upsample(const char2& hi, const uchar2& lo);
ushort2 upsample(const uchar2& hi, const uchar2& lo);
int2 upsample(const short2& hi, const ushort2& lo);
uint2 upsample(const ushort2& hi, const ushort2& lo);
long2 upsample(const int2& hi, const uint2& lo);
ulong2 upsample(const uint2& hi, const uint2& lo);

short3 upsample(const char3& hi, const uchar3& lo);
ushort3 upsample(const uchar3& hi, const uchar3& lo);
int3 upsample(const short3& hi, const ushort3& lo);
uint3 upsample(const ushort3& hi, const ushort3& lo);
long3 upsample(const int3& hi, const uint3& lo);
ulong3 upsample(const uint3& hi, const uint3& lo);

short4 upsample(const char4& hi, const uchar4& lo);
ushort4 upsample(const uchar4& hi, const uchar4& lo);
int4 upsample(const short4& hi, const ushort4& lo);
uint4 upsample(const ushort4& hi, const ushort4& lo);
long4 upsample(const int4& hi, const uint4& lo);
ulong4 upsample(const uint4& hi, const uint4& lo);

short8 upsample(const char8& hi, const uchar8& lo);
ushort8 upsample(const uchar8& hi, const uchar8& lo);
int8 upsample(const short8& hi, const ushort8& lo);
uint8 upsample(const ushort8& hi, const ushort8& lo);
long8 upsample(const int8& hi, const uint8& lo);
ulong8 upsample(const uint8& hi, const uint8& lo);

short16 upsample(const char16& hi, const uchar16& lo);
ushort16 upsample(const uchar16& hi, const uchar16& lo);
int16 upsample(const short16& hi, const ushort16& lo);
uint16 upsample(const ushort16& hi, const ushort16& lo);
long16 upsample(const int16& hi, const uint16& lo);
ulong16 upsample(const uint16& hi, const uint16& lo);

float clamp(const float& x, const float& minval, const float& maxval);
float degrees(const float& radians);
float max(const float& x, const float& y);
float min(const float& x, const float& y);
float mix(const float& x, const float& y, const float& a);
float radians(const float& degrees);
float sign(const float& x);
float smoothstep(const float& edge0, const float& edge1, const float& x);
float step(const float& edge, const float& x);
float acos(const float& x);
float acosh(const float& x);
float acospi(const float& x);
float asin(const float& x);
float asinh(const float& x);
float asinpi(const float& x);
float atan(const float& y_over_x);
float atan2(const float& y, const float& x);
float atanh(const float& y_over_x);
float atanpi(const float& y_over_x);
float atan2pi(const float& y, const float& x);
float cbrt(const float& x);
float ceil(const float& x);
float copysign(const float& x, const float& y);
float cos(const float& x);
float cosh(const float& xv);
float cospi(const float& x);
float half_cos(const float& x);
float native_cos(const float& x);
float half_divide(const float& x, const float& y);
float native_divide(const float& x, const float& y);
float erf(const float& x);
float erfc(const float& x);
float exp(const float& x);
float exp2(const float& x);
float exp10(const float& x);
float expm1(const float& x);
float half_exp(const float& x);
float half_exp2(const float& x);
float half_exp10(const float& x);
float native_exp(const float& x);
float native_exp2(const float& x);
float native_exp10(const float& x);
float fabs(const float& x);
float fdim(const float& x, const float& y);
float floor(const float& x);
float fma(const float& a, const float& b, const float& c);
float fmax(const float& x, const float& y);
float fmin(const float& x, const float& y);
float fmod(const float& x);
float fract(const float& x, float* itptr = 0);
float frexp(const float& x, int* exp = 0);
float hypot(const float& x, const float& y);
int ilogb(const float& x);
float ldexp(const float& x, const int& k);
float lgamma(const float& x);
float lgamma_r(const float& x, int* signp);
float log(const float& x);
float log2(const float& x);
float log10(const float& x);
float log1p(const float& x);
float logb(const float& x);
float half_log(const float& x);
float half_log2(const float& x);
float half_log10(const float& x);
float native_log(const float& x);
float native_log2(const float& x);
float native_log10(const float& x);
float mad(const float& a, const float& b, const float& c);
float maxmag(const float& x, const float& y);
float minmag(const float& x, const float& y);
float modf(const float& x, int* iptr);
float nan(const uint& nancode);
float nextafter(const float& x, const float& y);
float pow(const float& x, const float& y);
float pown(const float& x, const int& y);
float powr(const float& x, const float& y);
float half_powr(const float& x, const float& y);
float native_powr(const float& x, const float& y);
float half_recip(const float& x);
float native_recip(const float& x);
float remainder(const float& x, const float& y);
float remquo(const float& x, const float& y, int* quo = 0);
float rint(const float& x);
float round(const float& x);
float rootn(const float& x, const int& y);
float sqrt(const float& x);
float half_sqrt(const float& x);
float native_sqrt(const float& x);
float rsqrt(const float& x);
float half_rsqrt(const float& x);
float native_rsqrt(const float& x);
float sin(const float& x);
float sincos(const float& x, float* cosval);
float sinh(const float& x);
float sinpi(const float& x);
float half_sin(const float& x);
float native_sin(const float& x);
float tan(const float& x);
float tanh(const float& x);
float tanpi(const float& x);
float half_tan(const float& x);
float native_tan(const float& x);
float tgamma(const float& x);
float trunc(const float& x);

float2 clamp(const float2& x, const float2& minval, const float2& maxval);
float2 degrees(const float2& radians);
float2 max(const float2& x, const float2& y);
float2 min(const float2& x, const float2& y);
float2 mix(const float2& x, const float2& y, const float2& a);
float2 radians(const float2& degrees);
float2 sign(const float2& x);
float2 smoothstep(const float2& edge0, const float2& edge1, const float2& x);
float2 step(const float2& edge, const float2& x);
float2 acos(const float2& x);
float2 acosh(const float2& x);
float2 acospi(const float2& x);
float2 asin(const float2& x);
float2 asinh(const float2& x);
float2 asinpi(const float2& x);
float2 atan(const float2& y_over_x);
float2 atan2(const float2& y, const float2& x);
float2 atanh(const float2& y_over_x);
float2 atanpi(const float2& y_over_x);
float2 atan2pi(const float2& y, const float2& x);
float2 cbrt(const float2& x);
float2 ceil(const float2& x);
float2 copysign(const float2& x, const float2& y);
float2 cos(const float2& x);
float2 cosh(const float2& xv);
float2 cospi(const float2& x);
float2 half_cos(const float2& x);
float2 native_cos(const float2& x);
float2 half_divide(const float2& x, const float2& y);
float2 native_divide(const float2& x, const float2& y);
float2 erf(const float2& x);
float2 erfc(const float2& x);
float2 exp(const float2& x);
float2 exp2(const float2& x);
float2 exp10(const float2& x);
float2 expm1(const float2& x);
float2 half_exp(const float2& x);
float2 half_exp2(const float2& x);
float2 half_exp10(const float2& x);
float2 native_exp(const float2& x);
float2 native_exp2(const float2& x);
float2 native_exp10(const float2& x);
float2 fabs(const float2& x);
float2 fdim(const float2& x, const float2& y);
float2 floor(const float2& x);
float2 fma(const float2& a, const float2& b, const float2& c);
float2 fmax(const float2& x, const float2& y);
float2 fmin(const float2& x, const float2& y);
float2 fmod(const float2& x);
float2 fract(const float2& x, float2* itptr = 0);
float2 frexp(const float2& x, int2* exp = 0);
float2 hypot(const float2& x, const float2& y);
int2 ilogb(const float2& x);
float2 ldexp(const float2& x, const int2& k);
float2 lgamma(const float2& x);
float2 lgamma_r(const float2& x, int2* signp);
float2 log(const float2& x);
float2 log2(const float2& x);
float2 log10(const float2& x);
float2 log1p(const float2& x);
float2 logb(const float2& x);
float2 half_log(const float2& x);
float2 half_log2(const float2& x);
float2 half_log10(const float2& x);
float2 native_log(const float2& x);
float2 native_log2(const float2& x);
float2 native_log10(const float2& x);
float2 mad(const float2& a, const float2& b, const float2& c);
float2 maxmag(const float2& x, const float2& y);
float2 minmag(const float2& x, const float2& y);
float2 modf(const float2& x, int2* iptr);
float2 nan(const uint2& nancode);
float2 nextafter(const float2& x, const float2& y);
float2 pow(const float2& x, const float2& y);
float2 pown(const float2& x, const int2& y);
float2 powr(const float2& x, const float2& y);
float2 half_powr(const float2& x, const float2& y);
float2 native_powr(const float2& x, const float2& y);
float2 half_recip(const float2& x);
float2 native_recip(const float2& x);
float2 remainder(const float2& x, const float2& y);
float2 remquo(const float2& x, const float2& y, int2* quo = 0);
float2 rint(const float2& x);
float2 round(const float2& x);
float2 rootn(const float2& x, const int2& y);
float2 sqrt(const float2& x);
float2 half_sqrt(const float2& x);
float2 native_sqrt(const float2& x);
float2 rsqrt(const float2& x);
float2 half_rsqrt(const float2& x);
float2 native_rsqrt(const float2& x);
float2 sin(const float2& x);
float2 sincos(const float2& x, float2* cosval);
float2 sinh(const float2& x);
float2 sinpi(const float2& x);
float2 half_sin(const float2& x);
float2 native_sin(const float2& x);
float2 tan(const float2& x);
float2 tanh(const float2& x);
float2 tanpi(const float2& x);
float2 half_tan(const float2& x);
float2 native_tan(const float2& x);
float2 tgamma(const float2& x);
float2 trunc(const float2& x);

float3 clamp(const float3& x, const float3& minval, const float3& maxval);
float3 degrees(const float3& radians);
float3 max(const float3& x, const float3& y);
float3 min(const float3& x, const float3& y);
float3 mix(const float3& x, const float3& y, const float3& a);
float3 radians(const float3& degrees);
float3 sign(const float3& x);
float3 smoothstep(const float3& edge0, const float3& edge1, const float3& x);
float3 step(const float3& edge, const float3& x);
float3 acos(const float3& x);
float3 acosh(const float3& x);
float3 acospi(const float3& x);
float3 asin(const float3& x);
float3 asinh(const float3& x);
float3 asinpi(const float3& x);
float3 atan(const float3& y_over_x);
float3 atan2(const float3& y, const float3& x);
float3 atanh(const float3& y_over_x);
float3 atanpi(const float3& y_over_x);
float3 atan2pi(const float3& y, const float3& x);
float3 cbrt(const float3& x);
float3 ceil(const float3& x);
float3 copysign(const float3& x, const float3& y);
float3 cos(const float3& x);
float3 cosh(const float3& xv);
float3 cospi(const float3& x);
float3 half_cos(const float3& x);
float3 native_cos(const float3& x);
float3 half_divide(const float3& x, const float3& y);
float3 native_divide(const float3& x, const float3& y);
float3 erf(const float3& x);
float3 erfc(const float3& x);
float3 exp(const float3& x);
float3 exp2(const float3& x);
float3 exp10(const float3& x);
float3 expm1(const float3& x);
float3 half_exp(const float3& x);
float3 half_exp2(const float3& x);
float3 half_exp10(const float3& x);
float3 native_exp(const float3& x);
float3 native_exp2(const float3& x);
float3 native_exp10(const float3& x);
float3 fabs(const float3& x);
float3 fdim(const float3& x, const float3& y);
float3 floor(const float3& x);
float3 fma(const float3& a, const float3& b, const float3& c);
float3 fmax(const float3& x, const float3& y);
float3 fmin(const float3& x, const float3& y);
float3 fmod(const float3& x);
float3 fract(const float3& x, float3* itptr = 0);
float3 frexp(const float3& x, int3* exp = 0);
float3 hypot(const float3& x, const float3& y);
int3 ilogb(const float3& x);
float3 ldexp(const float3& x, const int3& k);
float3 lgamma(const float3& x);
float3 lgamma_r(const float3& x, int3* signp);
float3 log(const float3& x);
float3 log2(const float3& x);
float3 log10(const float3& x);
float3 log1p(const float3& x);
float3 logb(const float3& x);
float3 half_log(const float3& x);
float3 half_log2(const float3& x);
float3 half_log10(const float3& x);
float3 native_log(const float3& x);
float3 native_log2(const float3& x);
float3 native_log10(const float3& x);
float3 mad(const float3& a, const float3& b, const float3& c);
float3 maxmag(const float3& x, const float3& y);
float3 minmag(const float3& x, const float3& y);
float3 modf(const float3& x, int3* iptr);
float3 nan(const uint3& nancode);
float3 nextafter(const float3& x, const float3& y);
float3 pow(const float3& x, const float3& y);
float3 pown(const float3& x, const int3& y);
float3 powr(const float3& x, const float3& y);
float3 half_powr(const float3& x, const float3& y);
float3 native_powr(const float3& x, const float3& y);
float3 half_recip(const float3& x);
float3 native_recip(const float3& x);
float3 remainder(const float3& x, const float3& y);
float3 remquo(const float3& x, const float3& y, int3* quo = 0);
float3 rint(const float3& x);
float3 round(const float3& x);
float3 rootn(const float3& x, const int3& y);
float3 sqrt(const float3& x);
float3 half_sqrt(const float3& x);
float3 native_sqrt(const float3& x);
float3 rsqrt(const float3& x);
float3 half_rsqrt(const float3& x);
float3 native_rsqrt(const float3& x);
float3 sin(const float3& x);
float3 sincos(const float3& x, float3* cosval);
float3 sinh(const float3& x);
float3 sinpi(const float3& x);
float3 half_sin(const float3& x);
float3 native_sin(const float3& x);
float3 tan(const float3& x);
float3 tanh(const float3& x);
float3 tanpi(const float3& x);
float3 half_tan(const float3& x);
float3 native_tan(const float3& x);
float3 tgamma(const float3& x);
float3 trunc(const float3& x);

float4 clamp(const float4& x, const float4& minval, const float4& maxval);
float4 degrees(const float4& radians);
float4 max(const float4& x, const float4& y);
float4 min(const float4& x, const float4& y);
float4 mix(const float4& x, const float4& y, const float4& a);
float4 radians(const float4& degrees);
float4 sign(const float4& x);
float4 smoothstep(const float4& edge0, const float4& edge1, const float4& x);
float4 step(const float4& edge, const float4& x);
float4 acos(const float4& x);
float4 acosh(const float4& x);
float4 acospi(const float4& x);
float4 asin(const float4& x);
float4 asinh(const float4& x);
float4 asinpi(const float4& x);
float4 atan(const float4& y_over_x);
float4 atan2(const float4& y, const float4& x);
float4 atanh(const float4& y_over_x);
float4 atanpi(const float4& y_over_x);
float4 atan2pi(const float4& y, const float4& x);
float4 cbrt(const float4& x);
float4 ceil(const float4& x);
float4 copysign(const float4& x, const float4& y);
float4 cos(const float4& x);
float4 cosh(const float4& xv);
float4 cospi(const float4& x);
float4 half_cos(const float4& x);
float4 native_cos(const float4& x);
float4 half_divide(const float4& x, const float4& y);
float4 native_divide(const float4& x, const float4& y);
float4 erf(const float4& x);
float4 erfc(const float4& x);
float4 exp(const float4& x);
float4 exp2(const float4& x);
float4 exp10(const float4& x);
float4 expm1(const float4& x);
float4 half_exp(const float4& x);
float4 half_exp2(const float4& x);
float4 half_exp10(const float4& x);
float4 native_exp(const float4& x);
float4 native_exp2(const float4& x);
float4 native_exp10(const float4& x);
float4 fabs(const float4& x);
float4 fdim(const float4& x, const float4& y);
float4 floor(const float4& x);
float4 fma(const float4& a, const float4& b, const float4& c);
float4 fmax(const float4& x, const float4& y);
float4 fmin(const float4& x, const float4& y);
float4 fmod(const float4& x);
float4 fract(const float4& x, float4* itptr = 0);
float4 frexp(const float4& x, int4* exp = 0);
float4 hypot(const float4& x, const float4& y);
int4 ilogb(const float4& x);
float4 ldexp(const float4& x, const int4& k);
float4 lgamma(const float4& x);
float4 lgamma_r(const float4& x, int4* signp);
float4 log(const float4& x);
float4 log2(const float4& x);
float4 log10(const float4& x);
float4 log1p(const float4& x);
float4 logb(const float4& x);
float4 half_log(const float4& x);
float4 half_log2(const float4& x);
float4 half_log10(const float4& x);
float4 native_log(const float4& x);
float4 native_log2(const float4& x);
float4 native_log10(const float4& x);
float4 mad(const float4& a, const float4& b, const float4& c);
float4 maxmag(const float4& x, const float4& y);
float4 minmag(const float4& x, const float4& y);
float4 modf(const float4& x, int4* iptr);
float4 nan(const uint4& nancode);
float4 nextafter(const float4& x, const float4& y);
float4 pow(const float4& x, const float4& y);
float4 pown(const float4& x, const int4& y);
float4 powr(const float4& x, const float4& y);
float4 half_powr(const float4& x, const float4& y);
float4 native_powr(const float4& x, const float4& y);
float4 half_recip(const float4& x);
float4 native_recip(const float4& x);
float4 remainder(const float4& x, const float4& y);
float4 remquo(const float4& x, const float4& y, int4* quo = 0);
float4 rint(const float4& x);
float4 round(const float4& x);
float4 rootn(const float4& x, const int4& y);
float4 sqrt(const float4& x);
float4 half_sqrt(const float4& x);
float4 native_sqrt(const float4& x);
float4 rsqrt(const float4& x);
float4 half_rsqrt(const float4& x);
float4 native_rsqrt(const float4& x);
float4 sin(const float4& x);
float4 sincos(const float4& x, float4* cosval);
float4 sinh(const float4& x);
float4 sinpi(const float4& x);
float4 half_sin(const float4& x);
float4 native_sin(const float4& x);
float4 tan(const float4& x);
float4 tanh(const float4& x);
float4 tanpi(const float4& x);
float4 half_tan(const float4& x);
float4 native_tan(const float4& x);
float4 tgamma(const float4& x);
float4 trunc(const float4& x);

float8 clamp(const float8& x, const float8& minval, const float8& maxval);
float8 degrees(const float8& radians);
float8 max(const float8& x, const float8& y);
float8 min(const float8& x, const float8& y);
float8 mix(const float8& x, const float8& y, const float8& a);
float8 radians(const float8& degrees);
float8 sign(const float8& x);
float8 smoothstep(const float8& edge0, const float8& edge1, const float8& x);
float8 step(const float8& edge, const float8& x);
float8 acos(const float8& x);
float8 acosh(const float8& x);
float8 acospi(const float8& x);
float8 asin(const float8& x);
float8 asinh(const float8& x);
float8 asinpi(const float8& x);
float8 atan(const float8& y_over_x);
float8 atan2(const float8& y, const float8& x);
float8 atanh(const float8& y_over_x);
float8 atanpi(const float8& y_over_x);
float8 atan2pi(const float8& y, const float8& x);
float8 cbrt(const float8& x);
float8 ceil(const float8& x);
float8 copysign(const float8& x, const float8& y);
float8 cos(const float8& x);
float8 cosh(const float8& xv);
float8 cospi(const float8& x);
float8 half_cos(const float8& x);
float8 native_cos(const float8& x);
float8 half_divide(const float8& x, const float8& y);
float8 native_divide(const float8& x, const float8& y);
float8 erf(const float8& x);
float8 erfc(const float8& x);
float8 exp(const float8& x);
float8 exp2(const float8& x);
float8 exp10(const float8& x);
float8 expm1(const float8& x);
float8 half_exp(const float8& x);
float8 half_exp2(const float8& x);
float8 half_exp10(const float8& x);
float8 native_exp(const float8& x);
float8 native_exp2(const float8& x);
float8 native_exp10(const float8& x);
float8 fabs(const float8& x);
float8 fdim(const float8& x, const float8& y);
float8 floor(const float8& x);
float8 fma(const float8& a, const float8& b, const float8& c);
float8 fmax(const float8& x, const float8& y);
float8 fmin(const float8& x, const float8& y);
float8 fmod(const float8& x);
float8 fract(const float8& x, float8* itptr = 0);
float8 frexp(const float8& x, int8* exp = 0);
float8 hypot(const float8& x, const float8& y);
int8 ilogb(const float8& x);
float8 ldexp(const float8& x, const int8& k);
float8 lgamma(const float8& x);
float8 lgamma_r(const float8& x, int8* signp);
float8 log(const float8& x);
float8 log2(const float8& x);
float8 log10(const float8& x);
float8 log1p(const float8& x);
float8 logb(const float8& x);
float8 half_log(const float8& x);
float8 half_log2(const float8& x);
float8 half_log10(const float8& x);
float8 native_log(const float8& x);
float8 native_log2(const float8& x);
float8 native_log10(const float8& x);
float8 mad(const float8& a, const float8& b, const float8& c);
float8 maxmag(const float8& x, const float8& y);
float8 minmag(const float8& x, const float8& y);
float8 modf(const float8& x, int8* iptr);
float8 nan(const uint8& nancode);
float8 nextafter(const float8& x, const float8& y);
float8 pow(const float8& x, const float8& y);
float8 pown(const float8& x, const int8& y);
float8 powr(const float8& x, const float8& y);
float8 half_powr(const float8& x, const float8& y);
float8 native_powr(const float8& x, const float8& y);
float8 half_recip(const float8& x);
float8 native_recip(const float8& x);
float8 remainder(const float8& x, const float8& y);
float8 remquo(const float8& x, const float8& y, int8* quo = 0);
float8 rint(const float8& x);
float8 round(const float8& x);
float8 rootn(const float8& x, const int8& y);
float8 sqrt(const float8& x);
float8 half_sqrt(const float8& x);
float8 native_sqrt(const float8& x);
float8 rsqrt(const float8& x);
float8 half_rsqrt(const float8& x);
float8 native_rsqrt(const float8& x);
float8 sin(const float8& x);
float8 sincos(const float8& x, float8* cosval);
float8 sinh(const float8& x);
float8 sinpi(const float8& x);
float8 half_sin(const float8& x);
float8 native_sin(const float8& x);
float8 tan(const float8& x);
float8 tanh(const float8& x);
float8 tanpi(const float8& x);
float8 half_tan(const float8& x);
float8 native_tan(const float8& x);
float8 tgamma(const float8& x);
float8 trunc(const float8& x);

float16 clamp(const float16& x, const float16& minval, const float16& maxval);
float16 degrees(const float16& radians);
float16 max(const float16& x, const float16& y);
float16 min(const float16& x, const float16& y);
float16 mix(const float16& x, const float16& y, const float16& a);
float16 radians(const float16& degrees);
float16 sign(const float16& x);
float16 smoothstep(const float16& edge0, const float16& edge1, const float16& x);
float16 step(const float16& edge, const float16& x);
float16 acos(const float16& x);
float16 acosh(const float16& x);
float16 acospi(const float16& x);
float16 asin(const float16& x);
float16 asinh(const float16& x);
float16 asinpi(const float16& x);
float16 atan(const float16& y_over_x);
float16 atan2(const float16& y, const float16& x);
float16 atanh(const float16& y_over_x);
float16 atanpi(const float16& y_over_x);
float16 atan2pi(const float16& y, const float16& x);
float16 cbrt(const float16& x);
float16 ceil(const float16& x);
float16 copysign(const float16& x, const float16& y);
float16 cos(const float16& x);
float16 cosh(const float16& xv);
float16 cospi(const float16& x);
float16 half_cos(const float16& x);
float16 native_cos(const float16& x);
float16 half_divide(const float16& x, const float16& y);
float16 native_divide(const float16& x, const float16& y);
float16 erf(const float16& x);
float16 erfc(const float16& x);
float16 exp(const float16& x);
float16 exp2(const float16& x);
float16 exp10(const float16& x);
float16 expm1(const float16& x);
float16 half_exp(const float16& x);
float16 half_exp2(const float16& x);
float16 half_exp10(const float16& x);
float16 native_exp(const float16& x);
float16 native_exp2(const float16& x);
float16 native_exp10(const float16& x);
float16 fabs(const float16& x);
float16 fdim(const float16& x, const float16& y);
float16 floor(const float16& x);
float16 fma(const float16& a, const float16& b, const float16& c);
float16 fmax(const float16& x, const float16& y);
float16 fmin(const float16& x, const float16& y);
float16 fmod(const float16& x);
float16 fract(const float16& x, float16* itptr = 0);
float16 frexp(const float16& x, int16* exp = 0);
float16 hypot(const float16& x, const float16& y);
int16 ilogb(const float16& x);
float16 ldexp(const float16& x, const int16& k);
float16 lgamma(const float16& x);
float16 lgamma_r(const float16& x, int16* signp);
float16 log(const float16& x);
float16 log2(const float16& x);
float16 log10(const float16& x);
float16 log1p(const float16& x);
float16 logb(const float16& x);
float16 half_log(const float16& x);
float16 half_log2(const float16& x);
float16 half_log10(const float16& x);
float16 native_log(const float16& x);
float16 native_log2(const float16& x);
float16 native_log10(const float16& x);
float16 mad(const float16& a, const float16& b, const float16& c);
float16 maxmag(const float16& x, const float16& y);
float16 minmag(const float16& x, const float16& y);
float16 modf(const float16& x, int16* iptr);
float16 nan(const uint16& nancode);
float16 nextafter(const float16& x, const float16& y);
float16 pow(const float16& x, const float16& y);
float16 pown(const float16& x, const int16& y);
float16 powr(const float16& x, const float16& y);
float16 half_powr(const float16& x, const float16& y);
float16 native_powr(const float16& x, const float16& y);
float16 half_recip(const float16& x);
float16 native_recip(const float16& x);
float16 remainder(const float16& x, const float16& y);
float16 remquo(const float16& x, const float16& y, int16* quo = 0);
float16 rint(const float16& x);
float16 round(const float16& x);
float16 rootn(const float16& x, const int16& y);
float16 sqrt(const float16& x);
float16 half_sqrt(const float16& x);
float16 native_sqrt(const float16& x);
float16 rsqrt(const float16& x);
float16 half_rsqrt(const float16& x);
float16 native_rsqrt(const float16& x);
float16 sin(const float16& x);
float16 sincos(const float16& x, float16* cosval);
float16 sinh(const float16& x);
float16 sinpi(const float16& x);
float16 half_sin(const float16& x);
float16 native_sin(const float16& x);
float16 tan(const float16& x);
float16 tanh(const float16& x);
float16 tanpi(const float16& x);
float16 half_tan(const float16& x);
float16 native_tan(const float16& x);
float16 tgamma(const float16& x);
float16 trunc(const float16& x);

double clamp(const double& x, const double& minval, const double& maxval);
double degrees(const double& radians);
double max(const double& x, const double& y);
double min(const double& x, const double& y);
double mix(const double& x, const double& y, const double& a);
double radians(const double& degrees);
double sign(const double& x);
double smoothstep(const double& edge0, const double& edge1, const double& x);
double step(const double& edge, const double& x);
double acos(const double& x);
double acosh(const double& x);
double acospi(const double& x);
double asin(const double& x);
double asinh(const double& x);
double asinpi(const double& x);
double atan(const double& y_over_x);
double atan2(const double& y, const double& x);
double atanh(const double& y_over_x);
double atanpi(const double& y_over_x);
double atan2pi(const double& y, const double& x);
double cbrt(const double& x);
double ceil(const double& x);
double copysign(const double& x, const double& y);
double cos(const double& x);
double cosh(const double& xv);
double cospi(const double& x);
double half_cos(const double& x);
double native_cos(const double& x);
double half_divide(const double& x, const double& y);
double native_divide(const double& x, const double& y);
double erf(const double& x);
double erfc(const double& x);
double exp(const double& x);
double exp2(const double& x);
double exp10(const double& x);
double expm1(const double& x);
double half_exp(const double& x);
double half_exp2(const double& x);
double half_exp10(const double& x);
double native_exp(const double& x);
double native_exp2(const double& x);
double native_exp10(const double& x);
double fabs(const double& x);
double fdim(const double& x, const double& y);
double floor(const double& x);
double fma(const double& a, const double& b, const double& c);
double fmax(const double& x, const double& y);
double fmin(const double& x, const double& y);
double fmod(const double& x);
double fract(const double& x, double* itptr = 0);
double frexp(const double& x, int* exp = 0);
double hypot(const double& x, const double& y);
int ilogb(const double& x);
double ldexp(const double& x, const int& k);
double lgamma(const double& x);
double lgamma_r(const double& x, int* signp);
double log(const double& x);
double log2(const double& x);
double log10(const double& x);
double log1p(const double& x);
double logb(const double& x);
double half_log(const double& x);
double half_log2(const double& x);
double half_log10(const double& x);
double native_log(const double& x);
double native_log2(const double& x);
double native_log10(const double& x);
double mad(const double& a, const double& b, const double& c);
double maxmag(const double& x, const double& y);
double minmag(const double& x, const double& y);
double modf(const double& x, int* iptr);
double nan(const uint& nancode);
double nextafter(const double& x, const double& y);
double pow(const double& x, const double& y);
double pown(const double& x, const int& y);
double powr(const double& x, const double& y);
double half_powr(const double& x, const double& y);
double native_powr(const double& x, const double& y);
double half_recip(const double& x);
double native_recip(const double& x);
double remainder(const double& x, const double& y);
double remquo(const double& x, const double& y, int* quo = 0);
double rint(const double& x);
double round(const double& x);
double rootn(const double& x, const int& y);
double sqrt(const double& x);
double half_sqrt(const double& x);
double native_sqrt(const double& x);
double rsqrt(const double& x);
double half_rsqrt(const double& x);
double native_rsqrt(const double& x);
double sin(const double& x);
double sincos(const double& x, double* cosval);
double sinh(const double& x);
double sinpi(const double& x);
double half_sin(const double& x);
double native_sin(const double& x);
double tan(const double& x);
double tanh(const double& x);
double tanpi(const double& x);
double half_tan(const double& x);
double native_tan(const double& x);
double tgamma(const double& x);
double trunc(const double& x);

double2 clamp(const double2& x, const double2& minval, const double2& maxval);
double2 degrees(const double2& radians);
double2 max(const double2& x, const double2& y);
double2 min(const double2& x, const double2& y);
double2 mix(const double2& x, const double2& y, const double2& a);
double2 radians(const double2& degrees);
double2 sign(const double2& x);
double2 smoothstep(const double2& edge0, const double2& edge1, const double2& x);
double2 step(const double2& edge, const double2& x);
double2 acos(const double2& x);
double2 acosh(const double2& x);
double2 acospi(const double2& x);
double2 asin(const double2& x);
double2 asinh(const double2& x);
double2 asinpi(const double2& x);
double2 atan(const double2& y_over_x);
double2 atan2(const double2& y, const double2& x);
double2 atanh(const double2& y_over_x);
double2 atanpi(const double2& y_over_x);
double2 atan2pi(const double2& y, const double2& x);
double2 cbrt(const double2& x);
double2 ceil(const double2& x);
double2 copysign(const double2& x, const double2& y);
double2 cos(const double2& x);
double2 cosh(const double2& xv);
double2 cospi(const double2& x);
double2 half_cos(const double2& x);
double2 native_cos(const double2& x);
double2 half_divide(const double2& x, const double2& y);
double2 native_divide(const double2& x, const double2& y);
double2 erf(const double2& x);
double2 erfc(const double2& x);
double2 exp(const double2& x);
double2 exp2(const double2& x);
double2 exp10(const double2& x);
double2 expm1(const double2& x);
double2 half_exp(const double2& x);
double2 half_exp2(const double2& x);
double2 half_exp10(const double2& x);
double2 native_exp(const double2& x);
double2 native_exp2(const double2& x);
double2 native_exp10(const double2& x);
double2 fabs(const double2& x);
double2 fdim(const double2& x, const double2& y);
double2 floor(const double2& x);
double2 fma(const double2& a, const double2& b, const double2& c);
double2 fmax(const double2& x, const double2& y);
double2 fmin(const double2& x, const double2& y);
double2 fmod(const double2& x);
double2 fract(const double2& x, double2* itptr = 0);
double2 frexp(const double2& x, int2* exp = 0);
double2 hypot(const double2& x, const double2& y);
int2 ilogb(const double2& x);
double2 ldexp(const double2& x, const int2& k);
double2 lgamma(const double2& x);
double2 lgamma_r(const double2& x, int2* signp);
double2 log(const double2& x);
double2 log2(const double2& x);
double2 log10(const double2& x);
double2 log1p(const double2& x);
double2 logb(const double2& x);
double2 half_log(const double2& x);
double2 half_log2(const double2& x);
double2 half_log10(const double2& x);
double2 native_log(const double2& x);
double2 native_log2(const double2& x);
double2 native_log10(const double2& x);
double2 mad(const double2& a, const double2& b, const double2& c);
double2 maxmag(const double2& x, const double2& y);
double2 minmag(const double2& x, const double2& y);
double2 modf(const double2& x, int2* iptr);
double2 nan(const uint2& nancode);
double2 nextafter(const double2& x, const double2& y);
double2 pow(const double2& x, const double2& y);
double2 pown(const double2& x, const int2& y);
double2 powr(const double2& x, const double2& y);
double2 half_powr(const double2& x, const double2& y);
double2 native_powr(const double2& x, const double2& y);
double2 half_recip(const double2& x);
double2 native_recip(const double2& x);
double2 remainder(const double2& x, const double2& y);
double2 remquo(const double2& x, const double2& y, int2* quo = 0);
double2 rint(const double2& x);
double2 round(const double2& x);
double2 rootn(const double2& x, const int2& y);
double2 sqrt(const double2& x);
double2 half_sqrt(const double2& x);
double2 native_sqrt(const double2& x);
double2 rsqrt(const double2& x);
double2 half_rsqrt(const double2& x);
double2 native_rsqrt(const double2& x);
double2 sin(const double2& x);
double2 sincos(const double2& x, double2* cosval);
double2 sinh(const double2& x);
double2 sinpi(const double2& x);
double2 half_sin(const double2& x);
double2 native_sin(const double2& x);
double2 tan(const double2& x);
double2 tanh(const double2& x);
double2 tanpi(const double2& x);
double2 half_tan(const double2& x);
double2 native_tan(const double2& x);
double2 tgamma(const double2& x);
double2 trunc(const double2& x);

double3 clamp(const double3& x, const double3& minval, const double3& maxval);
double3 degrees(const double3& radians);
double3 max(const double3& x, const double3& y);
double3 min(const double3& x, const double3& y);
double3 mix(const double3& x, const double3& y, const double3& a);
double3 radians(const double3& degrees);
double3 sign(const double3& x);
double3 smoothstep(const double3& edge0, const double3& edge1, const double3& x);
double3 step(const double3& edge, const double3& x);
double3 acos(const double3& x);
double3 acosh(const double3& x);
double3 acospi(const double3& x);
double3 asin(const double3& x);
double3 asinh(const double3& x);
double3 asinpi(const double3& x);
double3 atan(const double3& y_over_x);
double3 atan2(const double3& y, const double3& x);
double3 atanh(const double3& y_over_x);
double3 atanpi(const double3& y_over_x);
double3 atan2pi(const double3& y, const double3& x);
double3 cbrt(const double3& x);
double3 ceil(const double3& x);
double3 copysign(const double3& x, const double3& y);
double3 cos(const double3& x);
double3 cosh(const double3& xv);
double3 cospi(const double3& x);
double3 half_cos(const double3& x);
double3 native_cos(const double3& x);
double3 half_divide(const double3& x, const double3& y);
double3 native_divide(const double3& x, const double3& y);
double3 erf(const double3& x);
double3 erfc(const double3& x);
double3 exp(const double3& x);
double3 exp2(const double3& x);
double3 exp10(const double3& x);
double3 expm1(const double3& x);
double3 half_exp(const double3& x);
double3 half_exp2(const double3& x);
double3 half_exp10(const double3& x);
double3 native_exp(const double3& x);
double3 native_exp2(const double3& x);
double3 native_exp10(const double3& x);
double3 fabs(const double3& x);
double3 fdim(const double3& x, const double3& y);
double3 floor(const double3& x);
double3 fma(const double3& a, const double3& b, const double3& c);
double3 fmax(const double3& x, const double3& y);
double3 fmin(const double3& x, const double3& y);
double3 fmod(const double3& x);
double3 fract(const double3& x, double3* itptr = 0);
double3 frexp(const double3& x, int3* exp = 0);
double3 hypot(const double3& x, const double3& y);
int3 ilogb(const double3& x);
double3 ldexp(const double3& x, const int3& k);
double3 lgamma(const double3& x);
double3 lgamma_r(const double3& x, int3* signp);
double3 log(const double3& x);
double3 log2(const double3& x);
double3 log10(const double3& x);
double3 log1p(const double3& x);
double3 logb(const double3& x);
double3 half_log(const double3& x);
double3 half_log2(const double3& x);
double3 half_log10(const double3& x);
double3 native_log(const double3& x);
double3 native_log2(const double3& x);
double3 native_log10(const double3& x);
double3 mad(const double3& a, const double3& b, const double3& c);
double3 maxmag(const double3& x, const double3& y);
double3 minmag(const double3& x, const double3& y);
double3 modf(const double3& x, int3* iptr);
double3 nan(const uint3& nancode);
double3 nextafter(const double3& x, const double3& y);
double3 pow(const double3& x, const double3& y);
double3 pown(const double3& x, const int3& y);
double3 powr(const double3& x, const double3& y);
double3 half_powr(const double3& x, const double3& y);
double3 native_powr(const double3& x, const double3& y);
double3 half_recip(const double3& x);
double3 native_recip(const double3& x);
double3 remainder(const double3& x, const double3& y);
double3 remquo(const double3& x, const double3& y, int3* quo = 0);
double3 rint(const double3& x);
double3 round(const double3& x);
double3 rootn(const double3& x, const int3& y);
double3 sqrt(const double3& x);
double3 half_sqrt(const double3& x);
double3 native_sqrt(const double3& x);
double3 rsqrt(const double3& x);
double3 half_rsqrt(const double3& x);
double3 native_rsqrt(const double3& x);
double3 sin(const double3& x);
double3 sincos(const double3& x, double3* cosval);
double3 sinh(const double3& x);
double3 sinpi(const double3& x);
double3 half_sin(const double3& x);
double3 native_sin(const double3& x);
double3 tan(const double3& x);
double3 tanh(const double3& x);
double3 tanpi(const double3& x);
double3 half_tan(const double3& x);
double3 native_tan(const double3& x);
double3 tgamma(const double3& x);
double3 trunc(const double3& x);

double4 clamp(const double4& x, const double4& minval, const double4& maxval);
double4 degrees(const double4& radians);
double4 max(const double4& x, const double4& y);
double4 min(const double4& x, const double4& y);
double4 mix(const double4& x, const double4& y, const double4& a);
double4 radians(const double4& degrees);
double4 sign(const double4& x);
double4 smoothstep(const double4& edge0, const double4& edge1, const double4& x);
double4 step(const double4& edge, const double4& x);
double4 acos(const double4& x);
double4 acosh(const double4& x);
double4 acospi(const double4& x);
double4 asin(const double4& x);
double4 asinh(const double4& x);
double4 asinpi(const double4& x);
double4 atan(const double4& y_over_x);
double4 atan2(const double4& y, const double4& x);
double4 atanh(const double4& y_over_x);
double4 atanpi(const double4& y_over_x);
double4 atan2pi(const double4& y, const double4& x);
double4 cbrt(const double4& x);
double4 ceil(const double4& x);
double4 copysign(const double4& x, const double4& y);
double4 cos(const double4& x);
double4 cosh(const double4& xv);
double4 cospi(const double4& x);
double4 half_cos(const double4& x);
double4 native_cos(const double4& x);
double4 half_divide(const double4& x, const double4& y);
double4 native_divide(const double4& x, const double4& y);
double4 erf(const double4& x);
double4 erfc(const double4& x);
double4 exp(const double4& x);
double4 exp2(const double4& x);
double4 exp10(const double4& x);
double4 expm1(const double4& x);
double4 half_exp(const double4& x);
double4 half_exp2(const double4& x);
double4 half_exp10(const double4& x);
double4 native_exp(const double4& x);
double4 native_exp2(const double4& x);
double4 native_exp10(const double4& x);
double4 fabs(const double4& x);
double4 fdim(const double4& x, const double4& y);
double4 floor(const double4& x);
double4 fma(const double4& a, const double4& b, const double4& c);
double4 fmax(const double4& x, const double4& y);
double4 fmin(const double4& x, const double4& y);
double4 fmod(const double4& x);
double4 fract(const double4& x, double4* itptr = 0);
double4 frexp(const double4& x, int4* exp = 0);
double4 hypot(const double4& x, const double4& y);
int4 ilogb(const double4& x);
double4 ldexp(const double4& x, const int4& k);
double4 lgamma(const double4& x);
double4 lgamma_r(const double4& x, int4* signp);
double4 log(const double4& x);
double4 log2(const double4& x);
double4 log10(const double4& x);
double4 log1p(const double4& x);
double4 logb(const double4& x);
double4 half_log(const double4& x);
double4 half_log2(const double4& x);
double4 half_log10(const double4& x);
double4 native_log(const double4& x);
double4 native_log2(const double4& x);
double4 native_log10(const double4& x);
double4 mad(const double4& a, const double4& b, const double4& c);
double4 maxmag(const double4& x, const double4& y);
double4 minmag(const double4& x, const double4& y);
double4 modf(const double4& x, int4* iptr);
double4 nan(const uint4& nancode);
double4 nextafter(const double4& x, const double4& y);
double4 pow(const double4& x, const double4& y);
double4 pown(const double4& x, const int4& y);
double4 powr(const double4& x, const double4& y);
double4 half_powr(const double4& x, const double4& y);
double4 native_powr(const double4& x, const double4& y);
double4 half_recip(const double4& x);
double4 native_recip(const double4& x);
double4 remainder(const double4& x, const double4& y);
double4 remquo(const double4& x, const double4& y, int4* quo = 0);
double4 rint(const double4& x);
double4 round(const double4& x);
double4 rootn(const double4& x, const int4& y);
double4 sqrt(const double4& x);
double4 half_sqrt(const double4& x);
double4 native_sqrt(const double4& x);
double4 rsqrt(const double4& x);
double4 half_rsqrt(const double4& x);
double4 native_rsqrt(const double4& x);
double4 sin(const double4& x);
double4 sincos(const double4& x, double4* cosval);
double4 sinh(const double4& x);
double4 sinpi(const double4& x);
double4 half_sin(const double4& x);
double4 native_sin(const double4& x);
double4 tan(const double4& x);
double4 tanh(const double4& x);
double4 tanpi(const double4& x);
double4 half_tan(const double4& x);
double4 native_tan(const double4& x);
double4 tgamma(const double4& x);
double4 trunc(const double4& x);

double8 clamp(const double8& x, const double8& minval, const double8& maxval);
double8 degrees(const double8& radians);
double8 max(const double8& x, const double8& y);
double8 min(const double8& x, const double8& y);
double8 mix(const double8& x, const double8& y, const double8& a);
double8 radians(const double8& degrees);
double8 sign(const double8& x);
double8 smoothstep(const double8& edge0, const double8& edge1, const double8& x);
double8 step(const double8& edge, const double8& x);
double8 acos(const double8& x);
double8 acosh(const double8& x);
double8 acospi(const double8& x);
double8 asin(const double8& x);
double8 asinh(const double8& x);
double8 asinpi(const double8& x);
double8 atan(const double8& y_over_x);
double8 atan2(const double8& y, const double8& x);
double8 atanh(const double8& y_over_x);
double8 atanpi(const double8& y_over_x);
double8 atan2pi(const double8& y, const double8& x);
double8 cbrt(const double8& x);
double8 ceil(const double8& x);
double8 copysign(const double8& x, const double8& y);
double8 cos(const double8& x);
double8 cosh(const double8& xv);
double8 cospi(const double8& x);
double8 half_cos(const double8& x);
double8 native_cos(const double8& x);
double8 half_divide(const double8& x, const double8& y);
double8 native_divide(const double8& x, const double8& y);
double8 erf(const double8& x);
double8 erfc(const double8& x);
double8 exp(const double8& x);
double8 exp2(const double8& x);
double8 exp10(const double8& x);
double8 expm1(const double8& x);
double8 half_exp(const double8& x);
double8 half_exp2(const double8& x);
double8 half_exp10(const double8& x);
double8 native_exp(const double8& x);
double8 native_exp2(const double8& x);
double8 native_exp10(const double8& x);
double8 fabs(const double8& x);
double8 fdim(const double8& x, const double8& y);
double8 floor(const double8& x);
double8 fma(const double8& a, const double8& b, const double8& c);
double8 fmax(const double8& x, const double8& y);
double8 fmin(const double8& x, const double8& y);
double8 fmod(const double8& x);
double8 fract(const double8& x, double8* itptr = 0);
double8 frexp(const double8& x, int8* exp = 0);
double8 hypot(const double8& x, const double8& y);
int8 ilogb(const double8& x);
double8 ldexp(const double8& x, const int8& k);
double8 lgamma(const double8& x);
double8 lgamma_r(const double8& x, int8* signp);
double8 log(const double8& x);
double8 log2(const double8& x);
double8 log10(const double8& x);
double8 log1p(const double8& x);
double8 logb(const double8& x);
double8 half_log(const double8& x);
double8 half_log2(const double8& x);
double8 half_log10(const double8& x);
double8 native_log(const double8& x);
double8 native_log2(const double8& x);
double8 native_log10(const double8& x);
double8 mad(const double8& a, const double8& b, const double8& c);
double8 maxmag(const double8& x, const double8& y);
double8 minmag(const double8& x, const double8& y);
double8 modf(const double8& x, int8* iptr);
double8 nan(const uint8& nancode);
double8 nextafter(const double8& x, const double8& y);
double8 pow(const double8& x, const double8& y);
double8 pown(const double8& x, const int8& y);
double8 powr(const double8& x, const double8& y);
double8 half_powr(const double8& x, const double8& y);
double8 native_powr(const double8& x, const double8& y);
double8 half_recip(const double8& x);
double8 native_recip(const double8& x);
double8 remainder(const double8& x, const double8& y);
double8 remquo(const double8& x, const double8& y, int8* quo = 0);
double8 rint(const double8& x);
double8 round(const double8& x);
double8 rootn(const double8& x, const int8& y);
double8 sqrt(const double8& x);
double8 half_sqrt(const double8& x);
double8 native_sqrt(const double8& x);
double8 rsqrt(const double8& x);
double8 half_rsqrt(const double8& x);
double8 native_rsqrt(const double8& x);
double8 sin(const double8& x);
double8 sincos(const double8& x, double8* cosval);
double8 sinh(const double8& x);
double8 sinpi(const double8& x);
double8 half_sin(const double8& x);
double8 native_sin(const double8& x);
double8 tan(const double8& x);
double8 tanh(const double8& x);
double8 tanpi(const double8& x);
double8 half_tan(const double8& x);
double8 native_tan(const double8& x);
double8 tgamma(const double8& x);
double8 trunc(const double8& x);

double16 clamp(const double16& x, const double16& minval, const double16& maxval);
double16 degrees(const double16& radians);
double16 max(const double16& x, const double16& y);
double16 min(const double16& x, const double16& y);
double16 mix(const double16& x, const double16& y, const double16& a);
double16 radians(const double16& degrees);
double16 sign(const double16& x);
double16 smoothstep(const double16& edge0, const double16& edge1, const double16& x);
double16 step(const double16& edge, const double16& x);
double16 acos(const double16& x);
double16 acosh(const double16& x);
double16 acospi(const double16& x);
double16 asin(const double16& x);
double16 asinh(const double16& x);
double16 asinpi(const double16& x);
double16 atan(const double16& y_over_x);
double16 atan2(const double16& y, const double16& x);
double16 atanh(const double16& y_over_x);
double16 atanpi(const double16& y_over_x);
double16 atan2pi(const double16& y, const double16& x);
double16 cbrt(const double16& x);
double16 ceil(const double16& x);
double16 copysign(const double16& x, const double16& y);
double16 cos(const double16& x);
double16 cosh(const double16& xv);
double16 cospi(const double16& x);
double16 half_cos(const double16& x);
double16 native_cos(const double16& x);
double16 half_divide(const double16& x, const double16& y);
double16 native_divide(const double16& x, const double16& y);
double16 erf(const double16& x);
double16 erfc(const double16& x);
double16 exp(const double16& x);
double16 exp2(const double16& x);
double16 exp10(const double16& x);
double16 expm1(const double16& x);
double16 half_exp(const double16& x);
double16 half_exp2(const double16& x);
double16 half_exp10(const double16& x);
double16 native_exp(const double16& x);
double16 native_exp2(const double16& x);
double16 native_exp10(const double16& x);
double16 fabs(const double16& x);
double16 fdim(const double16& x, const double16& y);
double16 floor(const double16& x);
double16 fma(const double16& a, const double16& b, const double16& c);
double16 fmax(const double16& x, const double16& y);
double16 fmin(const double16& x, const double16& y);
double16 fmod(const double16& x);
double16 fract(const double16& x, double16* itptr = 0);
double16 frexp(const double16& x, int16* exp = 0);
double16 hypot(const double16& x, const double16& y);
int16 ilogb(const double16& x);
double16 ldexp(const double16& x, const int16& k);
double16 lgamma(const double16& x);
double16 lgamma_r(const double16& x, int16* signp);
double16 log(const double16& x);
double16 log2(const double16& x);
double16 log10(const double16& x);
double16 log1p(const double16& x);
double16 logb(const double16& x);
double16 half_log(const double16& x);
double16 half_log2(const double16& x);
double16 half_log10(const double16& x);
double16 native_log(const double16& x);
double16 native_log2(const double16& x);
double16 native_log10(const double16& x);
double16 mad(const double16& a, const double16& b, const double16& c);
double16 maxmag(const double16& x, const double16& y);
double16 minmag(const double16& x, const double16& y);
double16 modf(const double16& x, int16* iptr);
double16 nan(const uint16& nancode);
double16 nextafter(const double16& x, const double16& y);
double16 pow(const double16& x, const double16& y);
double16 pown(const double16& x, const int16& y);
double16 powr(const double16& x, const double16& y);
double16 half_powr(const double16& x, const double16& y);
double16 native_powr(const double16& x, const double16& y);
double16 half_recip(const double16& x);
double16 native_recip(const double16& x);
double16 remainder(const double16& x, const double16& y);
double16 remquo(const double16& x, const double16& y, int16* quo = 0);
double16 rint(const double16& x);
double16 round(const double16& x);
double16 rootn(const double16& x, const int16& y);
double16 sqrt(const double16& x);
double16 half_sqrt(const double16& x);
double16 native_sqrt(const double16& x);
double16 rsqrt(const double16& x);
double16 half_rsqrt(const double16& x);
double16 native_rsqrt(const double16& x);
double16 sin(const double16& x);
double16 sincos(const double16& x, double16* cosval);
double16 sinh(const double16& x);
double16 sinpi(const double16& x);
double16 half_sin(const double16& x);
double16 native_sin(const double16& x);
double16 tan(const double16& x);
double16 tanh(const double16& x);
double16 tanpi(const double16& x);
double16 half_tan(const double16& x);
double16 native_tan(const double16& x);
double16 tgamma(const double16& x);
double16 trunc(const double16& x);

int isequal(const float& x, const float& y);
int isnotequal(const float& x, const float& y);
int isgreater(const float& x, const float& y);
int isgreaterequal(const float& x, const float& y);
int isless(const float& x, const float& y);
int islessequal(const float& x, const float& y);
int islessgreater(const float& x, const float& y);
int isfinite(const float& x);
int isinf(const float& x);
int isnan(const float& x);
int isnormal(const float& x);
int isordered(const float& x, const float& y);
int isunordered(const float& x, const float& y);
int signbit(const float& x);

int2 isequal(const float2& x, const float2& y);
int2 isnotequal(const float2& x, const float2& y);
int2 isgreater(const float2& x, const float2& y);
int2 isgreaterequal(const float2& x, const float2& y);
int2 isless(const float2& x, const float2& y);
int2 islessequal(const float2& x, const float2& y);
int2 islessgreater(const float2& x, const float2& y);
int2 isfinite(const float2& x);
int2 isinf(const float2& x);
int2 isnan(const float2& x);
int2 isnormal(const float2& x);
int2 isordered(const float2& x, const float2& y);
int2 isunordered(const float2& x, const float2& y);
int2 signbit(const float2& x);

int3 isequal(const float3& x, const float3& y);
int3 isnotequal(const float3& x, const float3& y);
int3 isgreater(const float3& x, const float3& y);
int3 isgreaterequal(const float3& x, const float3& y);
int3 isless(const float3& x, const float3& y);
int3 islessequal(const float3& x, const float3& y);
int3 islessgreater(const float3& x, const float3& y);
int3 isfinite(const float3& x);
int3 isinf(const float3& x);
int3 isnan(const float3& x);
int3 isnormal(const float3& x);
int3 isordered(const float3& x, const float3& y);
int3 isunordered(const float3& x, const float3& y);
int3 signbit(const float3& x);

int8 isequal(const float8& x, const float8& y);
int8 isnotequal(const float8& x, const float8& y);
int8 isgreater(const float8& x, const float8& y);
int8 isgreaterequal(const float8& x, const float8& y);
int8 isless(const float8& x, const float8& y);
int8 islessequal(const float8& x, const float8& y);
int8 islessgreater(const float8& x, const float8& y);
int8 isfinite(const float8& x);
int8 isinf(const float8& x);
int8 isnan(const float8& x);
int8 isnormal(const float8& x);
int8 isordered(const float8& x, const float8& y);
int8 isunordered(const float8& x, const float8& y);
int8 signbit(const float8& x);

int16 isequal(const float16& x, const float16& y);
int16 isnotequal(const float16& x, const float16& y);
int16 isgreater(const float16& x, const float16& y);
int16 isgreaterequal(const float16& x, const float16& y);
int16 isless(const float16& x, const float16& y);
int16 islessequal(const float16& x, const float16& y);
int16 islessgreater(const float16& x, const float16& y);
int16 isfinite(const float16& x);
int16 isinf(const float16& x);
int16 isnan(const float16& x);
int16 isnormal(const float16& x);
int16 isordered(const float16& x, const float16& y);
int16 isunordered(const float16& x, const float16& y);
int16 signbit(const float16& x);

int isequal(const double& x, const double& y);
int isnotequal(const double& x, const double& y);
int isgreater(const double& x, const double& y);
int isgreaterequal(const double& x, const double& y);
int isless(const double& x, const double& y);
int islessequal(const double& x, const double& y);
int islessgreater(const double& x, const double& y);
int isfinite(const double& x);
int isinf(const double& x);
int isnan(const double& x);
int isnormal(const double& x);
int isordered(const double& x, const double& y);
int isunordered(const double& x, const double& y);
int signbit(const double& x);

long2 isequal(const double2& x, const double2& y);
long2 isnotequal(const double2& x, const double2& y);
long2 isgreater(const double2& x, const double2& y);
long2 isgreaterequal(const double2& x, const double2& y);
long2 isless(const double2& x, const double2& y);
long2 islessequal(const double2& x, const double2& y);
long2 islessgreater(const double2& x, const double2& y);
long2 isfinite(const double2& x);
long2 isinf(const double2& x);
long2 isnan(const double2& x);
long2 isnormal(const double2& x);
long2 isordered(const double2& x, const double2& y);
long2 isunordered(const double2& x, const double2& y);
long2 signbit(const double2& x);

long3 isequal(const double3& x, const double3& y);
long3 isnotequal(const double3& x, const double3& y);
long3 isgreater(const double3& x, const double3& y);
long3 isgreaterequal(const double3& x, const double3& y);
long3 isless(const double3& x, const double3& y);
long3 islessequal(const double3& x, const double3& y);
long3 islessgreater(const double3& x, const double3& y);
long3 isfinite(const double3& x);
long3 isinf(const double3& x);
long3 isnan(const double3& x);
long3 isnormal(const double3& x);
long3 isordered(const double3& x, const double3& y);
long3 isunordered(const double3& x, const double3& y);
long3 signbit(const double3& x);

long4 isequal(const double4& x, const double4& y);
long4 isnotequal(const double4& x, const double4& y);
long4 isgreater(const double4& x, const double4& y);
long4 isgreaterequal(const double4& x, const double4& y);
long4 isless(const double4& x, const double4& y);
long4 islessequal(const double4& x, const double4& y);
long4 islessgreater(const double4& x, const double4& y);
long4 isfinite(const double4& x);
long4 isinf(const double4& x);
long4 isnan(const double4& x);
long4 isnormal(const double4& x);
long4 isordered(const double4& x, const double4& y);
long4 isunordered(const double4& x, const double4& y);
long4 signbit(const double4& x);

long8 isequal(const double8& x, const double8& y);
long8 isnotequal(const double8& x, const double8& y);
long8 isgreater(const double8& x, const double8& y);
long8 isgreaterequal(const double8& x, const double8& y);
long8 isless(const double8& x, const double8& y);
long8 islessequal(const double8& x, const double8& y);
long8 islessgreater(const double8& x, const double8& y);
long8 isfinite(const double8& x);
long8 isinf(const double8& x);
long8 isnan(const double8& x);
long8 isnormal(const double8& x);
long8 isordered(const double8& x, const double8& y);
long8 isunordered(const double8& x, const double8& y);
long8 signbit(const double8& x);

long16 isequal(const double16& x, const double16& y);
long16 isnotequal(const double16& x, const double16& y);
long16 isgreater(const double16& x, const double16& y);
long16 isgreaterequal(const double16& x, const double16& y);
long16 isless(const double16& x, const double16& y);
long16 islessequal(const double16& x, const double16& y);
long16 islessgreater(const double16& x, const double16& y);
long16 isfinite(const double16& x);
long16 isinf(const double16& x);
long16 isnan(const double16& x);
long16 isnormal(const double16& x);
long16 isordered(const double16& x, const double16& y);
long16 isunordered(const double16& x, const double16& y);
long16 signbit(const double16& x);

int any(const char& x);
int all(const char& x);

int any(const char2& x);
int all(const char2& x);

int any(const char3& x);
int all(const char3& x);

int any(const char4& x);
int all(const char4& x);

int any(const char8& x);
int all(const char8& x);

int any(const char16& x);
int all(const char16& x);

int any(const uchar& x);
int all(const uchar& x);

int any(const uchar2& x);
int all(const uchar2& x);

int any(const uchar3& x);
int all(const uchar3& x);

int any(const uchar4& x);
int all(const uchar4& x);

int any(const uchar8& x);
int all(const uchar8& x);

int any(const uchar16& x);
int all(const uchar16& x);

int any(const short& x);
int all(const short& x);

int any(const short2& x);
int all(const short2& x);

int any(const short3& x);
int all(const short3& x);

int any(const short4& x);
int all(const short4& x);

int any(const short8& x);
int all(const short8& x);

int any(const short16& x);
int all(const short16& x);

int any(const ushort& x);
int all(const ushort& x);

int any(const ushort2& x);
int all(const ushort2& x);

int any(const ushort3& x);
int all(const ushort3& x);

int any(const ushort4& x);
int all(const ushort4& x);

int any(const ushort8& x);
int all(const ushort8& x);

int any(const ushort16& x);
int all(const ushort16& x);

int any(const int& x);
int all(const int& x);

int any(const int2& x);
int all(const int2& x);

int any(const int3& x);
int all(const int3& x);

int any(const int4& x);
int all(const int4& x);

int any(const int8& x);
int all(const int8& x);

int any(const int16& x);
int all(const int16& x);

int any(const uint& x);
int all(const uint& x);

int any(const uint2& x);
int all(const uint2& x);

int any(const uint3& x);
int all(const uint3& x);

int any(const uint4& x);
int all(const uint4& x);

int any(const uint8& x);
int all(const uint8& x);

int any(const uint16& x);
int all(const uint16& x);

int any(const long& x);
int all(const long& x);

int any(const long2& x);
int all(const long2& x);

int any(const long3& x);
int all(const long3& x);

int any(const long4& x);
int all(const long4& x);

int any(const long8& x);
int all(const long8& x);

int any(const long16& x);
int all(const long16& x);

int any(const ulong& x);
int all(const ulong& x);

int any(const ulong2& x);
int all(const ulong2& x);

int any(const ulong3& x);
int all(const ulong3& x);

int any(const ulong4& x);
int all(const ulong4& x);

int any(const ulong8& x);
int all(const ulong8& x);

int any(const ulong16& x);
int all(const ulong16& x);

char bitselect(const char& a, const char& b, const char& c);
char2 bitselect(const char2& a, const char2& b, const char2& c);
char3 bitselect(const char3& a, const char3& b, const char3& c);
char4 bitselect(const char4& a, const char4& b, const char4& c);
char8 bitselect(const char8& a, const char8& b, const char8& c);
char16 bitselect(const char16& a, const char16& b, const char16& c);

uchar bitselect(const uchar& a, const uchar& b, const uchar& c);
uchar2 bitselect(const uchar2& a, const uchar2& b, const uchar2& c);
uchar3 bitselect(const uchar3& a, const uchar3& b, const uchar3& c);
uchar4 bitselect(const uchar4& a, const uchar4& b, const uchar4& c);
uchar8 bitselect(const uchar8& a, const uchar8& b, const uchar8& c);
uchar16 bitselect(const uchar16& a, const uchar16& b, const uchar16& c);

short bitselect(const short& a, const short& b, const short& c);
short2 bitselect(const short2& a, const short2& b, const short2& c);
short3 bitselect(const short3& a, const short3& b, const short3& c);
short4 bitselect(const short4& a, const short4& b, const short4& c);
short8 bitselect(const short8& a, const short8& b, const short8& c);
short16 bitselect(const short16& a, const short16& b, const short16& c);

ushort bitselect(const ushort& a, const ushort& b, const ushort& c);
ushort2 bitselect(const ushort2& a, const ushort2& b, const ushort2& c);
ushort3 bitselect(const ushort3& a, const ushort3& b, const ushort3& c);
ushort4 bitselect(const ushort4& a, const ushort4& b, const ushort4& c);
ushort8 bitselect(const ushort8& a, const ushort8& b, const ushort8& c);
ushort16 bitselect(const ushort16& a, const ushort16& b, const ushort16& c);

int bitselect(const int& a, const int& b, const int& c);
int2 bitselect(const int2& a, const int2& b, const int2& c);
int3 bitselect(const int3& a, const int3& b, const int3& c);
int4 bitselect(const int4& a, const int4& b, const int4& c);
int8 bitselect(const int8& a, const int8& b, const int8& c);
int16 bitselect(const int16& a, const int16& b, const int16& c);

uint bitselect(const uint& a, const uint& b, const uint& c);
uint2 bitselect(const uint2& a, const uint2& b, const uint2& c);
uint3 bitselect(const uint3& a, const uint3& b, const uint3& c);
uint4 bitselect(const uint4& a, const uint4& b, const uint4& c);
uint8 bitselect(const uint8& a, const uint8& b, const uint8& c);
uint16 bitselect(const uint16& a, const uint16& b, const uint16& c);

long bitselect(const long& a, const long& b, const long& c);
long2 bitselect(const long2& a, const long2& b, const long2& c);
long3 bitselect(const long3& a, const long3& b, const long3& c);
long4 bitselect(const long4& a, const long4& b, const long4& c);
long8 bitselect(const long8& a, const long8& b, const long8& c);
long16 bitselect(const long16& a, const long16& b, const long16& c);

ulong bitselect(const ulong& a, const ulong& b, const ulong& c);
ulong2 bitselect(const ulong2& a, const ulong2& b, const ulong2& c);
ulong3 bitselect(const ulong3& a, const ulong3& b, const ulong3& c);
ulong4 bitselect(const ulong4& a, const ulong4& b, const ulong4& c);
ulong8 bitselect(const ulong8& a, const ulong8& b, const ulong8& c);
ulong16 bitselect(const ulong16& a, const ulong16& b, const ulong16& c);

float bitselect(const float& a, const float& b, const float& c);
float2 bitselect(const float2& a, const float2& b, const float2& c);
float3 bitselect(const float3& a, const float3& b, const float3& c);
float4 bitselect(const float4& a, const float4& b, const float4& c);
float8 bitselect(const float8& a, const float8& b, const float8& c);
float16 bitselect(const float16& a, const float16& b, const float16& c);

double bitselect(const double& a, const double& b, const double& c);
double2 bitselect(const double2& a, const double2& b, const double2& c);
double3 bitselect(const double3& a, const double3& b, const double3& c);
double4 bitselect(const double4& a, const double4& b, const double4& c);
double8 bitselect(const double8& a, const double8& b, const double8& c);
double16 bitselect(const double16& a, const double16& b, const double16& c);

float select(const float& a, const float& b, const char& c);
float2 select(const float2& a, const float2& b, const char2& c);
float3 select(const float3& a, const float3& b, const char3& c);
float4 select(const float4& a, const float4& b, const char4& c);
float8 select(const float8& a, const float8& b, const char8& c);
float16 select(const float16& a, const float16& b, const char16& c);

float select(const float& a, const float& b, const uchar& c);
float2 select(const float2& a, const float2& b, const uchar2& c);
float3 select(const float3& a, const float3& b, const uchar3& c);
float4 select(const float4& a, const float4& b, const uchar4& c);
float8 select(const float8& a, const float8& b, const uchar8& c);
float16 select(const float16& a, const float16& b, const uchar16& c);

float select(const float& a, const float& b, const short& c);
float2 select(const float2& a, const float2& b, const short2& c);
float3 select(const float3& a, const float3& b, const short3& c);
float4 select(const float4& a, const float4& b, const short4& c);
float8 select(const float8& a, const float8& b, const short8& c);
float16 select(const float16& a, const float16& b, const short16& c);

float select(const float& a, const float& b, const ushort& c);
float2 select(const float2& a, const float2& b, const ushort2& c);
float3 select(const float3& a, const float3& b, const ushort3& c);
float4 select(const float4& a, const float4& b, const ushort4& c);
float8 select(const float8& a, const float8& b, const ushort8& c);
float16 select(const float16& a, const float16& b, const ushort16& c);

float select(const float& a, const float& b, const int& c);
float2 select(const float2& a, const float2& b, const int2& c);
float3 select(const float3& a, const float3& b, const int3& c);
float4 select(const float4& a, const float4& b, const int4& c);
float8 select(const float8& a, const float8& b, const int8& c);
float16 select(const float16& a, const float16& b, const int16& c);

float select(const float& a, const float& b, const uint& c);
float2 select(const float2& a, const float2& b, const uint2& c);
float3 select(const float3& a, const float3& b, const uint3& c);
float4 select(const float4& a, const float4& b, const uint4& c);
float8 select(const float8& a, const float8& b, const uint8& c);
float16 select(const float16& a, const float16& b, const uint16& c);

float select(const float& a, const float& b, const long& c);
float2 select(const float2& a, const float2& b, const long2& c);
float3 select(const float3& a, const float3& b, const long3& c);
float4 select(const float4& a, const float4& b, const long4& c);
float8 select(const float8& a, const float8& b, const long8& c);
float16 select(const float16& a, const float16& b, const long16& c);

float select(const float& a, const float& b, const ulong& c);
float2 select(const float2& a, const float2& b, const ulong2& c);
float3 select(const float3& a, const float3& b, const ulong3& c);
float4 select(const float4& a, const float4& b, const ulong4& c);
float8 select(const float8& a, const float8& b, const ulong8& c);
float16 select(const float16& a, const float16& b, const ulong16& c);

double select(const double& a, const double& b, const char& c);
double2 select(const double2& a, const double2& b, const char2& c);
double3 select(const double3& a, const double3& b, const char3& c);
double4 select(const double4& a, const double4& b, const char4& c);
double8 select(const double8& a, const double8& b, const char8& c);
double16 select(const double16& a, const double16& b, const char16& c);

double select(const double& a, const double& b, const uchar& c);
double2 select(const double2& a, const double2& b, const uchar2& c);
double3 select(const double3& a, const double3& b, const uchar3& c);
double4 select(const double4& a, const double4& b, const uchar4& c);
double8 select(const double8& a, const double8& b, const uchar8& c);
double16 select(const double16& a, const double16& b, const uchar16& c);

double select(const double& a, const double& b, const short& c);
double2 select(const double2& a, const double2& b, const short2& c);
double3 select(const double3& a, const double3& b, const short3& c);
double4 select(const double4& a, const double4& b, const short4& c);
double8 select(const double8& a, const double8& b, const short8& c);
double16 select(const double16& a, const double16& b, const short16& c);

double select(const double& a, const double& b, const ushort& c);
double2 select(const double2& a, const double2& b, const ushort2& c);
double3 select(const double3& a, const double3& b, const ushort3& c);
double4 select(const double4& a, const double4& b, const ushort4& c);
double8 select(const double8& a, const double8& b, const ushort8& c);
double16 select(const double16& a, const double16& b, const ushort16& c);

double select(const double& a, const double& b, const int& c);
double2 select(const double2& a, const double2& b, const int2& c);
double3 select(const double3& a, const double3& b, const int3& c);
double4 select(const double4& a, const double4& b, const int4& c);
double8 select(const double8& a, const double8& b, const int8& c);
double16 select(const double16& a, const double16& b, const int16& c);

double select(const double& a, const double& b, const uint& c);
double2 select(const double2& a, const double2& b, const uint2& c);
double3 select(const double3& a, const double3& b, const uint3& c);
double4 select(const double4& a, const double4& b, const uint4& c);
double8 select(const double8& a, const double8& b, const uint8& c);
double16 select(const double16& a, const double16& b, const uint16& c);

double select(const double& a, const double& b, const long& c);
double2 select(const double2& a, const double2& b, const long2& c);
double3 select(const double3& a, const double3& b, const long3& c);
double4 select(const double4& a, const double4& b, const long4& c);
double8 select(const double8& a, const double8& b, const long8& c);
double16 select(const double16& a, const double16& b, const long16& c);

double select(const double& a, const double& b, const ulong& c);
double2 select(const double2& a, const double2& b, const ulong2& c);
double3 select(const double3& a, const double3& b, const ulong3& c);
double4 select(const double4& a, const double4& b, const ulong4& c);
double8 select(const double8& a, const double8& b, const ulong8& c);
double16 select(const double16& a, const double16& b, const ulong16& c);

char2 shuffle(const char2& x, const uchar2& mask);
char2 shuffle2(const char2& x, const char2& y, const uchar2& mask);
char3 shuffle(const char3& x, const uchar3& mask);
char3 shuffle2(const char3& x, const char3& y, const uchar3& mask);
char4 shuffle(const char4& x, const uchar4& mask);
char4 shuffle2(const char4& x, const char4& y, const uchar4& mask);
char8 shuffle(const char8& x, const uchar8& mask);
char8 shuffle2(const char8& x, const char8& y, const uchar8& mask);
char16 shuffle(const char16& x, const uchar16& mask);
char16 shuffle2(const char16& x, const char16& y, const uchar16& mask);

char2 shuffle(const char2& x, const uchar2& mask);
char2 shuffle2(const char2& x, const char2& y, const uchar2& mask);
char3 shuffle(const char3& x, const uchar3& mask);
char3 shuffle2(const char3& x, const char3& y, const uchar3& mask);
char4 shuffle(const char4& x, const uchar4& mask);
char4 shuffle2(const char4& x, const char4& y, const uchar4& mask);
char8 shuffle(const char8& x, const uchar8& mask);
char8 shuffle2(const char8& x, const char8& y, const uchar8& mask);
char16 shuffle(const char16& x, const uchar16& mask);
char16 shuffle2(const char16& x, const char16& y, const uchar16& mask);

char2 shuffle(const char2& x, const ushort2& mask);
char2 shuffle2(const char2& x, const char2& y, const ushort2& mask);
char3 shuffle(const char3& x, const ushort3& mask);
char3 shuffle2(const char3& x, const char3& y, const ushort3& mask);
char4 shuffle(const char4& x, const ushort4& mask);
char4 shuffle2(const char4& x, const char4& y, const ushort4& mask);
char8 shuffle(const char8& x, const ushort8& mask);
char8 shuffle2(const char8& x, const char8& y, const ushort8& mask);
char16 shuffle(const char16& x, const ushort16& mask);
char16 shuffle2(const char16& x, const char16& y, const ushort16& mask);

char2 shuffle(const char2& x, const ushort2& mask);
char2 shuffle2(const char2& x, const char2& y, const ushort2& mask);
char3 shuffle(const char3& x, const ushort3& mask);
char3 shuffle2(const char3& x, const char3& y, const ushort3& mask);
char4 shuffle(const char4& x, const ushort4& mask);
char4 shuffle2(const char4& x, const char4& y, const ushort4& mask);
char8 shuffle(const char8& x, const ushort8& mask);
char8 shuffle2(const char8& x, const char8& y, const ushort8& mask);
char16 shuffle(const char16& x, const ushort16& mask);
char16 shuffle2(const char16& x, const char16& y, const ushort16& mask);

char2 shuffle(const char2& x, const uint2& mask);
char2 shuffle2(const char2& x, const char2& y, const uint2& mask);
char3 shuffle(const char3& x, const uint3& mask);
char3 shuffle2(const char3& x, const char3& y, const uint3& mask);
char4 shuffle(const char4& x, const uint4& mask);
char4 shuffle2(const char4& x, const char4& y, const uint4& mask);
char8 shuffle(const char8& x, const uint8& mask);
char8 shuffle2(const char8& x, const char8& y, const uint8& mask);
char16 shuffle(const char16& x, const uint16& mask);
char16 shuffle2(const char16& x, const char16& y, const uint16& mask);

char2 shuffle(const char2& x, const uint2& mask);
char2 shuffle2(const char2& x, const char2& y, const uint2& mask);
char3 shuffle(const char3& x, const uint3& mask);
char3 shuffle2(const char3& x, const char3& y, const uint3& mask);
char4 shuffle(const char4& x, const uint4& mask);
char4 shuffle2(const char4& x, const char4& y, const uint4& mask);
char8 shuffle(const char8& x, const uint8& mask);
char8 shuffle2(const char8& x, const char8& y, const uint8& mask);
char16 shuffle(const char16& x, const uint16& mask);
char16 shuffle2(const char16& x, const char16& y, const uint16& mask);

char2 shuffle(const char2& x, const ulong2& mask);
char2 shuffle2(const char2& x, const char2& y, const ulong2& mask);
char3 shuffle(const char3& x, const ulong3& mask);
char3 shuffle2(const char3& x, const char3& y, const ulong3& mask);
char4 shuffle(const char4& x, const ulong4& mask);
char4 shuffle2(const char4& x, const char4& y, const ulong4& mask);
char8 shuffle(const char8& x, const ulong8& mask);
char8 shuffle2(const char8& x, const char8& y, const ulong8& mask);
char16 shuffle(const char16& x, const ulong16& mask);
char16 shuffle2(const char16& x, const char16& y, const ulong16& mask);

char2 shuffle(const char2& x, const ulong2& mask);
char2 shuffle2(const char2& x, const char2& y, const ulong2& mask);
char3 shuffle(const char3& x, const ulong3& mask);
char3 shuffle2(const char3& x, const char3& y, const ulong3& mask);
char4 shuffle(const char4& x, const ulong4& mask);
char4 shuffle2(const char4& x, const char4& y, const ulong4& mask);
char8 shuffle(const char8& x, const ulong8& mask);
char8 shuffle2(const char8& x, const char8& y, const ulong8& mask);
char16 shuffle(const char16& x, const ulong16& mask);
char16 shuffle2(const char16& x, const char16& y, const ulong16& mask);

uchar2 shuffle(const uchar2& x, const uchar2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const uchar2& mask);
uchar3 shuffle(const uchar3& x, const uchar3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const uchar3& mask);
uchar4 shuffle(const uchar4& x, const uchar4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const uchar4& mask);
uchar8 shuffle(const uchar8& x, const uchar8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const uchar8& mask);
uchar16 shuffle(const uchar16& x, const uchar16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const uchar16& mask);

uchar2 shuffle(const uchar2& x, const uchar2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const uchar2& mask);
uchar3 shuffle(const uchar3& x, const uchar3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const uchar3& mask);
uchar4 shuffle(const uchar4& x, const uchar4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const uchar4& mask);
uchar8 shuffle(const uchar8& x, const uchar8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const uchar8& mask);
uchar16 shuffle(const uchar16& x, const uchar16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const uchar16& mask);

uchar2 shuffle(const uchar2& x, const ushort2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const ushort2& mask);
uchar3 shuffle(const uchar3& x, const ushort3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const ushort3& mask);
uchar4 shuffle(const uchar4& x, const ushort4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const ushort4& mask);
uchar8 shuffle(const uchar8& x, const ushort8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const ushort8& mask);
uchar16 shuffle(const uchar16& x, const ushort16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const ushort16& mask);

uchar2 shuffle(const uchar2& x, const ushort2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const ushort2& mask);
uchar3 shuffle(const uchar3& x, const ushort3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const ushort3& mask);
uchar4 shuffle(const uchar4& x, const ushort4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const ushort4& mask);
uchar8 shuffle(const uchar8& x, const ushort8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const ushort8& mask);
uchar16 shuffle(const uchar16& x, const ushort16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const ushort16& mask);

uchar2 shuffle(const uchar2& x, const uint2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const uint2& mask);
uchar3 shuffle(const uchar3& x, const uint3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const uint3& mask);
uchar4 shuffle(const uchar4& x, const uint4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const uint4& mask);
uchar8 shuffle(const uchar8& x, const uint8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const uint8& mask);
uchar16 shuffle(const uchar16& x, const uint16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const uint16& mask);

uchar2 shuffle(const uchar2& x, const uint2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const uint2& mask);
uchar3 shuffle(const uchar3& x, const uint3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const uint3& mask);
uchar4 shuffle(const uchar4& x, const uint4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const uint4& mask);
uchar8 shuffle(const uchar8& x, const uint8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const uint8& mask);
uchar16 shuffle(const uchar16& x, const uint16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const uint16& mask);

uchar2 shuffle(const uchar2& x, const ulong2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const ulong2& mask);
uchar3 shuffle(const uchar3& x, const ulong3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const ulong3& mask);
uchar4 shuffle(const uchar4& x, const ulong4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const ulong4& mask);
uchar8 shuffle(const uchar8& x, const ulong8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const ulong8& mask);
uchar16 shuffle(const uchar16& x, const ulong16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const ulong16& mask);

uchar2 shuffle(const uchar2& x, const ulong2& mask);
uchar2 shuffle2(const uchar2& x, const uchar2& y, const ulong2& mask);
uchar3 shuffle(const uchar3& x, const ulong3& mask);
uchar3 shuffle2(const uchar3& x, const uchar3& y, const ulong3& mask);
uchar4 shuffle(const uchar4& x, const ulong4& mask);
uchar4 shuffle2(const uchar4& x, const uchar4& y, const ulong4& mask);
uchar8 shuffle(const uchar8& x, const ulong8& mask);
uchar8 shuffle2(const uchar8& x, const uchar8& y, const ulong8& mask);
uchar16 shuffle(const uchar16& x, const ulong16& mask);
uchar16 shuffle2(const uchar16& x, const uchar16& y, const ulong16& mask);

short2 shuffle(const short2& x, const uchar2& mask);
short2 shuffle2(const short2& x, const short2& y, const uchar2& mask);
short3 shuffle(const short3& x, const uchar3& mask);
short3 shuffle2(const short3& x, const short3& y, const uchar3& mask);
short4 shuffle(const short4& x, const uchar4& mask);
short4 shuffle2(const short4& x, const short4& y, const uchar4& mask);
short8 shuffle(const short8& x, const uchar8& mask);
short8 shuffle2(const short8& x, const short8& y, const uchar8& mask);
short16 shuffle(const short16& x, const uchar16& mask);
short16 shuffle2(const short16& x, const short16& y, const uchar16& mask);

short2 shuffle(const short2& x, const uchar2& mask);
short2 shuffle2(const short2& x, const short2& y, const uchar2& mask);
short3 shuffle(const short3& x, const uchar3& mask);
short3 shuffle2(const short3& x, const short3& y, const uchar3& mask);
short4 shuffle(const short4& x, const uchar4& mask);
short4 shuffle2(const short4& x, const short4& y, const uchar4& mask);
short8 shuffle(const short8& x, const uchar8& mask);
short8 shuffle2(const short8& x, const short8& y, const uchar8& mask);
short16 shuffle(const short16& x, const uchar16& mask);
short16 shuffle2(const short16& x, const short16& y, const uchar16& mask);

short2 shuffle(const short2& x, const ushort2& mask);
short2 shuffle2(const short2& x, const short2& y, const ushort2& mask);
short3 shuffle(const short3& x, const ushort3& mask);
short3 shuffle2(const short3& x, const short3& y, const ushort3& mask);
short4 shuffle(const short4& x, const ushort4& mask);
short4 shuffle2(const short4& x, const short4& y, const ushort4& mask);
short8 shuffle(const short8& x, const ushort8& mask);
short8 shuffle2(const short8& x, const short8& y, const ushort8& mask);
short16 shuffle(const short16& x, const ushort16& mask);
short16 shuffle2(const short16& x, const short16& y, const ushort16& mask);

short2 shuffle(const short2& x, const ushort2& mask);
short2 shuffle2(const short2& x, const short2& y, const ushort2& mask);
short3 shuffle(const short3& x, const ushort3& mask);
short3 shuffle2(const short3& x, const short3& y, const ushort3& mask);
short4 shuffle(const short4& x, const ushort4& mask);
short4 shuffle2(const short4& x, const short4& y, const ushort4& mask);
short8 shuffle(const short8& x, const ushort8& mask);
short8 shuffle2(const short8& x, const short8& y, const ushort8& mask);
short16 shuffle(const short16& x, const ushort16& mask);
short16 shuffle2(const short16& x, const short16& y, const ushort16& mask);

short2 shuffle(const short2& x, const uint2& mask);
short2 shuffle2(const short2& x, const short2& y, const uint2& mask);
short3 shuffle(const short3& x, const uint3& mask);
short3 shuffle2(const short3& x, const short3& y, const uint3& mask);
short4 shuffle(const short4& x, const uint4& mask);
short4 shuffle2(const short4& x, const short4& y, const uint4& mask);
short8 shuffle(const short8& x, const uint8& mask);
short8 shuffle2(const short8& x, const short8& y, const uint8& mask);
short16 shuffle(const short16& x, const uint16& mask);
short16 shuffle2(const short16& x, const short16& y, const uint16& mask);

short2 shuffle(const short2& x, const uint2& mask);
short2 shuffle2(const short2& x, const short2& y, const uint2& mask);
short3 shuffle(const short3& x, const uint3& mask);
short3 shuffle2(const short3& x, const short3& y, const uint3& mask);
short4 shuffle(const short4& x, const uint4& mask);
short4 shuffle2(const short4& x, const short4& y, const uint4& mask);
short8 shuffle(const short8& x, const uint8& mask);
short8 shuffle2(const short8& x, const short8& y, const uint8& mask);
short16 shuffle(const short16& x, const uint16& mask);
short16 shuffle2(const short16& x, const short16& y, const uint16& mask);

short2 shuffle(const short2& x, const ulong2& mask);
short2 shuffle2(const short2& x, const short2& y, const ulong2& mask);
short3 shuffle(const short3& x, const ulong3& mask);
short3 shuffle2(const short3& x, const short3& y, const ulong3& mask);
short4 shuffle(const short4& x, const ulong4& mask);
short4 shuffle2(const short4& x, const short4& y, const ulong4& mask);
short8 shuffle(const short8& x, const ulong8& mask);
short8 shuffle2(const short8& x, const short8& y, const ulong8& mask);
short16 shuffle(const short16& x, const ulong16& mask);
short16 shuffle2(const short16& x, const short16& y, const ulong16& mask);

short2 shuffle(const short2& x, const ulong2& mask);
short2 shuffle2(const short2& x, const short2& y, const ulong2& mask);
short3 shuffle(const short3& x, const ulong3& mask);
short3 shuffle2(const short3& x, const short3& y, const ulong3& mask);
short4 shuffle(const short4& x, const ulong4& mask);
short4 shuffle2(const short4& x, const short4& y, const ulong4& mask);
short8 shuffle(const short8& x, const ulong8& mask);
short8 shuffle2(const short8& x, const short8& y, const ulong8& mask);
short16 shuffle(const short16& x, const ulong16& mask);
short16 shuffle2(const short16& x, const short16& y, const ulong16& mask);

ushort2 shuffle(const ushort2& x, const uchar2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const uchar2& mask);
ushort3 shuffle(const ushort3& x, const uchar3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const uchar3& mask);
ushort4 shuffle(const ushort4& x, const uchar4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const uchar4& mask);
ushort8 shuffle(const ushort8& x, const uchar8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const uchar8& mask);
ushort16 shuffle(const ushort16& x, const uchar16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const uchar16& mask);

ushort2 shuffle(const ushort2& x, const uchar2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const uchar2& mask);
ushort3 shuffle(const ushort3& x, const uchar3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const uchar3& mask);
ushort4 shuffle(const ushort4& x, const uchar4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const uchar4& mask);
ushort8 shuffle(const ushort8& x, const uchar8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const uchar8& mask);
ushort16 shuffle(const ushort16& x, const uchar16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const uchar16& mask);

ushort2 shuffle(const ushort2& x, const ushort2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const ushort2& mask);
ushort3 shuffle(const ushort3& x, const ushort3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const ushort3& mask);
ushort4 shuffle(const ushort4& x, const ushort4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const ushort4& mask);
ushort8 shuffle(const ushort8& x, const ushort8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const ushort8& mask);
ushort16 shuffle(const ushort16& x, const ushort16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const ushort16& mask);

ushort2 shuffle(const ushort2& x, const ushort2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const ushort2& mask);
ushort3 shuffle(const ushort3& x, const ushort3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const ushort3& mask);
ushort4 shuffle(const ushort4& x, const ushort4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const ushort4& mask);
ushort8 shuffle(const ushort8& x, const ushort8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const ushort8& mask);
ushort16 shuffle(const ushort16& x, const ushort16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const ushort16& mask);

ushort2 shuffle(const ushort2& x, const uint2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const uint2& mask);
ushort3 shuffle(const ushort3& x, const uint3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const uint3& mask);
ushort4 shuffle(const ushort4& x, const uint4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const uint4& mask);
ushort8 shuffle(const ushort8& x, const uint8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const uint8& mask);
ushort16 shuffle(const ushort16& x, const uint16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const uint16& mask);

ushort2 shuffle(const ushort2& x, const uint2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const uint2& mask);
ushort3 shuffle(const ushort3& x, const uint3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const uint3& mask);
ushort4 shuffle(const ushort4& x, const uint4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const uint4& mask);
ushort8 shuffle(const ushort8& x, const uint8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const uint8& mask);
ushort16 shuffle(const ushort16& x, const uint16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const uint16& mask);

ushort2 shuffle(const ushort2& x, const ulong2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const ulong2& mask);
ushort3 shuffle(const ushort3& x, const ulong3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const ulong3& mask);
ushort4 shuffle(const ushort4& x, const ulong4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const ulong4& mask);
ushort8 shuffle(const ushort8& x, const ulong8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const ulong8& mask);
ushort16 shuffle(const ushort16& x, const ulong16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const ulong16& mask);

ushort2 shuffle(const ushort2& x, const ulong2& mask);
ushort2 shuffle2(const ushort2& x, const ushort2& y, const ulong2& mask);
ushort3 shuffle(const ushort3& x, const ulong3& mask);
ushort3 shuffle2(const ushort3& x, const ushort3& y, const ulong3& mask);
ushort4 shuffle(const ushort4& x, const ulong4& mask);
ushort4 shuffle2(const ushort4& x, const ushort4& y, const ulong4& mask);
ushort8 shuffle(const ushort8& x, const ulong8& mask);
ushort8 shuffle2(const ushort8& x, const ushort8& y, const ulong8& mask);
ushort16 shuffle(const ushort16& x, const ulong16& mask);
ushort16 shuffle2(const ushort16& x, const ushort16& y, const ulong16& mask);

int2 shuffle(const int2& x, const uchar2& mask);
int2 shuffle2(const int2& x, const int2& y, const uchar2& mask);
int3 shuffle(const int3& x, const uchar3& mask);
int3 shuffle2(const int3& x, const int3& y, const uchar3& mask);
int4 shuffle(const int4& x, const uchar4& mask);
int4 shuffle2(const int4& x, const int4& y, const uchar4& mask);
int8 shuffle(const int8& x, const uchar8& mask);
int8 shuffle2(const int8& x, const int8& y, const uchar8& mask);
int16 shuffle(const int16& x, const uchar16& mask);
int16 shuffle2(const int16& x, const int16& y, const uchar16& mask);

int2 shuffle(const int2& x, const uchar2& mask);
int2 shuffle2(const int2& x, const int2& y, const uchar2& mask);
int3 shuffle(const int3& x, const uchar3& mask);
int3 shuffle2(const int3& x, const int3& y, const uchar3& mask);
int4 shuffle(const int4& x, const uchar4& mask);
int4 shuffle2(const int4& x, const int4& y, const uchar4& mask);
int8 shuffle(const int8& x, const uchar8& mask);
int8 shuffle2(const int8& x, const int8& y, const uchar8& mask);
int16 shuffle(const int16& x, const uchar16& mask);
int16 shuffle2(const int16& x, const int16& y, const uchar16& mask);

int2 shuffle(const int2& x, const ushort2& mask);
int2 shuffle2(const int2& x, const int2& y, const ushort2& mask);
int3 shuffle(const int3& x, const ushort3& mask);
int3 shuffle2(const int3& x, const int3& y, const ushort3& mask);
int4 shuffle(const int4& x, const ushort4& mask);
int4 shuffle2(const int4& x, const int4& y, const ushort4& mask);
int8 shuffle(const int8& x, const ushort8& mask);
int8 shuffle2(const int8& x, const int8& y, const ushort8& mask);
int16 shuffle(const int16& x, const ushort16& mask);
int16 shuffle2(const int16& x, const int16& y, const ushort16& mask);

int2 shuffle(const int2& x, const ushort2& mask);
int2 shuffle2(const int2& x, const int2& y, const ushort2& mask);
int3 shuffle(const int3& x, const ushort3& mask);
int3 shuffle2(const int3& x, const int3& y, const ushort3& mask);
int4 shuffle(const int4& x, const ushort4& mask);
int4 shuffle2(const int4& x, const int4& y, const ushort4& mask);
int8 shuffle(const int8& x, const ushort8& mask);
int8 shuffle2(const int8& x, const int8& y, const ushort8& mask);
int16 shuffle(const int16& x, const ushort16& mask);
int16 shuffle2(const int16& x, const int16& y, const ushort16& mask);

int2 shuffle(const int2& x, const uint2& mask);
int2 shuffle2(const int2& x, const int2& y, const uint2& mask);
int3 shuffle(const int3& x, const uint3& mask);
int3 shuffle2(const int3& x, const int3& y, const uint3& mask);
int4 shuffle(const int4& x, const uint4& mask);
int4 shuffle2(const int4& x, const int4& y, const uint4& mask);
int8 shuffle(const int8& x, const uint8& mask);
int8 shuffle2(const int8& x, const int8& y, const uint8& mask);
int16 shuffle(const int16& x, const uint16& mask);
int16 shuffle2(const int16& x, const int16& y, const uint16& mask);

int2 shuffle(const int2& x, const uint2& mask);
int2 shuffle2(const int2& x, const int2& y, const uint2& mask);
int3 shuffle(const int3& x, const uint3& mask);
int3 shuffle2(const int3& x, const int3& y, const uint3& mask);
int4 shuffle(const int4& x, const uint4& mask);
int4 shuffle2(const int4& x, const int4& y, const uint4& mask);
int8 shuffle(const int8& x, const uint8& mask);
int8 shuffle2(const int8& x, const int8& y, const uint8& mask);
int16 shuffle(const int16& x, const uint16& mask);
int16 shuffle2(const int16& x, const int16& y, const uint16& mask);

int2 shuffle(const int2& x, const ulong2& mask);
int2 shuffle2(const int2& x, const int2& y, const ulong2& mask);
int3 shuffle(const int3& x, const ulong3& mask);
int3 shuffle2(const int3& x, const int3& y, const ulong3& mask);
int4 shuffle(const int4& x, const ulong4& mask);
int4 shuffle2(const int4& x, const int4& y, const ulong4& mask);
int8 shuffle(const int8& x, const ulong8& mask);
int8 shuffle2(const int8& x, const int8& y, const ulong8& mask);
int16 shuffle(const int16& x, const ulong16& mask);
int16 shuffle2(const int16& x, const int16& y, const ulong16& mask);

int2 shuffle(const int2& x, const ulong2& mask);
int2 shuffle2(const int2& x, const int2& y, const ulong2& mask);
int3 shuffle(const int3& x, const ulong3& mask);
int3 shuffle2(const int3& x, const int3& y, const ulong3& mask);
int4 shuffle(const int4& x, const ulong4& mask);
int4 shuffle2(const int4& x, const int4& y, const ulong4& mask);
int8 shuffle(const int8& x, const ulong8& mask);
int8 shuffle2(const int8& x, const int8& y, const ulong8& mask);
int16 shuffle(const int16& x, const ulong16& mask);
int16 shuffle2(const int16& x, const int16& y, const ulong16& mask);

uint2 shuffle(const uint2& x, const uchar2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const uchar2& mask);
uint3 shuffle(const uint3& x, const uchar3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const uchar3& mask);
uint4 shuffle(const uint4& x, const uchar4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const uchar4& mask);
uint8 shuffle(const uint8& x, const uchar8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const uchar8& mask);
uint16 shuffle(const uint16& x, const uchar16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const uchar16& mask);

uint2 shuffle(const uint2& x, const uchar2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const uchar2& mask);
uint3 shuffle(const uint3& x, const uchar3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const uchar3& mask);
uint4 shuffle(const uint4& x, const uchar4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const uchar4& mask);
uint8 shuffle(const uint8& x, const uchar8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const uchar8& mask);
uint16 shuffle(const uint16& x, const uchar16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const uchar16& mask);

uint2 shuffle(const uint2& x, const ushort2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const ushort2& mask);
uint3 shuffle(const uint3& x, const ushort3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const ushort3& mask);
uint4 shuffle(const uint4& x, const ushort4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const ushort4& mask);
uint8 shuffle(const uint8& x, const ushort8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const ushort8& mask);
uint16 shuffle(const uint16& x, const ushort16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const ushort16& mask);

uint2 shuffle(const uint2& x, const ushort2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const ushort2& mask);
uint3 shuffle(const uint3& x, const ushort3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const ushort3& mask);
uint4 shuffle(const uint4& x, const ushort4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const ushort4& mask);
uint8 shuffle(const uint8& x, const ushort8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const ushort8& mask);
uint16 shuffle(const uint16& x, const ushort16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const ushort16& mask);

uint2 shuffle(const uint2& x, const uint2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const uint2& mask);
uint3 shuffle(const uint3& x, const uint3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const uint3& mask);
uint4 shuffle(const uint4& x, const uint4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const uint4& mask);
uint8 shuffle(const uint8& x, const uint8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const uint8& mask);
uint16 shuffle(const uint16& x, const uint16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const uint16& mask);

uint2 shuffle(const uint2& x, const uint2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const uint2& mask);
uint3 shuffle(const uint3& x, const uint3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const uint3& mask);
uint4 shuffle(const uint4& x, const uint4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const uint4& mask);
uint8 shuffle(const uint8& x, const uint8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const uint8& mask);
uint16 shuffle(const uint16& x, const uint16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const uint16& mask);

uint2 shuffle(const uint2& x, const ulong2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const ulong2& mask);
uint3 shuffle(const uint3& x, const ulong3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const ulong3& mask);
uint4 shuffle(const uint4& x, const ulong4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const ulong4& mask);
uint8 shuffle(const uint8& x, const ulong8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const ulong8& mask);
uint16 shuffle(const uint16& x, const ulong16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const ulong16& mask);

uint2 shuffle(const uint2& x, const ulong2& mask);
uint2 shuffle2(const uint2& x, const uint2& y, const ulong2& mask);
uint3 shuffle(const uint3& x, const ulong3& mask);
uint3 shuffle2(const uint3& x, const uint3& y, const ulong3& mask);
uint4 shuffle(const uint4& x, const ulong4& mask);
uint4 shuffle2(const uint4& x, const uint4& y, const ulong4& mask);
uint8 shuffle(const uint8& x, const ulong8& mask);
uint8 shuffle2(const uint8& x, const uint8& y, const ulong8& mask);
uint16 shuffle(const uint16& x, const ulong16& mask);
uint16 shuffle2(const uint16& x, const uint16& y, const ulong16& mask);

long2 shuffle(const long2& x, const uchar2& mask);
long2 shuffle2(const long2& x, const long2& y, const uchar2& mask);
long3 shuffle(const long3& x, const uchar3& mask);
long3 shuffle2(const long3& x, const long3& y, const uchar3& mask);
long4 shuffle(const long4& x, const uchar4& mask);
long4 shuffle2(const long4& x, const long4& y, const uchar4& mask);
long8 shuffle(const long8& x, const uchar8& mask);
long8 shuffle2(const long8& x, const long8& y, const uchar8& mask);
long16 shuffle(const long16& x, const uchar16& mask);
long16 shuffle2(const long16& x, const long16& y, const uchar16& mask);

long2 shuffle(const long2& x, const uchar2& mask);
long2 shuffle2(const long2& x, const long2& y, const uchar2& mask);
long3 shuffle(const long3& x, const uchar3& mask);
long3 shuffle2(const long3& x, const long3& y, const uchar3& mask);
long4 shuffle(const long4& x, const uchar4& mask);
long4 shuffle2(const long4& x, const long4& y, const uchar4& mask);
long8 shuffle(const long8& x, const uchar8& mask);
long8 shuffle2(const long8& x, const long8& y, const uchar8& mask);
long16 shuffle(const long16& x, const uchar16& mask);
long16 shuffle2(const long16& x, const long16& y, const uchar16& mask);

long2 shuffle(const long2& x, const ushort2& mask);
long2 shuffle2(const long2& x, const long2& y, const ushort2& mask);
long3 shuffle(const long3& x, const ushort3& mask);
long3 shuffle2(const long3& x, const long3& y, const ushort3& mask);
long4 shuffle(const long4& x, const ushort4& mask);
long4 shuffle2(const long4& x, const long4& y, const ushort4& mask);
long8 shuffle(const long8& x, const ushort8& mask);
long8 shuffle2(const long8& x, const long8& y, const ushort8& mask);
long16 shuffle(const long16& x, const ushort16& mask);
long16 shuffle2(const long16& x, const long16& y, const ushort16& mask);

long2 shuffle(const long2& x, const ushort2& mask);
long2 shuffle2(const long2& x, const long2& y, const ushort2& mask);
long3 shuffle(const long3& x, const ushort3& mask);
long3 shuffle2(const long3& x, const long3& y, const ushort3& mask);
long4 shuffle(const long4& x, const ushort4& mask);
long4 shuffle2(const long4& x, const long4& y, const ushort4& mask);
long8 shuffle(const long8& x, const ushort8& mask);
long8 shuffle2(const long8& x, const long8& y, const ushort8& mask);
long16 shuffle(const long16& x, const ushort16& mask);
long16 shuffle2(const long16& x, const long16& y, const ushort16& mask);

long2 shuffle(const long2& x, const uint2& mask);
long2 shuffle2(const long2& x, const long2& y, const uint2& mask);
long3 shuffle(const long3& x, const uint3& mask);
long3 shuffle2(const long3& x, const long3& y, const uint3& mask);
long4 shuffle(const long4& x, const uint4& mask);
long4 shuffle2(const long4& x, const long4& y, const uint4& mask);
long8 shuffle(const long8& x, const uint8& mask);
long8 shuffle2(const long8& x, const long8& y, const uint8& mask);
long16 shuffle(const long16& x, const uint16& mask);
long16 shuffle2(const long16& x, const long16& y, const uint16& mask);

long2 shuffle(const long2& x, const uint2& mask);
long2 shuffle2(const long2& x, const long2& y, const uint2& mask);
long3 shuffle(const long3& x, const uint3& mask);
long3 shuffle2(const long3& x, const long3& y, const uint3& mask);
long4 shuffle(const long4& x, const uint4& mask);
long4 shuffle2(const long4& x, const long4& y, const uint4& mask);
long8 shuffle(const long8& x, const uint8& mask);
long8 shuffle2(const long8& x, const long8& y, const uint8& mask);
long16 shuffle(const long16& x, const uint16& mask);
long16 shuffle2(const long16& x, const long16& y, const uint16& mask);

long2 shuffle(const long2& x, const ulong2& mask);
long2 shuffle2(const long2& x, const long2& y, const ulong2& mask);
long3 shuffle(const long3& x, const ulong3& mask);
long3 shuffle2(const long3& x, const long3& y, const ulong3& mask);
long4 shuffle(const long4& x, const ulong4& mask);
long4 shuffle2(const long4& x, const long4& y, const ulong4& mask);
long8 shuffle(const long8& x, const ulong8& mask);
long8 shuffle2(const long8& x, const long8& y, const ulong8& mask);
long16 shuffle(const long16& x, const ulong16& mask);
long16 shuffle2(const long16& x, const long16& y, const ulong16& mask);

long2 shuffle(const long2& x, const ulong2& mask);
long2 shuffle2(const long2& x, const long2& y, const ulong2& mask);
long3 shuffle(const long3& x, const ulong3& mask);
long3 shuffle2(const long3& x, const long3& y, const ulong3& mask);
long4 shuffle(const long4& x, const ulong4& mask);
long4 shuffle2(const long4& x, const long4& y, const ulong4& mask);
long8 shuffle(const long8& x, const ulong8& mask);
long8 shuffle2(const long8& x, const long8& y, const ulong8& mask);
long16 shuffle(const long16& x, const ulong16& mask);
long16 shuffle2(const long16& x, const long16& y, const ulong16& mask);

ulong2 shuffle(const ulong2& x, const uchar2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const uchar2& mask);
ulong3 shuffle(const ulong3& x, const uchar3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const uchar3& mask);
ulong4 shuffle(const ulong4& x, const uchar4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const uchar4& mask);
ulong8 shuffle(const ulong8& x, const uchar8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const uchar8& mask);
ulong16 shuffle(const ulong16& x, const uchar16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const uchar16& mask);

ulong2 shuffle(const ulong2& x, const uchar2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const uchar2& mask);
ulong3 shuffle(const ulong3& x, const uchar3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const uchar3& mask);
ulong4 shuffle(const ulong4& x, const uchar4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const uchar4& mask);
ulong8 shuffle(const ulong8& x, const uchar8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const uchar8& mask);
ulong16 shuffle(const ulong16& x, const uchar16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const uchar16& mask);

ulong2 shuffle(const ulong2& x, const ushort2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const ushort2& mask);
ulong3 shuffle(const ulong3& x, const ushort3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const ushort3& mask);
ulong4 shuffle(const ulong4& x, const ushort4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const ushort4& mask);
ulong8 shuffle(const ulong8& x, const ushort8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const ushort8& mask);
ulong16 shuffle(const ulong16& x, const ushort16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const ushort16& mask);

ulong2 shuffle(const ulong2& x, const ushort2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const ushort2& mask);
ulong3 shuffle(const ulong3& x, const ushort3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const ushort3& mask);
ulong4 shuffle(const ulong4& x, const ushort4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const ushort4& mask);
ulong8 shuffle(const ulong8& x, const ushort8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const ushort8& mask);
ulong16 shuffle(const ulong16& x, const ushort16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const ushort16& mask);

ulong2 shuffle(const ulong2& x, const uint2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const uint2& mask);
ulong3 shuffle(const ulong3& x, const uint3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const uint3& mask);
ulong4 shuffle(const ulong4& x, const uint4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const uint4& mask);
ulong8 shuffle(const ulong8& x, const uint8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const uint8& mask);
ulong16 shuffle(const ulong16& x, const uint16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const uint16& mask);

ulong2 shuffle(const ulong2& x, const uint2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const uint2& mask);
ulong3 shuffle(const ulong3& x, const uint3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const uint3& mask);
ulong4 shuffle(const ulong4& x, const uint4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const uint4& mask);
ulong8 shuffle(const ulong8& x, const uint8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const uint8& mask);
ulong16 shuffle(const ulong16& x, const uint16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const uint16& mask);

ulong2 shuffle(const ulong2& x, const ulong2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const ulong2& mask);
ulong3 shuffle(const ulong3& x, const ulong3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const ulong3& mask);
ulong4 shuffle(const ulong4& x, const ulong4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const ulong4& mask);
ulong8 shuffle(const ulong8& x, const ulong8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const ulong8& mask);
ulong16 shuffle(const ulong16& x, const ulong16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const ulong16& mask);

ulong2 shuffle(const ulong2& x, const ulong2& mask);
ulong2 shuffle2(const ulong2& x, const ulong2& y, const ulong2& mask);
ulong3 shuffle(const ulong3& x, const ulong3& mask);
ulong3 shuffle2(const ulong3& x, const ulong3& y, const ulong3& mask);
ulong4 shuffle(const ulong4& x, const ulong4& mask);
ulong4 shuffle2(const ulong4& x, const ulong4& y, const ulong4& mask);
ulong8 shuffle(const ulong8& x, const ulong8& mask);
ulong8 shuffle2(const ulong8& x, const ulong8& y, const ulong8& mask);
ulong16 shuffle(const ulong16& x, const ulong16& mask);
ulong16 shuffle2(const ulong16& x, const ulong16& y, const ulong16& mask);

float2 shuffle(const float2& x, const uchar2& mask);
float2 shuffle2(const float2& x, const float2& y, const uchar2& mask);
float3 shuffle(const float3& x, const uchar3& mask);
float3 shuffle2(const float3& x, const float3& y, const uchar3& mask);
float4 shuffle(const float4& x, const uchar4& mask);
float4 shuffle2(const float4& x, const float4& y, const uchar4& mask);
float8 shuffle(const float8& x, const uchar8& mask);
float8 shuffle2(const float8& x, const float8& y, const uchar8& mask);
float16 shuffle(const float16& x, const uchar16& mask);
float16 shuffle2(const float16& x, const float16& y, const uchar16& mask);

float2 shuffle(const float2& x, const uchar2& mask);
float2 shuffle2(const float2& x, const float2& y, const uchar2& mask);
float3 shuffle(const float3& x, const uchar3& mask);
float3 shuffle2(const float3& x, const float3& y, const uchar3& mask);
float4 shuffle(const float4& x, const uchar4& mask);
float4 shuffle2(const float4& x, const float4& y, const uchar4& mask);
float8 shuffle(const float8& x, const uchar8& mask);
float8 shuffle2(const float8& x, const float8& y, const uchar8& mask);
float16 shuffle(const float16& x, const uchar16& mask);
float16 shuffle2(const float16& x, const float16& y, const uchar16& mask);

float2 shuffle(const float2& x, const ushort2& mask);
float2 shuffle2(const float2& x, const float2& y, const ushort2& mask);
float3 shuffle(const float3& x, const ushort3& mask);
float3 shuffle2(const float3& x, const float3& y, const ushort3& mask);
float4 shuffle(const float4& x, const ushort4& mask);
float4 shuffle2(const float4& x, const float4& y, const ushort4& mask);
float8 shuffle(const float8& x, const ushort8& mask);
float8 shuffle2(const float8& x, const float8& y, const ushort8& mask);
float16 shuffle(const float16& x, const ushort16& mask);
float16 shuffle2(const float16& x, const float16& y, const ushort16& mask);

float2 shuffle(const float2& x, const ushort2& mask);
float2 shuffle2(const float2& x, const float2& y, const ushort2& mask);
float3 shuffle(const float3& x, const ushort3& mask);
float3 shuffle2(const float3& x, const float3& y, const ushort3& mask);
float4 shuffle(const float4& x, const ushort4& mask);
float4 shuffle2(const float4& x, const float4& y, const ushort4& mask);
float8 shuffle(const float8& x, const ushort8& mask);
float8 shuffle2(const float8& x, const float8& y, const ushort8& mask);
float16 shuffle(const float16& x, const ushort16& mask);
float16 shuffle2(const float16& x, const float16& y, const ushort16& mask);

float2 shuffle(const float2& x, const uint2& mask);
float2 shuffle2(const float2& x, const float2& y, const uint2& mask);
float3 shuffle(const float3& x, const uint3& mask);
float3 shuffle2(const float3& x, const float3& y, const uint3& mask);
float4 shuffle(const float4& x, const uint4& mask);
float4 shuffle2(const float4& x, const float4& y, const uint4& mask);
float8 shuffle(const float8& x, const uint8& mask);
float8 shuffle2(const float8& x, const float8& y, const uint8& mask);
float16 shuffle(const float16& x, const uint16& mask);
float16 shuffle2(const float16& x, const float16& y, const uint16& mask);

float2 shuffle(const float2& x, const uint2& mask);
float2 shuffle2(const float2& x, const float2& y, const uint2& mask);
float3 shuffle(const float3& x, const uint3& mask);
float3 shuffle2(const float3& x, const float3& y, const uint3& mask);
float4 shuffle(const float4& x, const uint4& mask);
float4 shuffle2(const float4& x, const float4& y, const uint4& mask);
float8 shuffle(const float8& x, const uint8& mask);
float8 shuffle2(const float8& x, const float8& y, const uint8& mask);
float16 shuffle(const float16& x, const uint16& mask);
float16 shuffle2(const float16& x, const float16& y, const uint16& mask);

float2 shuffle(const float2& x, const ulong2& mask);
float2 shuffle2(const float2& x, const float2& y, const ulong2& mask);
float3 shuffle(const float3& x, const ulong3& mask);
float3 shuffle2(const float3& x, const float3& y, const ulong3& mask);
float4 shuffle(const float4& x, const ulong4& mask);
float4 shuffle2(const float4& x, const float4& y, const ulong4& mask);
float8 shuffle(const float8& x, const ulong8& mask);
float8 shuffle2(const float8& x, const float8& y, const ulong8& mask);
float16 shuffle(const float16& x, const ulong16& mask);
float16 shuffle2(const float16& x, const float16& y, const ulong16& mask);

float2 shuffle(const float2& x, const ulong2& mask);
float2 shuffle2(const float2& x, const float2& y, const ulong2& mask);
float3 shuffle(const float3& x, const ulong3& mask);
float3 shuffle2(const float3& x, const float3& y, const ulong3& mask);
float4 shuffle(const float4& x, const ulong4& mask);
float4 shuffle2(const float4& x, const float4& y, const ulong4& mask);
float8 shuffle(const float8& x, const ulong8& mask);
float8 shuffle2(const float8& x, const float8& y, const ulong8& mask);
float16 shuffle(const float16& x, const ulong16& mask);
float16 shuffle2(const float16& x, const float16& y, const ulong16& mask);

double2 shuffle(const double2& x, const uchar2& mask);
double2 shuffle2(const double2& x, const double2& y, const uchar2& mask);
double3 shuffle(const double3& x, const uchar3& mask);
double3 shuffle2(const double3& x, const double3& y, const uchar3& mask);
double4 shuffle(const double4& x, const uchar4& mask);
double4 shuffle2(const double4& x, const double4& y, const uchar4& mask);
double8 shuffle(const double8& x, const uchar8& mask);
double8 shuffle2(const double8& x, const double8& y, const uchar8& mask);
double16 shuffle(const double16& x, const uchar16& mask);
double16 shuffle2(const double16& x, const double16& y, const uchar16& mask);

double2 shuffle(const double2& x, const uchar2& mask);
double2 shuffle2(const double2& x, const double2& y, const uchar2& mask);
double3 shuffle(const double3& x, const uchar3& mask);
double3 shuffle2(const double3& x, const double3& y, const uchar3& mask);
double4 shuffle(const double4& x, const uchar4& mask);
double4 shuffle2(const double4& x, const double4& y, const uchar4& mask);
double8 shuffle(const double8& x, const uchar8& mask);
double8 shuffle2(const double8& x, const double8& y, const uchar8& mask);
double16 shuffle(const double16& x, const uchar16& mask);
double16 shuffle2(const double16& x, const double16& y, const uchar16& mask);

double2 shuffle(const double2& x, const ushort2& mask);
double2 shuffle2(const double2& x, const double2& y, const ushort2& mask);
double3 shuffle(const double3& x, const ushort3& mask);
double3 shuffle2(const double3& x, const double3& y, const ushort3& mask);
double4 shuffle(const double4& x, const ushort4& mask);
double4 shuffle2(const double4& x, const double4& y, const ushort4& mask);
double8 shuffle(const double8& x, const ushort8& mask);
double8 shuffle2(const double8& x, const double8& y, const ushort8& mask);
double16 shuffle(const double16& x, const ushort16& mask);
double16 shuffle2(const double16& x, const double16& y, const ushort16& mask);

double2 shuffle(const double2& x, const ushort2& mask);
double2 shuffle2(const double2& x, const double2& y, const ushort2& mask);
double3 shuffle(const double3& x, const ushort3& mask);
double3 shuffle2(const double3& x, const double3& y, const ushort3& mask);
double4 shuffle(const double4& x, const ushort4& mask);
double4 shuffle2(const double4& x, const double4& y, const ushort4& mask);
double8 shuffle(const double8& x, const ushort8& mask);
double8 shuffle2(const double8& x, const double8& y, const ushort8& mask);
double16 shuffle(const double16& x, const ushort16& mask);
double16 shuffle2(const double16& x, const double16& y, const ushort16& mask);

double2 shuffle(const double2& x, const uint2& mask);
double2 shuffle2(const double2& x, const double2& y, const uint2& mask);
double3 shuffle(const double3& x, const uint3& mask);
double3 shuffle2(const double3& x, const double3& y, const uint3& mask);
double4 shuffle(const double4& x, const uint4& mask);
double4 shuffle2(const double4& x, const double4& y, const uint4& mask);
double8 shuffle(const double8& x, const uint8& mask);
double8 shuffle2(const double8& x, const double8& y, const uint8& mask);
double16 shuffle(const double16& x, const uint16& mask);
double16 shuffle2(const double16& x, const double16& y, const uint16& mask);

double2 shuffle(const double2& x, const uint2& mask);
double2 shuffle2(const double2& x, const double2& y, const uint2& mask);
double3 shuffle(const double3& x, const uint3& mask);
double3 shuffle2(const double3& x, const double3& y, const uint3& mask);
double4 shuffle(const double4& x, const uint4& mask);
double4 shuffle2(const double4& x, const double4& y, const uint4& mask);
double8 shuffle(const double8& x, const uint8& mask);
double8 shuffle2(const double8& x, const double8& y, const uint8& mask);
double16 shuffle(const double16& x, const uint16& mask);
double16 shuffle2(const double16& x, const double16& y, const uint16& mask);

double2 shuffle(const double2& x, const ulong2& mask);
double2 shuffle2(const double2& x, const double2& y, const ulong2& mask);
double3 shuffle(const double3& x, const ulong3& mask);
double3 shuffle2(const double3& x, const double3& y, const ulong3& mask);
double4 shuffle(const double4& x, const ulong4& mask);
double4 shuffle2(const double4& x, const double4& y, const ulong4& mask);
double8 shuffle(const double8& x, const ulong8& mask);
double8 shuffle2(const double8& x, const double8& y, const ulong8& mask);
double16 shuffle(const double16& x, const ulong16& mask);
double16 shuffle2(const double16& x, const double16& y, const ulong16& mask);

double2 shuffle(const double2& x, const ulong2& mask);
double2 shuffle2(const double2& x, const double2& y, const ulong2& mask);
double3 shuffle(const double3& x, const ulong3& mask);
double3 shuffle2(const double3& x, const double3& y, const ulong3& mask);
double4 shuffle(const double4& x, const ulong4& mask);
double4 shuffle2(const double4& x, const double4& y, const ulong4& mask);
double8 shuffle(const double8& x, const ulong8& mask);
double8 shuffle2(const double8& x, const double8& y, const ulong8& mask);
double16 shuffle(const double16& x, const ulong16& mask);
double16 shuffle2(const double16& x, const double16& y, const ulong16& mask);

int vec_step(const char2& a);
int vec_step(const char3& a);
int vec_step(const char4& a);
int vec_step(const char8& a);
int vec_step(const char16& a);

int vec_step(const uchar2& a);
int vec_step(const uchar3& a);
int vec_step(const uchar4& a);
int vec_step(const uchar8& a);
int vec_step(const uchar16& a);

int vec_step(const short2& a);
int vec_step(const short3& a);
int vec_step(const short4& a);
int vec_step(const short8& a);
int vec_step(const short16& a);

int vec_step(const ushort2& a);
int vec_step(const ushort3& a);
int vec_step(const ushort4& a);
int vec_step(const ushort8& a);
int vec_step(const ushort16& a);

int vec_step(const int2& a);
int vec_step(const int3& a);
int vec_step(const int4& a);
int vec_step(const int8& a);
int vec_step(const int16& a);

int vec_step(const uint2& a);
int vec_step(const uint3& a);
int vec_step(const uint4& a);
int vec_step(const uint8& a);
int vec_step(const uint16& a);

int vec_step(const long2& a);
int vec_step(const long3& a);
int vec_step(const long4& a);
int vec_step(const long8& a);
int vec_step(const long16& a);

int vec_step(const ulong2& a);
int vec_step(const ulong3& a);
int vec_step(const ulong4& a);
int vec_step(const ulong8& a);
int vec_step(const ulong16& a);

int vec_step(const float2& a);
int vec_step(const float3& a);
int vec_step(const float4& a);
int vec_step(const float8& a);
int vec_step(const float16& a);

int vec_step(const double2& a);
int vec_step(const double3& a);
int vec_step(const double4& a);
int vec_step(const double8& a);
int vec_step(const double16& a);

uchar convert_uchar(const char& v);
uchar convert_uchar_sat(const char& v);
uchar2 convert_uchar2(const char2& v);
uchar2 convert_uchar2_sat(const char2& v);
uchar3 convert_uchar3(const char3& v);
uchar3 convert_uchar3_sat(const char3& v);
uchar4 convert_uchar4(const char4& v);
uchar4 convert_uchar4_sat(const char4& v);
uchar8 convert_uchar8(const char8& v);
uchar8 convert_uchar8_sat(const char8& v);
uchar16 convert_uchar16(const char16& v);
uchar16 convert_uchar16_sat(const char16& v);

short convert_short(const char& v);
short convert_short_sat(const char& v);
short2 convert_short2(const char2& v);
short2 convert_short2_sat(const char2& v);
short3 convert_short3(const char3& v);
short3 convert_short3_sat(const char3& v);
short4 convert_short4(const char4& v);
short4 convert_short4_sat(const char4& v);
short8 convert_short8(const char8& v);
short8 convert_short8_sat(const char8& v);
short16 convert_short16(const char16& v);
short16 convert_short16_sat(const char16& v);

ushort convert_ushort(const char& v);
ushort convert_ushort_sat(const char& v);
ushort2 convert_ushort2(const char2& v);
ushort2 convert_ushort2_sat(const char2& v);
ushort3 convert_ushort3(const char3& v);
ushort3 convert_ushort3_sat(const char3& v);
ushort4 convert_ushort4(const char4& v);
ushort4 convert_ushort4_sat(const char4& v);
ushort8 convert_ushort8(const char8& v);
ushort8 convert_ushort8_sat(const char8& v);
ushort16 convert_ushort16(const char16& v);
ushort16 convert_ushort16_sat(const char16& v);

int convert_int(const char& v);
int convert_int_sat(const char& v);
int2 convert_int2(const char2& v);
int2 convert_int2_sat(const char2& v);
int3 convert_int3(const char3& v);
int3 convert_int3_sat(const char3& v);
int4 convert_int4(const char4& v);
int4 convert_int4_sat(const char4& v);
int8 convert_int8(const char8& v);
int8 convert_int8_sat(const char8& v);
int16 convert_int16(const char16& v);
int16 convert_int16_sat(const char16& v);

uint convert_uint(const char& v);
uint convert_uint_sat(const char& v);
uint2 convert_uint2(const char2& v);
uint2 convert_uint2_sat(const char2& v);
uint3 convert_uint3(const char3& v);
uint3 convert_uint3_sat(const char3& v);
uint4 convert_uint4(const char4& v);
uint4 convert_uint4_sat(const char4& v);
uint8 convert_uint8(const char8& v);
uint8 convert_uint8_sat(const char8& v);
uint16 convert_uint16(const char16& v);
uint16 convert_uint16_sat(const char16& v);

long convert_long(const char& v);
long convert_long_sat(const char& v);
long2 convert_long2(const char2& v);
long2 convert_long2_sat(const char2& v);
long3 convert_long3(const char3& v);
long3 convert_long3_sat(const char3& v);
long4 convert_long4(const char4& v);
long4 convert_long4_sat(const char4& v);
long8 convert_long8(const char8& v);
long8 convert_long8_sat(const char8& v);
long16 convert_long16(const char16& v);
long16 convert_long16_sat(const char16& v);

ulong convert_ulong(const char& v);
ulong convert_ulong_sat(const char& v);
ulong2 convert_ulong2(const char2& v);
ulong2 convert_ulong2_sat(const char2& v);
ulong3 convert_ulong3(const char3& v);
ulong3 convert_ulong3_sat(const char3& v);
ulong4 convert_ulong4(const char4& v);
ulong4 convert_ulong4_sat(const char4& v);
ulong8 convert_ulong8(const char8& v);
ulong8 convert_ulong8_sat(const char8& v);
ulong16 convert_ulong16(const char16& v);
ulong16 convert_ulong16_sat(const char16& v);

char convert_char(const uchar& v);
char convert_char_sat(const uchar& v);
char2 convert_char2(const uchar2& v);
char2 convert_char2_sat(const uchar2& v);
char3 convert_char3(const uchar3& v);
char3 convert_char3_sat(const uchar3& v);
char4 convert_char4(const uchar4& v);
char4 convert_char4_sat(const uchar4& v);
char8 convert_char8(const uchar8& v);
char8 convert_char8_sat(const uchar8& v);
char16 convert_char16(const uchar16& v);
char16 convert_char16_sat(const uchar16& v);

short convert_short(const uchar& v);
short convert_short_sat(const uchar& v);
short2 convert_short2(const uchar2& v);
short2 convert_short2_sat(const uchar2& v);
short3 convert_short3(const uchar3& v);
short3 convert_short3_sat(const uchar3& v);
short4 convert_short4(const uchar4& v);
short4 convert_short4_sat(const uchar4& v);
short8 convert_short8(const uchar8& v);
short8 convert_short8_sat(const uchar8& v);
short16 convert_short16(const uchar16& v);
short16 convert_short16_sat(const uchar16& v);

ushort convert_ushort(const uchar& v);
ushort convert_ushort_sat(const uchar& v);
ushort2 convert_ushort2(const uchar2& v);
ushort2 convert_ushort2_sat(const uchar2& v);
ushort3 convert_ushort3(const uchar3& v);
ushort3 convert_ushort3_sat(const uchar3& v);
ushort4 convert_ushort4(const uchar4& v);
ushort4 convert_ushort4_sat(const uchar4& v);
ushort8 convert_ushort8(const uchar8& v);
ushort8 convert_ushort8_sat(const uchar8& v);
ushort16 convert_ushort16(const uchar16& v);
ushort16 convert_ushort16_sat(const uchar16& v);

int convert_int(const uchar& v);
int convert_int_sat(const uchar& v);
int2 convert_int2(const uchar2& v);
int2 convert_int2_sat(const uchar2& v);
int3 convert_int3(const uchar3& v);
int3 convert_int3_sat(const uchar3& v);
int4 convert_int4(const uchar4& v);
int4 convert_int4_sat(const uchar4& v);
int8 convert_int8(const uchar8& v);
int8 convert_int8_sat(const uchar8& v);
int16 convert_int16(const uchar16& v);
int16 convert_int16_sat(const uchar16& v);

uint convert_uint(const uchar& v);
uint convert_uint_sat(const uchar& v);
uint2 convert_uint2(const uchar2& v);
uint2 convert_uint2_sat(const uchar2& v);
uint3 convert_uint3(const uchar3& v);
uint3 convert_uint3_sat(const uchar3& v);
uint4 convert_uint4(const uchar4& v);
uint4 convert_uint4_sat(const uchar4& v);
uint8 convert_uint8(const uchar8& v);
uint8 convert_uint8_sat(const uchar8& v);
uint16 convert_uint16(const uchar16& v);
uint16 convert_uint16_sat(const uchar16& v);

long convert_long(const uchar& v);
long convert_long_sat(const uchar& v);
long2 convert_long2(const uchar2& v);
long2 convert_long2_sat(const uchar2& v);
long3 convert_long3(const uchar3& v);
long3 convert_long3_sat(const uchar3& v);
long4 convert_long4(const uchar4& v);
long4 convert_long4_sat(const uchar4& v);
long8 convert_long8(const uchar8& v);
long8 convert_long8_sat(const uchar8& v);
long16 convert_long16(const uchar16& v);
long16 convert_long16_sat(const uchar16& v);

ulong convert_ulong(const uchar& v);
ulong convert_ulong_sat(const uchar& v);
ulong2 convert_ulong2(const uchar2& v);
ulong2 convert_ulong2_sat(const uchar2& v);
ulong3 convert_ulong3(const uchar3& v);
ulong3 convert_ulong3_sat(const uchar3& v);
ulong4 convert_ulong4(const uchar4& v);
ulong4 convert_ulong4_sat(const uchar4& v);
ulong8 convert_ulong8(const uchar8& v);
ulong8 convert_ulong8_sat(const uchar8& v);
ulong16 convert_ulong16(const uchar16& v);
ulong16 convert_ulong16_sat(const uchar16& v);

char convert_char(const short& v);
char convert_char_sat(const short& v);
char2 convert_char2(const short2& v);
char2 convert_char2_sat(const short2& v);
char3 convert_char3(const short3& v);
char3 convert_char3_sat(const short3& v);
char4 convert_char4(const short4& v);
char4 convert_char4_sat(const short4& v);
char8 convert_char8(const short8& v);
char8 convert_char8_sat(const short8& v);
char16 convert_char16(const short16& v);
char16 convert_char16_sat(const short16& v);

uchar convert_uchar(const short& v);
uchar convert_uchar_sat(const short& v);
uchar2 convert_uchar2(const short2& v);
uchar2 convert_uchar2_sat(const short2& v);
uchar3 convert_uchar3(const short3& v);
uchar3 convert_uchar3_sat(const short3& v);
uchar4 convert_uchar4(const short4& v);
uchar4 convert_uchar4_sat(const short4& v);
uchar8 convert_uchar8(const short8& v);
uchar8 convert_uchar8_sat(const short8& v);
uchar16 convert_uchar16(const short16& v);
uchar16 convert_uchar16_sat(const short16& v);

ushort convert_ushort(const short& v);
ushort convert_ushort_sat(const short& v);
ushort2 convert_ushort2(const short2& v);
ushort2 convert_ushort2_sat(const short2& v);
ushort3 convert_ushort3(const short3& v);
ushort3 convert_ushort3_sat(const short3& v);
ushort4 convert_ushort4(const short4& v);
ushort4 convert_ushort4_sat(const short4& v);
ushort8 convert_ushort8(const short8& v);
ushort8 convert_ushort8_sat(const short8& v);
ushort16 convert_ushort16(const short16& v);
ushort16 convert_ushort16_sat(const short16& v);

int convert_int(const short& v);
int convert_int_sat(const short& v);
int2 convert_int2(const short2& v);
int2 convert_int2_sat(const short2& v);
int3 convert_int3(const short3& v);
int3 convert_int3_sat(const short3& v);
int4 convert_int4(const short4& v);
int4 convert_int4_sat(const short4& v);
int8 convert_int8(const short8& v);
int8 convert_int8_sat(const short8& v);
int16 convert_int16(const short16& v);
int16 convert_int16_sat(const short16& v);

uint convert_uint(const short& v);
uint convert_uint_sat(const short& v);
uint2 convert_uint2(const short2& v);
uint2 convert_uint2_sat(const short2& v);
uint3 convert_uint3(const short3& v);
uint3 convert_uint3_sat(const short3& v);
uint4 convert_uint4(const short4& v);
uint4 convert_uint4_sat(const short4& v);
uint8 convert_uint8(const short8& v);
uint8 convert_uint8_sat(const short8& v);
uint16 convert_uint16(const short16& v);
uint16 convert_uint16_sat(const short16& v);

long convert_long(const short& v);
long convert_long_sat(const short& v);
long2 convert_long2(const short2& v);
long2 convert_long2_sat(const short2& v);
long3 convert_long3(const short3& v);
long3 convert_long3_sat(const short3& v);
long4 convert_long4(const short4& v);
long4 convert_long4_sat(const short4& v);
long8 convert_long8(const short8& v);
long8 convert_long8_sat(const short8& v);
long16 convert_long16(const short16& v);
long16 convert_long16_sat(const short16& v);

ulong convert_ulong(const short& v);
ulong convert_ulong_sat(const short& v);
ulong2 convert_ulong2(const short2& v);
ulong2 convert_ulong2_sat(const short2& v);
ulong3 convert_ulong3(const short3& v);
ulong3 convert_ulong3_sat(const short3& v);
ulong4 convert_ulong4(const short4& v);
ulong4 convert_ulong4_sat(const short4& v);
ulong8 convert_ulong8(const short8& v);
ulong8 convert_ulong8_sat(const short8& v);
ulong16 convert_ulong16(const short16& v);
ulong16 convert_ulong16_sat(const short16& v);

char convert_char(const ushort& v);
char convert_char_sat(const ushort& v);
char2 convert_char2(const ushort2& v);
char2 convert_char2_sat(const ushort2& v);
char3 convert_char3(const ushort3& v);
char3 convert_char3_sat(const ushort3& v);
char4 convert_char4(const ushort4& v);
char4 convert_char4_sat(const ushort4& v);
char8 convert_char8(const ushort8& v);
char8 convert_char8_sat(const ushort8& v);
char16 convert_char16(const ushort16& v);
char16 convert_char16_sat(const ushort16& v);

uchar convert_uchar(const ushort& v);
uchar convert_uchar_sat(const ushort& v);
uchar2 convert_uchar2(const ushort2& v);
uchar2 convert_uchar2_sat(const ushort2& v);
uchar3 convert_uchar3(const ushort3& v);
uchar3 convert_uchar3_sat(const ushort3& v);
uchar4 convert_uchar4(const ushort4& v);
uchar4 convert_uchar4_sat(const ushort4& v);
uchar8 convert_uchar8(const ushort8& v);
uchar8 convert_uchar8_sat(const ushort8& v);
uchar16 convert_uchar16(const ushort16& v);
uchar16 convert_uchar16_sat(const ushort16& v);

short convert_short(const ushort& v);
short convert_short_sat(const ushort& v);
short2 convert_short2(const ushort2& v);
short2 convert_short2_sat(const ushort2& v);
short3 convert_short3(const ushort3& v);
short3 convert_short3_sat(const ushort3& v);
short4 convert_short4(const ushort4& v);
short4 convert_short4_sat(const ushort4& v);
short8 convert_short8(const ushort8& v);
short8 convert_short8_sat(const ushort8& v);
short16 convert_short16(const ushort16& v);
short16 convert_short16_sat(const ushort16& v);

int convert_int(const ushort& v);
int convert_int_sat(const ushort& v);
int2 convert_int2(const ushort2& v);
int2 convert_int2_sat(const ushort2& v);
int3 convert_int3(const ushort3& v);
int3 convert_int3_sat(const ushort3& v);
int4 convert_int4(const ushort4& v);
int4 convert_int4_sat(const ushort4& v);
int8 convert_int8(const ushort8& v);
int8 convert_int8_sat(const ushort8& v);
int16 convert_int16(const ushort16& v);
int16 convert_int16_sat(const ushort16& v);

uint convert_uint(const ushort& v);
uint convert_uint_sat(const ushort& v);
uint2 convert_uint2(const ushort2& v);
uint2 convert_uint2_sat(const ushort2& v);
uint3 convert_uint3(const ushort3& v);
uint3 convert_uint3_sat(const ushort3& v);
uint4 convert_uint4(const ushort4& v);
uint4 convert_uint4_sat(const ushort4& v);
uint8 convert_uint8(const ushort8& v);
uint8 convert_uint8_sat(const ushort8& v);
uint16 convert_uint16(const ushort16& v);
uint16 convert_uint16_sat(const ushort16& v);

long convert_long(const ushort& v);
long convert_long_sat(const ushort& v);
long2 convert_long2(const ushort2& v);
long2 convert_long2_sat(const ushort2& v);
long3 convert_long3(const ushort3& v);
long3 convert_long3_sat(const ushort3& v);
long4 convert_long4(const ushort4& v);
long4 convert_long4_sat(const ushort4& v);
long8 convert_long8(const ushort8& v);
long8 convert_long8_sat(const ushort8& v);
long16 convert_long16(const ushort16& v);
long16 convert_long16_sat(const ushort16& v);

ulong convert_ulong(const ushort& v);
ulong convert_ulong_sat(const ushort& v);
ulong2 convert_ulong2(const ushort2& v);
ulong2 convert_ulong2_sat(const ushort2& v);
ulong3 convert_ulong3(const ushort3& v);
ulong3 convert_ulong3_sat(const ushort3& v);
ulong4 convert_ulong4(const ushort4& v);
ulong4 convert_ulong4_sat(const ushort4& v);
ulong8 convert_ulong8(const ushort8& v);
ulong8 convert_ulong8_sat(const ushort8& v);
ulong16 convert_ulong16(const ushort16& v);
ulong16 convert_ulong16_sat(const ushort16& v);

char convert_char(const int& v);
char convert_char_sat(const int& v);
char2 convert_char2(const int2& v);
char2 convert_char2_sat(const int2& v);
char3 convert_char3(const int3& v);
char3 convert_char3_sat(const int3& v);
char4 convert_char4(const int4& v);
char4 convert_char4_sat(const int4& v);
char8 convert_char8(const int8& v);
char8 convert_char8_sat(const int8& v);
char16 convert_char16(const int16& v);
char16 convert_char16_sat(const int16& v);

uchar convert_uchar(const int& v);
uchar convert_uchar_sat(const int& v);
uchar2 convert_uchar2(const int2& v);
uchar2 convert_uchar2_sat(const int2& v);
uchar3 convert_uchar3(const int3& v);
uchar3 convert_uchar3_sat(const int3& v);
uchar4 convert_uchar4(const int4& v);
uchar4 convert_uchar4_sat(const int4& v);
uchar8 convert_uchar8(const int8& v);
uchar8 convert_uchar8_sat(const int8& v);
uchar16 convert_uchar16(const int16& v);
uchar16 convert_uchar16_sat(const int16& v);

short convert_short(const int& v);
short convert_short_sat(const int& v);
short2 convert_short2(const int2& v);
short2 convert_short2_sat(const int2& v);
short3 convert_short3(const int3& v);
short3 convert_short3_sat(const int3& v);
short4 convert_short4(const int4& v);
short4 convert_short4_sat(const int4& v);
short8 convert_short8(const int8& v);
short8 convert_short8_sat(const int8& v);
short16 convert_short16(const int16& v);
short16 convert_short16_sat(const int16& v);

ushort convert_ushort(const int& v);
ushort convert_ushort_sat(const int& v);
ushort2 convert_ushort2(const int2& v);
ushort2 convert_ushort2_sat(const int2& v);
ushort3 convert_ushort3(const int3& v);
ushort3 convert_ushort3_sat(const int3& v);
ushort4 convert_ushort4(const int4& v);
ushort4 convert_ushort4_sat(const int4& v);
ushort8 convert_ushort8(const int8& v);
ushort8 convert_ushort8_sat(const int8& v);
ushort16 convert_ushort16(const int16& v);
ushort16 convert_ushort16_sat(const int16& v);

uint convert_uint(const int& v);
uint convert_uint_sat(const int& v);
uint2 convert_uint2(const int2& v);
uint2 convert_uint2_sat(const int2& v);
uint3 convert_uint3(const int3& v);
uint3 convert_uint3_sat(const int3& v);
uint4 convert_uint4(const int4& v);
uint4 convert_uint4_sat(const int4& v);
uint8 convert_uint8(const int8& v);
uint8 convert_uint8_sat(const int8& v);
uint16 convert_uint16(const int16& v);
uint16 convert_uint16_sat(const int16& v);

long convert_long(const int& v);
long convert_long_sat(const int& v);
long2 convert_long2(const int2& v);
long2 convert_long2_sat(const int2& v);
long3 convert_long3(const int3& v);
long3 convert_long3_sat(const int3& v);
long4 convert_long4(const int4& v);
long4 convert_long4_sat(const int4& v);
long8 convert_long8(const int8& v);
long8 convert_long8_sat(const int8& v);
long16 convert_long16(const int16& v);
long16 convert_long16_sat(const int16& v);

ulong convert_ulong(const int& v);
ulong convert_ulong_sat(const int& v);
ulong2 convert_ulong2(const int2& v);
ulong2 convert_ulong2_sat(const int2& v);
ulong3 convert_ulong3(const int3& v);
ulong3 convert_ulong3_sat(const int3& v);
ulong4 convert_ulong4(const int4& v);
ulong4 convert_ulong4_sat(const int4& v);
ulong8 convert_ulong8(const int8& v);
ulong8 convert_ulong8_sat(const int8& v);
ulong16 convert_ulong16(const int16& v);
ulong16 convert_ulong16_sat(const int16& v);

char convert_char(const uint& v);
char convert_char_sat(const uint& v);
char2 convert_char2(const uint2& v);
char2 convert_char2_sat(const uint2& v);
char3 convert_char3(const uint3& v);
char3 convert_char3_sat(const uint3& v);
char4 convert_char4(const uint4& v);
char4 convert_char4_sat(const uint4& v);
char8 convert_char8(const uint8& v);
char8 convert_char8_sat(const uint8& v);
char16 convert_char16(const uint16& v);
char16 convert_char16_sat(const uint16& v);

uchar convert_uchar(const uint& v);
uchar convert_uchar_sat(const uint& v);
uchar2 convert_uchar2(const uint2& v);
uchar2 convert_uchar2_sat(const uint2& v);
uchar3 convert_uchar3(const uint3& v);
uchar3 convert_uchar3_sat(const uint3& v);
uchar4 convert_uchar4(const uint4& v);
uchar4 convert_uchar4_sat(const uint4& v);
uchar8 convert_uchar8(const uint8& v);
uchar8 convert_uchar8_sat(const uint8& v);
uchar16 convert_uchar16(const uint16& v);
uchar16 convert_uchar16_sat(const uint16& v);

short convert_short(const uint& v);
short convert_short_sat(const uint& v);
short2 convert_short2(const uint2& v);
short2 convert_short2_sat(const uint2& v);
short3 convert_short3(const uint3& v);
short3 convert_short3_sat(const uint3& v);
short4 convert_short4(const uint4& v);
short4 convert_short4_sat(const uint4& v);
short8 convert_short8(const uint8& v);
short8 convert_short8_sat(const uint8& v);
short16 convert_short16(const uint16& v);
short16 convert_short16_sat(const uint16& v);

ushort convert_ushort(const uint& v);
ushort convert_ushort_sat(const uint& v);
ushort2 convert_ushort2(const uint2& v);
ushort2 convert_ushort2_sat(const uint2& v);
ushort3 convert_ushort3(const uint3& v);
ushort3 convert_ushort3_sat(const uint3& v);
ushort4 convert_ushort4(const uint4& v);
ushort4 convert_ushort4_sat(const uint4& v);
ushort8 convert_ushort8(const uint8& v);
ushort8 convert_ushort8_sat(const uint8& v);
ushort16 convert_ushort16(const uint16& v);
ushort16 convert_ushort16_sat(const uint16& v);

int convert_int(const uint& v);
int convert_int_sat(const uint& v);
int2 convert_int2(const uint2& v);
int2 convert_int2_sat(const uint2& v);
int3 convert_int3(const uint3& v);
int3 convert_int3_sat(const uint3& v);
int4 convert_int4(const uint4& v);
int4 convert_int4_sat(const uint4& v);
int8 convert_int8(const uint8& v);
int8 convert_int8_sat(const uint8& v);
int16 convert_int16(const uint16& v);
int16 convert_int16_sat(const uint16& v);

long convert_long(const uint& v);
long convert_long_sat(const uint& v);
long2 convert_long2(const uint2& v);
long2 convert_long2_sat(const uint2& v);
long3 convert_long3(const uint3& v);
long3 convert_long3_sat(const uint3& v);
long4 convert_long4(const uint4& v);
long4 convert_long4_sat(const uint4& v);
long8 convert_long8(const uint8& v);
long8 convert_long8_sat(const uint8& v);
long16 convert_long16(const uint16& v);
long16 convert_long16_sat(const uint16& v);

ulong convert_ulong(const uint& v);
ulong convert_ulong_sat(const uint& v);
ulong2 convert_ulong2(const uint2& v);
ulong2 convert_ulong2_sat(const uint2& v);
ulong3 convert_ulong3(const uint3& v);
ulong3 convert_ulong3_sat(const uint3& v);
ulong4 convert_ulong4(const uint4& v);
ulong4 convert_ulong4_sat(const uint4& v);
ulong8 convert_ulong8(const uint8& v);
ulong8 convert_ulong8_sat(const uint8& v);
ulong16 convert_ulong16(const uint16& v);
ulong16 convert_ulong16_sat(const uint16& v);

char convert_char(const long& v);
char convert_char_sat(const long& v);
char2 convert_char2(const long2& v);
char2 convert_char2_sat(const long2& v);
char3 convert_char3(const long3& v);
char3 convert_char3_sat(const long3& v);
char4 convert_char4(const long4& v);
char4 convert_char4_sat(const long4& v);
char8 convert_char8(const long8& v);
char8 convert_char8_sat(const long8& v);
char16 convert_char16(const long16& v);
char16 convert_char16_sat(const long16& v);

uchar convert_uchar(const long& v);
uchar convert_uchar_sat(const long& v);
uchar2 convert_uchar2(const long2& v);
uchar2 convert_uchar2_sat(const long2& v);
uchar3 convert_uchar3(const long3& v);
uchar3 convert_uchar3_sat(const long3& v);
uchar4 convert_uchar4(const long4& v);
uchar4 convert_uchar4_sat(const long4& v);
uchar8 convert_uchar8(const long8& v);
uchar8 convert_uchar8_sat(const long8& v);
uchar16 convert_uchar16(const long16& v);
uchar16 convert_uchar16_sat(const long16& v);

short convert_short(const long& v);
short convert_short_sat(const long& v);
short2 convert_short2(const long2& v);
short2 convert_short2_sat(const long2& v);
short3 convert_short3(const long3& v);
short3 convert_short3_sat(const long3& v);
short4 convert_short4(const long4& v);
short4 convert_short4_sat(const long4& v);
short8 convert_short8(const long8& v);
short8 convert_short8_sat(const long8& v);
short16 convert_short16(const long16& v);
short16 convert_short16_sat(const long16& v);

ushort convert_ushort(const long& v);
ushort convert_ushort_sat(const long& v);
ushort2 convert_ushort2(const long2& v);
ushort2 convert_ushort2_sat(const long2& v);
ushort3 convert_ushort3(const long3& v);
ushort3 convert_ushort3_sat(const long3& v);
ushort4 convert_ushort4(const long4& v);
ushort4 convert_ushort4_sat(const long4& v);
ushort8 convert_ushort8(const long8& v);
ushort8 convert_ushort8_sat(const long8& v);
ushort16 convert_ushort16(const long16& v);
ushort16 convert_ushort16_sat(const long16& v);

int convert_int(const long& v);
int convert_int_sat(const long& v);
int2 convert_int2(const long2& v);
int2 convert_int2_sat(const long2& v);
int3 convert_int3(const long3& v);
int3 convert_int3_sat(const long3& v);
int4 convert_int4(const long4& v);
int4 convert_int4_sat(const long4& v);
int8 convert_int8(const long8& v);
int8 convert_int8_sat(const long8& v);
int16 convert_int16(const long16& v);
int16 convert_int16_sat(const long16& v);

uint convert_uint(const long& v);
uint convert_uint_sat(const long& v);
uint2 convert_uint2(const long2& v);
uint2 convert_uint2_sat(const long2& v);
uint3 convert_uint3(const long3& v);
uint3 convert_uint3_sat(const long3& v);
uint4 convert_uint4(const long4& v);
uint4 convert_uint4_sat(const long4& v);
uint8 convert_uint8(const long8& v);
uint8 convert_uint8_sat(const long8& v);
uint16 convert_uint16(const long16& v);
uint16 convert_uint16_sat(const long16& v);

ulong convert_ulong(const long& v);
ulong convert_ulong_sat(const long& v);
ulong2 convert_ulong2(const long2& v);
ulong2 convert_ulong2_sat(const long2& v);
ulong3 convert_ulong3(const long3& v);
ulong3 convert_ulong3_sat(const long3& v);
ulong4 convert_ulong4(const long4& v);
ulong4 convert_ulong4_sat(const long4& v);
ulong8 convert_ulong8(const long8& v);
ulong8 convert_ulong8_sat(const long8& v);
ulong16 convert_ulong16(const long16& v);
ulong16 convert_ulong16_sat(const long16& v);

char convert_char(const ulong& v);
char convert_char_sat(const ulong& v);
char2 convert_char2(const ulong2& v);
char2 convert_char2_sat(const ulong2& v);
char3 convert_char3(const ulong3& v);
char3 convert_char3_sat(const ulong3& v);
char4 convert_char4(const ulong4& v);
char4 convert_char4_sat(const ulong4& v);
char8 convert_char8(const ulong8& v);
char8 convert_char8_sat(const ulong8& v);
char16 convert_char16(const ulong16& v);
char16 convert_char16_sat(const ulong16& v);

uchar convert_uchar(const ulong& v);
uchar convert_uchar_sat(const ulong& v);
uchar2 convert_uchar2(const ulong2& v);
uchar2 convert_uchar2_sat(const ulong2& v);
uchar3 convert_uchar3(const ulong3& v);
uchar3 convert_uchar3_sat(const ulong3& v);
uchar4 convert_uchar4(const ulong4& v);
uchar4 convert_uchar4_sat(const ulong4& v);
uchar8 convert_uchar8(const ulong8& v);
uchar8 convert_uchar8_sat(const ulong8& v);
uchar16 convert_uchar16(const ulong16& v);
uchar16 convert_uchar16_sat(const ulong16& v);

short convert_short(const ulong& v);
short convert_short_sat(const ulong& v);
short2 convert_short2(const ulong2& v);
short2 convert_short2_sat(const ulong2& v);
short3 convert_short3(const ulong3& v);
short3 convert_short3_sat(const ulong3& v);
short4 convert_short4(const ulong4& v);
short4 convert_short4_sat(const ulong4& v);
short8 convert_short8(const ulong8& v);
short8 convert_short8_sat(const ulong8& v);
short16 convert_short16(const ulong16& v);
short16 convert_short16_sat(const ulong16& v);

ushort convert_ushort(const ulong& v);
ushort convert_ushort_sat(const ulong& v);
ushort2 convert_ushort2(const ulong2& v);
ushort2 convert_ushort2_sat(const ulong2& v);
ushort3 convert_ushort3(const ulong3& v);
ushort3 convert_ushort3_sat(const ulong3& v);
ushort4 convert_ushort4(const ulong4& v);
ushort4 convert_ushort4_sat(const ulong4& v);
ushort8 convert_ushort8(const ulong8& v);
ushort8 convert_ushort8_sat(const ulong8& v);
ushort16 convert_ushort16(const ulong16& v);
ushort16 convert_ushort16_sat(const ulong16& v);

int convert_int(const ulong& v);
int convert_int_sat(const ulong& v);
int2 convert_int2(const ulong2& v);
int2 convert_int2_sat(const ulong2& v);
int3 convert_int3(const ulong3& v);
int3 convert_int3_sat(const ulong3& v);
int4 convert_int4(const ulong4& v);
int4 convert_int4_sat(const ulong4& v);
int8 convert_int8(const ulong8& v);
int8 convert_int8_sat(const ulong8& v);
int16 convert_int16(const ulong16& v);
int16 convert_int16_sat(const ulong16& v);

uint convert_uint(const ulong& v);
uint convert_uint_sat(const ulong& v);
uint2 convert_uint2(const ulong2& v);
uint2 convert_uint2_sat(const ulong2& v);
uint3 convert_uint3(const ulong3& v);
uint3 convert_uint3_sat(const ulong3& v);
uint4 convert_uint4(const ulong4& v);
uint4 convert_uint4_sat(const ulong4& v);
uint8 convert_uint8(const ulong8& v);
uint8 convert_uint8_sat(const ulong8& v);
uint16 convert_uint16(const ulong16& v);
uint16 convert_uint16_sat(const ulong16& v);

long convert_long(const ulong& v);
long convert_long_sat(const ulong& v);
long2 convert_long2(const ulong2& v);
long2 convert_long2_sat(const ulong2& v);
long3 convert_long3(const ulong3& v);
long3 convert_long3_sat(const ulong3& v);
long4 convert_long4(const ulong4& v);
long4 convert_long4_sat(const ulong4& v);
long8 convert_long8(const ulong8& v);
long8 convert_long8_sat(const ulong8& v);
long16 convert_long16(const ulong16& v);
long16 convert_long16_sat(const ulong16& v);

char convert_char(const float& v);
char convert_char_rte(const float& v);
char convert_char_rtz(const float& v);
char convert_char_rtp(const float& v);
char convert_char_rtn(const float& v);
char convert_char_sat(const float& v);
char convert_char_sat_rte(const float& v);
char convert_char_sat_rtz(const float& v);
char convert_char_sat_rtp(const float& v);
char convert_char_sat_rtn(const float& v);

char2 convert_char2(const float2& v);
char2 convert_char2_rte(const float2& v);
char2 convert_char2_rtz(const float2& v);
char2 convert_char2_rtp(const float2& v);
char2 convert_char2_rtn(const float2& v);
char2 convert_char2_sat(const float2& v);
char2 convert_char2_sat_rte(const float2& v);
char2 convert_char2_sat_rtz(const float2& v);
char2 convert_char2_sat_rtp(const float2& v);
char2 convert_char2_sat_rtn(const float2& v);

char3 convert_char3(const float3& v);
char3 convert_char3_rte(const float3& v);
char3 convert_char3_rtz(const float3& v);
char3 convert_char3_rtp(const float3& v);
char3 convert_char3_rtn(const float3& v);
char3 convert_char3_sat(const float3& v);
char3 convert_char3_sat_rte(const float3& v);
char3 convert_char3_sat_rtz(const float3& v);
char3 convert_char3_sat_rtp(const float3& v);
char3 convert_char3_sat_rtn(const float3& v);

char4 convert_char4(const float4& v);
char4 convert_char4_rte(const float4& v);
char4 convert_char4_rtz(const float4& v);
char4 convert_char4_rtp(const float4& v);
char4 convert_char4_rtn(const float4& v);
char4 convert_char4_sat(const float4& v);
char4 convert_char4_sat_rte(const float4& v);
char4 convert_char4_sat_rtz(const float4& v);
char4 convert_char4_sat_rtp(const float4& v);
char4 convert_char4_sat_rtn(const float4& v);

char8 convert_char8(const float8& v);
char8 convert_char8_rte(const float8& v);
char8 convert_char8_rtz(const float8& v);
char8 convert_char8_rtp(const float8& v);
char8 convert_char8_rtn(const float8& v);
char8 convert_char8_sat(const float8& v);
char8 convert_char8_sat_rte(const float8& v);
char8 convert_char8_sat_rtz(const float8& v);
char8 convert_char8_sat_rtp(const float8& v);
char8 convert_char8_sat_rtn(const float8& v);

char16 convert_char16(const float16& v);
char16 convert_char16_rte(const float16& v);
char16 convert_char16_rtz(const float16& v);
char16 convert_char16_rtp(const float16& v);
char16 convert_char16_rtn(const float16& v);
char16 convert_char16_sat(const float16& v);
char16 convert_char16_sat_rte(const float16& v);
char16 convert_char16_sat_rtz(const float16& v);
char16 convert_char16_sat_rtp(const float16& v);
char16 convert_char16_sat_rtn(const float16& v);

uchar convert_uchar(const float& v);
uchar convert_uchar_rte(const float& v);
uchar convert_uchar_rtz(const float& v);
uchar convert_uchar_rtp(const float& v);
uchar convert_uchar_rtn(const float& v);
uchar convert_uchar_sat(const float& v);
uchar convert_uchar_sat_rte(const float& v);
uchar convert_uchar_sat_rtz(const float& v);
uchar convert_uchar_sat_rtp(const float& v);
uchar convert_uchar_sat_rtn(const float& v);

uchar2 convert_uchar2(const float2& v);
uchar2 convert_uchar2_rte(const float2& v);
uchar2 convert_uchar2_rtz(const float2& v);
uchar2 convert_uchar2_rtp(const float2& v);
uchar2 convert_uchar2_rtn(const float2& v);
uchar2 convert_uchar2_sat(const float2& v);
uchar2 convert_uchar2_sat_rte(const float2& v);
uchar2 convert_uchar2_sat_rtz(const float2& v);
uchar2 convert_uchar2_sat_rtp(const float2& v);
uchar2 convert_uchar2_sat_rtn(const float2& v);

uchar3 convert_uchar3(const float3& v);
uchar3 convert_uchar3_rte(const float3& v);
uchar3 convert_uchar3_rtz(const float3& v);
uchar3 convert_uchar3_rtp(const float3& v);
uchar3 convert_uchar3_rtn(const float3& v);
uchar3 convert_uchar3_sat(const float3& v);
uchar3 convert_uchar3_sat_rte(const float3& v);
uchar3 convert_uchar3_sat_rtz(const float3& v);
uchar3 convert_uchar3_sat_rtp(const float3& v);
uchar3 convert_uchar3_sat_rtn(const float3& v);

uchar4 convert_uchar4(const float4& v);
uchar4 convert_uchar4_rte(const float4& v);
uchar4 convert_uchar4_rtz(const float4& v);
uchar4 convert_uchar4_rtp(const float4& v);
uchar4 convert_uchar4_rtn(const float4& v);
uchar4 convert_uchar4_sat(const float4& v);
uchar4 convert_uchar4_sat_rte(const float4& v);
uchar4 convert_uchar4_sat_rtz(const float4& v);
uchar4 convert_uchar4_sat_rtp(const float4& v);
uchar4 convert_uchar4_sat_rtn(const float4& v);

uchar8 convert_uchar8(const float8& v);
uchar8 convert_uchar8_rte(const float8& v);
uchar8 convert_uchar8_rtz(const float8& v);
uchar8 convert_uchar8_rtp(const float8& v);
uchar8 convert_uchar8_rtn(const float8& v);
uchar8 convert_uchar8_sat(const float8& v);
uchar8 convert_uchar8_sat_rte(const float8& v);
uchar8 convert_uchar8_sat_rtz(const float8& v);
uchar8 convert_uchar8_sat_rtp(const float8& v);
uchar8 convert_uchar8_sat_rtn(const float8& v);

uchar16 convert_uchar16(const float16& v);
uchar16 convert_uchar16_rte(const float16& v);
uchar16 convert_uchar16_rtz(const float16& v);
uchar16 convert_uchar16_rtp(const float16& v);
uchar16 convert_uchar16_rtn(const float16& v);
uchar16 convert_uchar16_sat(const float16& v);
uchar16 convert_uchar16_sat_rte(const float16& v);
uchar16 convert_uchar16_sat_rtz(const float16& v);
uchar16 convert_uchar16_sat_rtp(const float16& v);
uchar16 convert_uchar16_sat_rtn(const float16& v);

short convert_short(const float& v);
short convert_short_rte(const float& v);
short convert_short_rtz(const float& v);
short convert_short_rtp(const float& v);
short convert_short_rtn(const float& v);
short convert_short_sat(const float& v);
short convert_short_sat_rte(const float& v);
short convert_short_sat_rtz(const float& v);
short convert_short_sat_rtp(const float& v);
short convert_short_sat_rtn(const float& v);

short2 convert_short2(const float2& v);
short2 convert_short2_rte(const float2& v);
short2 convert_short2_rtz(const float2& v);
short2 convert_short2_rtp(const float2& v);
short2 convert_short2_rtn(const float2& v);
short2 convert_short2_sat(const float2& v);
short2 convert_short2_sat_rte(const float2& v);
short2 convert_short2_sat_rtz(const float2& v);
short2 convert_short2_sat_rtp(const float2& v);
short2 convert_short2_sat_rtn(const float2& v);

short3 convert_short3(const float3& v);
short3 convert_short3_rte(const float3& v);
short3 convert_short3_rtz(const float3& v);
short3 convert_short3_rtp(const float3& v);
short3 convert_short3_rtn(const float3& v);
short3 convert_short3_sat(const float3& v);
short3 convert_short3_sat_rte(const float3& v);
short3 convert_short3_sat_rtz(const float3& v);
short3 convert_short3_sat_rtp(const float3& v);
short3 convert_short3_sat_rtn(const float3& v);

short4 convert_short4(const float4& v);
short4 convert_short4_rte(const float4& v);
short4 convert_short4_rtz(const float4& v);
short4 convert_short4_rtp(const float4& v);
short4 convert_short4_rtn(const float4& v);
short4 convert_short4_sat(const float4& v);
short4 convert_short4_sat_rte(const float4& v);
short4 convert_short4_sat_rtz(const float4& v);
short4 convert_short4_sat_rtp(const float4& v);
short4 convert_short4_sat_rtn(const float4& v);

short8 convert_short8(const float8& v);
short8 convert_short8_rte(const float8& v);
short8 convert_short8_rtz(const float8& v);
short8 convert_short8_rtp(const float8& v);
short8 convert_short8_rtn(const float8& v);
short8 convert_short8_sat(const float8& v);
short8 convert_short8_sat_rte(const float8& v);
short8 convert_short8_sat_rtz(const float8& v);
short8 convert_short8_sat_rtp(const float8& v);
short8 convert_short8_sat_rtn(const float8& v);

short16 convert_short16(const float16& v);
short16 convert_short16_rte(const float16& v);
short16 convert_short16_rtz(const float16& v);
short16 convert_short16_rtp(const float16& v);
short16 convert_short16_rtn(const float16& v);
short16 convert_short16_sat(const float16& v);
short16 convert_short16_sat_rte(const float16& v);
short16 convert_short16_sat_rtz(const float16& v);
short16 convert_short16_sat_rtp(const float16& v);
short16 convert_short16_sat_rtn(const float16& v);

ushort convert_ushort(const float& v);
ushort convert_ushort_rte(const float& v);
ushort convert_ushort_rtz(const float& v);
ushort convert_ushort_rtp(const float& v);
ushort convert_ushort_rtn(const float& v);
ushort convert_ushort_sat(const float& v);
ushort convert_ushort_sat_rte(const float& v);
ushort convert_ushort_sat_rtz(const float& v);
ushort convert_ushort_sat_rtp(const float& v);
ushort convert_ushort_sat_rtn(const float& v);

ushort2 convert_ushort2(const float2& v);
ushort2 convert_ushort2_rte(const float2& v);
ushort2 convert_ushort2_rtz(const float2& v);
ushort2 convert_ushort2_rtp(const float2& v);
ushort2 convert_ushort2_rtn(const float2& v);
ushort2 convert_ushort2_sat(const float2& v);
ushort2 convert_ushort2_sat_rte(const float2& v);
ushort2 convert_ushort2_sat_rtz(const float2& v);
ushort2 convert_ushort2_sat_rtp(const float2& v);
ushort2 convert_ushort2_sat_rtn(const float2& v);

ushort3 convert_ushort3(const float3& v);
ushort3 convert_ushort3_rte(const float3& v);
ushort3 convert_ushort3_rtz(const float3& v);
ushort3 convert_ushort3_rtp(const float3& v);
ushort3 convert_ushort3_rtn(const float3& v);
ushort3 convert_ushort3_sat(const float3& v);
ushort3 convert_ushort3_sat_rte(const float3& v);
ushort3 convert_ushort3_sat_rtz(const float3& v);
ushort3 convert_ushort3_sat_rtp(const float3& v);
ushort3 convert_ushort3_sat_rtn(const float3& v);

ushort4 convert_ushort4(const float4& v);
ushort4 convert_ushort4_rte(const float4& v);
ushort4 convert_ushort4_rtz(const float4& v);
ushort4 convert_ushort4_rtp(const float4& v);
ushort4 convert_ushort4_rtn(const float4& v);
ushort4 convert_ushort4_sat(const float4& v);
ushort4 convert_ushort4_sat_rte(const float4& v);
ushort4 convert_ushort4_sat_rtz(const float4& v);
ushort4 convert_ushort4_sat_rtp(const float4& v);
ushort4 convert_ushort4_sat_rtn(const float4& v);

ushort8 convert_ushort8(const float8& v);
ushort8 convert_ushort8_rte(const float8& v);
ushort8 convert_ushort8_rtz(const float8& v);
ushort8 convert_ushort8_rtp(const float8& v);
ushort8 convert_ushort8_rtn(const float8& v);
ushort8 convert_ushort8_sat(const float8& v);
ushort8 convert_ushort8_sat_rte(const float8& v);
ushort8 convert_ushort8_sat_rtz(const float8& v);
ushort8 convert_ushort8_sat_rtp(const float8& v);
ushort8 convert_ushort8_sat_rtn(const float8& v);

ushort16 convert_ushort16(const float16& v);
ushort16 convert_ushort16_rte(const float16& v);
ushort16 convert_ushort16_rtz(const float16& v);
ushort16 convert_ushort16_rtp(const float16& v);
ushort16 convert_ushort16_rtn(const float16& v);
ushort16 convert_ushort16_sat(const float16& v);
ushort16 convert_ushort16_sat_rte(const float16& v);
ushort16 convert_ushort16_sat_rtz(const float16& v);
ushort16 convert_ushort16_sat_rtp(const float16& v);
ushort16 convert_ushort16_sat_rtn(const float16& v);

int convert_int(const float& v);
int convert_int_rte(const float& v);
int convert_int_rtz(const float& v);
int convert_int_rtp(const float& v);
int convert_int_rtn(const float& v);
int convert_int_sat(const float& v);
int convert_int_sat_rte(const float& v);
int convert_int_sat_rtz(const float& v);
int convert_int_sat_rtp(const float& v);
int convert_int_sat_rtn(const float& v);

int2 convert_int2(const float2& v);
int2 convert_int2_rte(const float2& v);
int2 convert_int2_rtz(const float2& v);
int2 convert_int2_rtp(const float2& v);
int2 convert_int2_rtn(const float2& v);
int2 convert_int2_sat(const float2& v);
int2 convert_int2_sat_rte(const float2& v);
int2 convert_int2_sat_rtz(const float2& v);
int2 convert_int2_sat_rtp(const float2& v);
int2 convert_int2_sat_rtn(const float2& v);

int3 convert_int3(const float3& v);
int3 convert_int3_rte(const float3& v);
int3 convert_int3_rtz(const float3& v);
int3 convert_int3_rtp(const float3& v);
int3 convert_int3_rtn(const float3& v);
int3 convert_int3_sat(const float3& v);
int3 convert_int3_sat_rte(const float3& v);
int3 convert_int3_sat_rtz(const float3& v);
int3 convert_int3_sat_rtp(const float3& v);
int3 convert_int3_sat_rtn(const float3& v);

int4 convert_int4(const float4& v);
int4 convert_int4_rte(const float4& v);
int4 convert_int4_rtz(const float4& v);
int4 convert_int4_rtp(const float4& v);
int4 convert_int4_rtn(const float4& v);
int4 convert_int4_sat(const float4& v);
int4 convert_int4_sat_rte(const float4& v);
int4 convert_int4_sat_rtz(const float4& v);
int4 convert_int4_sat_rtp(const float4& v);
int4 convert_int4_sat_rtn(const float4& v);

int8 convert_int8(const float8& v);
int8 convert_int8_rte(const float8& v);
int8 convert_int8_rtz(const float8& v);
int8 convert_int8_rtp(const float8& v);
int8 convert_int8_rtn(const float8& v);
int8 convert_int8_sat(const float8& v);
int8 convert_int8_sat_rte(const float8& v);
int8 convert_int8_sat_rtz(const float8& v);
int8 convert_int8_sat_rtp(const float8& v);
int8 convert_int8_sat_rtn(const float8& v);

int16 convert_int16(const float16& v);
int16 convert_int16_rte(const float16& v);
int16 convert_int16_rtz(const float16& v);
int16 convert_int16_rtp(const float16& v);
int16 convert_int16_rtn(const float16& v);
int16 convert_int16_sat(const float16& v);
int16 convert_int16_sat_rte(const float16& v);
int16 convert_int16_sat_rtz(const float16& v);
int16 convert_int16_sat_rtp(const float16& v);
int16 convert_int16_sat_rtn(const float16& v);

uint convert_uint(const float& v);
uint convert_uint_rte(const float& v);
uint convert_uint_rtz(const float& v);
uint convert_uint_rtp(const float& v);
uint convert_uint_rtn(const float& v);
uint convert_uint_sat(const float& v);
uint convert_uint_sat_rte(const float& v);
uint convert_uint_sat_rtz(const float& v);
uint convert_uint_sat_rtp(const float& v);
uint convert_uint_sat_rtn(const float& v);

uint2 convert_uint2(const float2& v);
uint2 convert_uint2_rte(const float2& v);
uint2 convert_uint2_rtz(const float2& v);
uint2 convert_uint2_rtp(const float2& v);
uint2 convert_uint2_rtn(const float2& v);
uint2 convert_uint2_sat(const float2& v);
uint2 convert_uint2_sat_rte(const float2& v);
uint2 convert_uint2_sat_rtz(const float2& v);
uint2 convert_uint2_sat_rtp(const float2& v);
uint2 convert_uint2_sat_rtn(const float2& v);

uint3 convert_uint3(const float3& v);
uint3 convert_uint3_rte(const float3& v);
uint3 convert_uint3_rtz(const float3& v);
uint3 convert_uint3_rtp(const float3& v);
uint3 convert_uint3_rtn(const float3& v);
uint3 convert_uint3_sat(const float3& v);
uint3 convert_uint3_sat_rte(const float3& v);
uint3 convert_uint3_sat_rtz(const float3& v);
uint3 convert_uint3_sat_rtp(const float3& v);
uint3 convert_uint3_sat_rtn(const float3& v);

uint4 convert_uint4(const float4& v);
uint4 convert_uint4_rte(const float4& v);
uint4 convert_uint4_rtz(const float4& v);
uint4 convert_uint4_rtp(const float4& v);
uint4 convert_uint4_rtn(const float4& v);
uint4 convert_uint4_sat(const float4& v);
uint4 convert_uint4_sat_rte(const float4& v);
uint4 convert_uint4_sat_rtz(const float4& v);
uint4 convert_uint4_sat_rtp(const float4& v);
uint4 convert_uint4_sat_rtn(const float4& v);

uint8 convert_uint8(const float8& v);
uint8 convert_uint8_rte(const float8& v);
uint8 convert_uint8_rtz(const float8& v);
uint8 convert_uint8_rtp(const float8& v);
uint8 convert_uint8_rtn(const float8& v);
uint8 convert_uint8_sat(const float8& v);
uint8 convert_uint8_sat_rte(const float8& v);
uint8 convert_uint8_sat_rtz(const float8& v);
uint8 convert_uint8_sat_rtp(const float8& v);
uint8 convert_uint8_sat_rtn(const float8& v);

uint16 convert_uint16(const float16& v);
uint16 convert_uint16_rte(const float16& v);
uint16 convert_uint16_rtz(const float16& v);
uint16 convert_uint16_rtp(const float16& v);
uint16 convert_uint16_rtn(const float16& v);
uint16 convert_uint16_sat(const float16& v);
uint16 convert_uint16_sat_rte(const float16& v);
uint16 convert_uint16_sat_rtz(const float16& v);
uint16 convert_uint16_sat_rtp(const float16& v);
uint16 convert_uint16_sat_rtn(const float16& v);

long convert_long(const float& v);
long convert_long_rte(const float& v);
long convert_long_rtz(const float& v);
long convert_long_rtp(const float& v);
long convert_long_rtn(const float& v);
long convert_long_sat(const float& v);
long convert_long_sat_rte(const float& v);
long convert_long_sat_rtz(const float& v);
long convert_long_sat_rtp(const float& v);
long convert_long_sat_rtn(const float& v);

long2 convert_long2(const float2& v);
long2 convert_long2_rte(const float2& v);
long2 convert_long2_rtz(const float2& v);
long2 convert_long2_rtp(const float2& v);
long2 convert_long2_rtn(const float2& v);
long2 convert_long2_sat(const float2& v);
long2 convert_long2_sat_rte(const float2& v);
long2 convert_long2_sat_rtz(const float2& v);
long2 convert_long2_sat_rtp(const float2& v);
long2 convert_long2_sat_rtn(const float2& v);

long3 convert_long3(const float3& v);
long3 convert_long3_rte(const float3& v);
long3 convert_long3_rtz(const float3& v);
long3 convert_long3_rtp(const float3& v);
long3 convert_long3_rtn(const float3& v);
long3 convert_long3_sat(const float3& v);
long3 convert_long3_sat_rte(const float3& v);
long3 convert_long3_sat_rtz(const float3& v);
long3 convert_long3_sat_rtp(const float3& v);
long3 convert_long3_sat_rtn(const float3& v);

long4 convert_long4(const float4& v);
long4 convert_long4_rte(const float4& v);
long4 convert_long4_rtz(const float4& v);
long4 convert_long4_rtp(const float4& v);
long4 convert_long4_rtn(const float4& v);
long4 convert_long4_sat(const float4& v);
long4 convert_long4_sat_rte(const float4& v);
long4 convert_long4_sat_rtz(const float4& v);
long4 convert_long4_sat_rtp(const float4& v);
long4 convert_long4_sat_rtn(const float4& v);

long8 convert_long8(const float8& v);
long8 convert_long8_rte(const float8& v);
long8 convert_long8_rtz(const float8& v);
long8 convert_long8_rtp(const float8& v);
long8 convert_long8_rtn(const float8& v);
long8 convert_long8_sat(const float8& v);
long8 convert_long8_sat_rte(const float8& v);
long8 convert_long8_sat_rtz(const float8& v);
long8 convert_long8_sat_rtp(const float8& v);
long8 convert_long8_sat_rtn(const float8& v);

long16 convert_long16(const float16& v);
long16 convert_long16_rte(const float16& v);
long16 convert_long16_rtz(const float16& v);
long16 convert_long16_rtp(const float16& v);
long16 convert_long16_rtn(const float16& v);
long16 convert_long16_sat(const float16& v);
long16 convert_long16_sat_rte(const float16& v);
long16 convert_long16_sat_rtz(const float16& v);
long16 convert_long16_sat_rtp(const float16& v);
long16 convert_long16_sat_rtn(const float16& v);

ulong convert_ulong(const float& v);
ulong convert_ulong_rte(const float& v);
ulong convert_ulong_rtz(const float& v);
ulong convert_ulong_rtp(const float& v);
ulong convert_ulong_rtn(const float& v);
ulong convert_ulong_sat(const float& v);
ulong convert_ulong_sat_rte(const float& v);
ulong convert_ulong_sat_rtz(const float& v);
ulong convert_ulong_sat_rtp(const float& v);
ulong convert_ulong_sat_rtn(const float& v);

ulong2 convert_ulong2(const float2& v);
ulong2 convert_ulong2_rte(const float2& v);
ulong2 convert_ulong2_rtz(const float2& v);
ulong2 convert_ulong2_rtp(const float2& v);
ulong2 convert_ulong2_rtn(const float2& v);
ulong2 convert_ulong2_sat(const float2& v);
ulong2 convert_ulong2_sat_rte(const float2& v);
ulong2 convert_ulong2_sat_rtz(const float2& v);
ulong2 convert_ulong2_sat_rtp(const float2& v);
ulong2 convert_ulong2_sat_rtn(const float2& v);

ulong3 convert_ulong3(const float3& v);
ulong3 convert_ulong3_rte(const float3& v);
ulong3 convert_ulong3_rtz(const float3& v);
ulong3 convert_ulong3_rtp(const float3& v);
ulong3 convert_ulong3_rtn(const float3& v);
ulong3 convert_ulong3_sat(const float3& v);
ulong3 convert_ulong3_sat_rte(const float3& v);
ulong3 convert_ulong3_sat_rtz(const float3& v);
ulong3 convert_ulong3_sat_rtp(const float3& v);
ulong3 convert_ulong3_sat_rtn(const float3& v);

ulong4 convert_ulong4(const float4& v);
ulong4 convert_ulong4_rte(const float4& v);
ulong4 convert_ulong4_rtz(const float4& v);
ulong4 convert_ulong4_rtp(const float4& v);
ulong4 convert_ulong4_rtn(const float4& v);
ulong4 convert_ulong4_sat(const float4& v);
ulong4 convert_ulong4_sat_rte(const float4& v);
ulong4 convert_ulong4_sat_rtz(const float4& v);
ulong4 convert_ulong4_sat_rtp(const float4& v);
ulong4 convert_ulong4_sat_rtn(const float4& v);

ulong8 convert_ulong8(const float8& v);
ulong8 convert_ulong8_rte(const float8& v);
ulong8 convert_ulong8_rtz(const float8& v);
ulong8 convert_ulong8_rtp(const float8& v);
ulong8 convert_ulong8_rtn(const float8& v);
ulong8 convert_ulong8_sat(const float8& v);
ulong8 convert_ulong8_sat_rte(const float8& v);
ulong8 convert_ulong8_sat_rtz(const float8& v);
ulong8 convert_ulong8_sat_rtp(const float8& v);
ulong8 convert_ulong8_sat_rtn(const float8& v);

ulong16 convert_ulong16(const float16& v);
ulong16 convert_ulong16_rte(const float16& v);
ulong16 convert_ulong16_rtz(const float16& v);
ulong16 convert_ulong16_rtp(const float16& v);
ulong16 convert_ulong16_rtn(const float16& v);
ulong16 convert_ulong16_sat(const float16& v);
ulong16 convert_ulong16_sat_rte(const float16& v);
ulong16 convert_ulong16_sat_rtz(const float16& v);
ulong16 convert_ulong16_sat_rtp(const float16& v);
ulong16 convert_ulong16_sat_rtn(const float16& v);

char convert_char(const double& v);
char convert_char_rte(const double& v);
char convert_char_rtz(const double& v);
char convert_char_rtp(const double& v);
char convert_char_rtn(const double& v);
char convert_char_sat(const double& v);
char convert_char_sat_rte(const double& v);
char convert_char_sat_rtz(const double& v);
char convert_char_sat_rtp(const double& v);
char convert_char_sat_rtn(const double& v);

char2 convert_char2(const double2& v);
char2 convert_char2_rte(const double2& v);
char2 convert_char2_rtz(const double2& v);
char2 convert_char2_rtp(const double2& v);
char2 convert_char2_rtn(const double2& v);
char2 convert_char2_sat(const double2& v);
char2 convert_char2_sat_rte(const double2& v);
char2 convert_char2_sat_rtz(const double2& v);
char2 convert_char2_sat_rtp(const double2& v);
char2 convert_char2_sat_rtn(const double2& v);

char3 convert_char3(const double3& v);
char3 convert_char3_rte(const double3& v);
char3 convert_char3_rtz(const double3& v);
char3 convert_char3_rtp(const double3& v);
char3 convert_char3_rtn(const double3& v);
char3 convert_char3_sat(const double3& v);
char3 convert_char3_sat_rte(const double3& v);
char3 convert_char3_sat_rtz(const double3& v);
char3 convert_char3_sat_rtp(const double3& v);
char3 convert_char3_sat_rtn(const double3& v);

char4 convert_char4(const double4& v);
char4 convert_char4_rte(const double4& v);
char4 convert_char4_rtz(const double4& v);
char4 convert_char4_rtp(const double4& v);
char4 convert_char4_rtn(const double4& v);
char4 convert_char4_sat(const double4& v);
char4 convert_char4_sat_rte(const double4& v);
char4 convert_char4_sat_rtz(const double4& v);
char4 convert_char4_sat_rtp(const double4& v);
char4 convert_char4_sat_rtn(const double4& v);

char8 convert_char8(const double8& v);
char8 convert_char8_rte(const double8& v);
char8 convert_char8_rtz(const double8& v);
char8 convert_char8_rtp(const double8& v);
char8 convert_char8_rtn(const double8& v);
char8 convert_char8_sat(const double8& v);
char8 convert_char8_sat_rte(const double8& v);
char8 convert_char8_sat_rtz(const double8& v);
char8 convert_char8_sat_rtp(const double8& v);
char8 convert_char8_sat_rtn(const double8& v);

char16 convert_char16(const double16& v);
char16 convert_char16_rte(const double16& v);
char16 convert_char16_rtz(const double16& v);
char16 convert_char16_rtp(const double16& v);
char16 convert_char16_rtn(const double16& v);
char16 convert_char16_sat(const double16& v);
char16 convert_char16_sat_rte(const double16& v);
char16 convert_char16_sat_rtz(const double16& v);
char16 convert_char16_sat_rtp(const double16& v);
char16 convert_char16_sat_rtn(const double16& v);

uchar convert_uchar(const double& v);
uchar convert_uchar_rte(const double& v);
uchar convert_uchar_rtz(const double& v);
uchar convert_uchar_rtp(const double& v);
uchar convert_uchar_rtn(const double& v);
uchar convert_uchar_sat(const double& v);
uchar convert_uchar_sat_rte(const double& v);
uchar convert_uchar_sat_rtz(const double& v);
uchar convert_uchar_sat_rtp(const double& v);
uchar convert_uchar_sat_rtn(const double& v);

uchar2 convert_uchar2(const double2& v);
uchar2 convert_uchar2_rte(const double2& v);
uchar2 convert_uchar2_rtz(const double2& v);
uchar2 convert_uchar2_rtp(const double2& v);
uchar2 convert_uchar2_rtn(const double2& v);
uchar2 convert_uchar2_sat(const double2& v);
uchar2 convert_uchar2_sat_rte(const double2& v);
uchar2 convert_uchar2_sat_rtz(const double2& v);
uchar2 convert_uchar2_sat_rtp(const double2& v);
uchar2 convert_uchar2_sat_rtn(const double2& v);

uchar3 convert_uchar3(const double3& v);
uchar3 convert_uchar3_rte(const double3& v);
uchar3 convert_uchar3_rtz(const double3& v);
uchar3 convert_uchar3_rtp(const double3& v);
uchar3 convert_uchar3_rtn(const double3& v);
uchar3 convert_uchar3_sat(const double3& v);
uchar3 convert_uchar3_sat_rte(const double3& v);
uchar3 convert_uchar3_sat_rtz(const double3& v);
uchar3 convert_uchar3_sat_rtp(const double3& v);
uchar3 convert_uchar3_sat_rtn(const double3& v);

uchar4 convert_uchar4(const double4& v);
uchar4 convert_uchar4_rte(const double4& v);
uchar4 convert_uchar4_rtz(const double4& v);
uchar4 convert_uchar4_rtp(const double4& v);
uchar4 convert_uchar4_rtn(const double4& v);
uchar4 convert_uchar4_sat(const double4& v);
uchar4 convert_uchar4_sat_rte(const double4& v);
uchar4 convert_uchar4_sat_rtz(const double4& v);
uchar4 convert_uchar4_sat_rtp(const double4& v);
uchar4 convert_uchar4_sat_rtn(const double4& v);

uchar8 convert_uchar8(const double8& v);
uchar8 convert_uchar8_rte(const double8& v);
uchar8 convert_uchar8_rtz(const double8& v);
uchar8 convert_uchar8_rtp(const double8& v);
uchar8 convert_uchar8_rtn(const double8& v);
uchar8 convert_uchar8_sat(const double8& v);
uchar8 convert_uchar8_sat_rte(const double8& v);
uchar8 convert_uchar8_sat_rtz(const double8& v);
uchar8 convert_uchar8_sat_rtp(const double8& v);
uchar8 convert_uchar8_sat_rtn(const double8& v);

uchar16 convert_uchar16(const double16& v);
uchar16 convert_uchar16_rte(const double16& v);
uchar16 convert_uchar16_rtz(const double16& v);
uchar16 convert_uchar16_rtp(const double16& v);
uchar16 convert_uchar16_rtn(const double16& v);
uchar16 convert_uchar16_sat(const double16& v);
uchar16 convert_uchar16_sat_rte(const double16& v);
uchar16 convert_uchar16_sat_rtz(const double16& v);
uchar16 convert_uchar16_sat_rtp(const double16& v);
uchar16 convert_uchar16_sat_rtn(const double16& v);

short convert_short(const double& v);
short convert_short_rte(const double& v);
short convert_short_rtz(const double& v);
short convert_short_rtp(const double& v);
short convert_short_rtn(const double& v);
short convert_short_sat(const double& v);
short convert_short_sat_rte(const double& v);
short convert_short_sat_rtz(const double& v);
short convert_short_sat_rtp(const double& v);
short convert_short_sat_rtn(const double& v);

short2 convert_short2(const double2& v);
short2 convert_short2_rte(const double2& v);
short2 convert_short2_rtz(const double2& v);
short2 convert_short2_rtp(const double2& v);
short2 convert_short2_rtn(const double2& v);
short2 convert_short2_sat(const double2& v);
short2 convert_short2_sat_rte(const double2& v);
short2 convert_short2_sat_rtz(const double2& v);
short2 convert_short2_sat_rtp(const double2& v);
short2 convert_short2_sat_rtn(const double2& v);

short3 convert_short3(const double3& v);
short3 convert_short3_rte(const double3& v);
short3 convert_short3_rtz(const double3& v);
short3 convert_short3_rtp(const double3& v);
short3 convert_short3_rtn(const double3& v);
short3 convert_short3_sat(const double3& v);
short3 convert_short3_sat_rte(const double3& v);
short3 convert_short3_sat_rtz(const double3& v);
short3 convert_short3_sat_rtp(const double3& v);
short3 convert_short3_sat_rtn(const double3& v);

short4 convert_short4(const double4& v);
short4 convert_short4_rte(const double4& v);
short4 convert_short4_rtz(const double4& v);
short4 convert_short4_rtp(const double4& v);
short4 convert_short4_rtn(const double4& v);
short4 convert_short4_sat(const double4& v);
short4 convert_short4_sat_rte(const double4& v);
short4 convert_short4_sat_rtz(const double4& v);
short4 convert_short4_sat_rtp(const double4& v);
short4 convert_short4_sat_rtn(const double4& v);

short8 convert_short8(const double8& v);
short8 convert_short8_rte(const double8& v);
short8 convert_short8_rtz(const double8& v);
short8 convert_short8_rtp(const double8& v);
short8 convert_short8_rtn(const double8& v);
short8 convert_short8_sat(const double8& v);
short8 convert_short8_sat_rte(const double8& v);
short8 convert_short8_sat_rtz(const double8& v);
short8 convert_short8_sat_rtp(const double8& v);
short8 convert_short8_sat_rtn(const double8& v);

short16 convert_short16(const double16& v);
short16 convert_short16_rte(const double16& v);
short16 convert_short16_rtz(const double16& v);
short16 convert_short16_rtp(const double16& v);
short16 convert_short16_rtn(const double16& v);
short16 convert_short16_sat(const double16& v);
short16 convert_short16_sat_rte(const double16& v);
short16 convert_short16_sat_rtz(const double16& v);
short16 convert_short16_sat_rtp(const double16& v);
short16 convert_short16_sat_rtn(const double16& v);

ushort convert_ushort(const double& v);
ushort convert_ushort_rte(const double& v);
ushort convert_ushort_rtz(const double& v);
ushort convert_ushort_rtp(const double& v);
ushort convert_ushort_rtn(const double& v);
ushort convert_ushort_sat(const double& v);
ushort convert_ushort_sat_rte(const double& v);
ushort convert_ushort_sat_rtz(const double& v);
ushort convert_ushort_sat_rtp(const double& v);
ushort convert_ushort_sat_rtn(const double& v);

ushort2 convert_ushort2(const double2& v);
ushort2 convert_ushort2_rte(const double2& v);
ushort2 convert_ushort2_rtz(const double2& v);
ushort2 convert_ushort2_rtp(const double2& v);
ushort2 convert_ushort2_rtn(const double2& v);
ushort2 convert_ushort2_sat(const double2& v);
ushort2 convert_ushort2_sat_rte(const double2& v);
ushort2 convert_ushort2_sat_rtz(const double2& v);
ushort2 convert_ushort2_sat_rtp(const double2& v);
ushort2 convert_ushort2_sat_rtn(const double2& v);

ushort3 convert_ushort3(const double3& v);
ushort3 convert_ushort3_rte(const double3& v);
ushort3 convert_ushort3_rtz(const double3& v);
ushort3 convert_ushort3_rtp(const double3& v);
ushort3 convert_ushort3_rtn(const double3& v);
ushort3 convert_ushort3_sat(const double3& v);
ushort3 convert_ushort3_sat_rte(const double3& v);
ushort3 convert_ushort3_sat_rtz(const double3& v);
ushort3 convert_ushort3_sat_rtp(const double3& v);
ushort3 convert_ushort3_sat_rtn(const double3& v);

ushort4 convert_ushort4(const double4& v);
ushort4 convert_ushort4_rte(const double4& v);
ushort4 convert_ushort4_rtz(const double4& v);
ushort4 convert_ushort4_rtp(const double4& v);
ushort4 convert_ushort4_rtn(const double4& v);
ushort4 convert_ushort4_sat(const double4& v);
ushort4 convert_ushort4_sat_rte(const double4& v);
ushort4 convert_ushort4_sat_rtz(const double4& v);
ushort4 convert_ushort4_sat_rtp(const double4& v);
ushort4 convert_ushort4_sat_rtn(const double4& v);

ushort8 convert_ushort8(const double8& v);
ushort8 convert_ushort8_rte(const double8& v);
ushort8 convert_ushort8_rtz(const double8& v);
ushort8 convert_ushort8_rtp(const double8& v);
ushort8 convert_ushort8_rtn(const double8& v);
ushort8 convert_ushort8_sat(const double8& v);
ushort8 convert_ushort8_sat_rte(const double8& v);
ushort8 convert_ushort8_sat_rtz(const double8& v);
ushort8 convert_ushort8_sat_rtp(const double8& v);
ushort8 convert_ushort8_sat_rtn(const double8& v);

ushort16 convert_ushort16(const double16& v);
ushort16 convert_ushort16_rte(const double16& v);
ushort16 convert_ushort16_rtz(const double16& v);
ushort16 convert_ushort16_rtp(const double16& v);
ushort16 convert_ushort16_rtn(const double16& v);
ushort16 convert_ushort16_sat(const double16& v);
ushort16 convert_ushort16_sat_rte(const double16& v);
ushort16 convert_ushort16_sat_rtz(const double16& v);
ushort16 convert_ushort16_sat_rtp(const double16& v);
ushort16 convert_ushort16_sat_rtn(const double16& v);

int convert_int(const double& v);
int convert_int_rte(const double& v);
int convert_int_rtz(const double& v);
int convert_int_rtp(const double& v);
int convert_int_rtn(const double& v);
int convert_int_sat(const double& v);
int convert_int_sat_rte(const double& v);
int convert_int_sat_rtz(const double& v);
int convert_int_sat_rtp(const double& v);
int convert_int_sat_rtn(const double& v);

int2 convert_int2(const double2& v);
int2 convert_int2_rte(const double2& v);
int2 convert_int2_rtz(const double2& v);
int2 convert_int2_rtp(const double2& v);
int2 convert_int2_rtn(const double2& v);
int2 convert_int2_sat(const double2& v);
int2 convert_int2_sat_rte(const double2& v);
int2 convert_int2_sat_rtz(const double2& v);
int2 convert_int2_sat_rtp(const double2& v);
int2 convert_int2_sat_rtn(const double2& v);

int3 convert_int3(const double3& v);
int3 convert_int3_rte(const double3& v);
int3 convert_int3_rtz(const double3& v);
int3 convert_int3_rtp(const double3& v);
int3 convert_int3_rtn(const double3& v);
int3 convert_int3_sat(const double3& v);
int3 convert_int3_sat_rte(const double3& v);
int3 convert_int3_sat_rtz(const double3& v);
int3 convert_int3_sat_rtp(const double3& v);
int3 convert_int3_sat_rtn(const double3& v);

int4 convert_int4(const double4& v);
int4 convert_int4_rte(const double4& v);
int4 convert_int4_rtz(const double4& v);
int4 convert_int4_rtp(const double4& v);
int4 convert_int4_rtn(const double4& v);
int4 convert_int4_sat(const double4& v);
int4 convert_int4_sat_rte(const double4& v);
int4 convert_int4_sat_rtz(const double4& v);
int4 convert_int4_sat_rtp(const double4& v);
int4 convert_int4_sat_rtn(const double4& v);

int8 convert_int8(const double8& v);
int8 convert_int8_rte(const double8& v);
int8 convert_int8_rtz(const double8& v);
int8 convert_int8_rtp(const double8& v);
int8 convert_int8_rtn(const double8& v);
int8 convert_int8_sat(const double8& v);
int8 convert_int8_sat_rte(const double8& v);
int8 convert_int8_sat_rtz(const double8& v);
int8 convert_int8_sat_rtp(const double8& v);
int8 convert_int8_sat_rtn(const double8& v);

int16 convert_int16(const double16& v);
int16 convert_int16_rte(const double16& v);
int16 convert_int16_rtz(const double16& v);
int16 convert_int16_rtp(const double16& v);
int16 convert_int16_rtn(const double16& v);
int16 convert_int16_sat(const double16& v);
int16 convert_int16_sat_rte(const double16& v);
int16 convert_int16_sat_rtz(const double16& v);
int16 convert_int16_sat_rtp(const double16& v);
int16 convert_int16_sat_rtn(const double16& v);

uint convert_uint(const double& v);
uint convert_uint_rte(const double& v);
uint convert_uint_rtz(const double& v);
uint convert_uint_rtp(const double& v);
uint convert_uint_rtn(const double& v);
uint convert_uint_sat(const double& v);
uint convert_uint_sat_rte(const double& v);
uint convert_uint_sat_rtz(const double& v);
uint convert_uint_sat_rtp(const double& v);
uint convert_uint_sat_rtn(const double& v);

uint2 convert_uint2(const double2& v);
uint2 convert_uint2_rte(const double2& v);
uint2 convert_uint2_rtz(const double2& v);
uint2 convert_uint2_rtp(const double2& v);
uint2 convert_uint2_rtn(const double2& v);
uint2 convert_uint2_sat(const double2& v);
uint2 convert_uint2_sat_rte(const double2& v);
uint2 convert_uint2_sat_rtz(const double2& v);
uint2 convert_uint2_sat_rtp(const double2& v);
uint2 convert_uint2_sat_rtn(const double2& v);

uint3 convert_uint3(const double3& v);
uint3 convert_uint3_rte(const double3& v);
uint3 convert_uint3_rtz(const double3& v);
uint3 convert_uint3_rtp(const double3& v);
uint3 convert_uint3_rtn(const double3& v);
uint3 convert_uint3_sat(const double3& v);
uint3 convert_uint3_sat_rte(const double3& v);
uint3 convert_uint3_sat_rtz(const double3& v);
uint3 convert_uint3_sat_rtp(const double3& v);
uint3 convert_uint3_sat_rtn(const double3& v);

uint4 convert_uint4(const double4& v);
uint4 convert_uint4_rte(const double4& v);
uint4 convert_uint4_rtz(const double4& v);
uint4 convert_uint4_rtp(const double4& v);
uint4 convert_uint4_rtn(const double4& v);
uint4 convert_uint4_sat(const double4& v);
uint4 convert_uint4_sat_rte(const double4& v);
uint4 convert_uint4_sat_rtz(const double4& v);
uint4 convert_uint4_sat_rtp(const double4& v);
uint4 convert_uint4_sat_rtn(const double4& v);

uint8 convert_uint8(const double8& v);
uint8 convert_uint8_rte(const double8& v);
uint8 convert_uint8_rtz(const double8& v);
uint8 convert_uint8_rtp(const double8& v);
uint8 convert_uint8_rtn(const double8& v);
uint8 convert_uint8_sat(const double8& v);
uint8 convert_uint8_sat_rte(const double8& v);
uint8 convert_uint8_sat_rtz(const double8& v);
uint8 convert_uint8_sat_rtp(const double8& v);
uint8 convert_uint8_sat_rtn(const double8& v);

uint16 convert_uint16(const double16& v);
uint16 convert_uint16_rte(const double16& v);
uint16 convert_uint16_rtz(const double16& v);
uint16 convert_uint16_rtp(const double16& v);
uint16 convert_uint16_rtn(const double16& v);
uint16 convert_uint16_sat(const double16& v);
uint16 convert_uint16_sat_rte(const double16& v);
uint16 convert_uint16_sat_rtz(const double16& v);
uint16 convert_uint16_sat_rtp(const double16& v);
uint16 convert_uint16_sat_rtn(const double16& v);

long convert_long(const double& v);
long convert_long_rte(const double& v);
long convert_long_rtz(const double& v);
long convert_long_rtp(const double& v);
long convert_long_rtn(const double& v);
long convert_long_sat(const double& v);
long convert_long_sat_rte(const double& v);
long convert_long_sat_rtz(const double& v);
long convert_long_sat_rtp(const double& v);
long convert_long_sat_rtn(const double& v);

long2 convert_long2(const double2& v);
long2 convert_long2_rte(const double2& v);
long2 convert_long2_rtz(const double2& v);
long2 convert_long2_rtp(const double2& v);
long2 convert_long2_rtn(const double2& v);
long2 convert_long2_sat(const double2& v);
long2 convert_long2_sat_rte(const double2& v);
long2 convert_long2_sat_rtz(const double2& v);
long2 convert_long2_sat_rtp(const double2& v);
long2 convert_long2_sat_rtn(const double2& v);

long3 convert_long3(const double3& v);
long3 convert_long3_rte(const double3& v);
long3 convert_long3_rtz(const double3& v);
long3 convert_long3_rtp(const double3& v);
long3 convert_long3_rtn(const double3& v);
long3 convert_long3_sat(const double3& v);
long3 convert_long3_sat_rte(const double3& v);
long3 convert_long3_sat_rtz(const double3& v);
long3 convert_long3_sat_rtp(const double3& v);
long3 convert_long3_sat_rtn(const double3& v);

long4 convert_long4(const double4& v);
long4 convert_long4_rte(const double4& v);
long4 convert_long4_rtz(const double4& v);
long4 convert_long4_rtp(const double4& v);
long4 convert_long4_rtn(const double4& v);
long4 convert_long4_sat(const double4& v);
long4 convert_long4_sat_rte(const double4& v);
long4 convert_long4_sat_rtz(const double4& v);
long4 convert_long4_sat_rtp(const double4& v);
long4 convert_long4_sat_rtn(const double4& v);

long8 convert_long8(const double8& v);
long8 convert_long8_rte(const double8& v);
long8 convert_long8_rtz(const double8& v);
long8 convert_long8_rtp(const double8& v);
long8 convert_long8_rtn(const double8& v);
long8 convert_long8_sat(const double8& v);
long8 convert_long8_sat_rte(const double8& v);
long8 convert_long8_sat_rtz(const double8& v);
long8 convert_long8_sat_rtp(const double8& v);
long8 convert_long8_sat_rtn(const double8& v);

long16 convert_long16(const double16& v);
long16 convert_long16_rte(const double16& v);
long16 convert_long16_rtz(const double16& v);
long16 convert_long16_rtp(const double16& v);
long16 convert_long16_rtn(const double16& v);
long16 convert_long16_sat(const double16& v);
long16 convert_long16_sat_rte(const double16& v);
long16 convert_long16_sat_rtz(const double16& v);
long16 convert_long16_sat_rtp(const double16& v);
long16 convert_long16_sat_rtn(const double16& v);

ulong convert_ulong(const double& v);
ulong convert_ulong_rte(const double& v);
ulong convert_ulong_rtz(const double& v);
ulong convert_ulong_rtp(const double& v);
ulong convert_ulong_rtn(const double& v);
ulong convert_ulong_sat(const double& v);
ulong convert_ulong_sat_rte(const double& v);
ulong convert_ulong_sat_rtz(const double& v);
ulong convert_ulong_sat_rtp(const double& v);
ulong convert_ulong_sat_rtn(const double& v);

ulong2 convert_ulong2(const double2& v);
ulong2 convert_ulong2_rte(const double2& v);
ulong2 convert_ulong2_rtz(const double2& v);
ulong2 convert_ulong2_rtp(const double2& v);
ulong2 convert_ulong2_rtn(const double2& v);
ulong2 convert_ulong2_sat(const double2& v);
ulong2 convert_ulong2_sat_rte(const double2& v);
ulong2 convert_ulong2_sat_rtz(const double2& v);
ulong2 convert_ulong2_sat_rtp(const double2& v);
ulong2 convert_ulong2_sat_rtn(const double2& v);

ulong3 convert_ulong3(const double3& v);
ulong3 convert_ulong3_rte(const double3& v);
ulong3 convert_ulong3_rtz(const double3& v);
ulong3 convert_ulong3_rtp(const double3& v);
ulong3 convert_ulong3_rtn(const double3& v);
ulong3 convert_ulong3_sat(const double3& v);
ulong3 convert_ulong3_sat_rte(const double3& v);
ulong3 convert_ulong3_sat_rtz(const double3& v);
ulong3 convert_ulong3_sat_rtp(const double3& v);
ulong3 convert_ulong3_sat_rtn(const double3& v);

ulong4 convert_ulong4(const double4& v);
ulong4 convert_ulong4_rte(const double4& v);
ulong4 convert_ulong4_rtz(const double4& v);
ulong4 convert_ulong4_rtp(const double4& v);
ulong4 convert_ulong4_rtn(const double4& v);
ulong4 convert_ulong4_sat(const double4& v);
ulong4 convert_ulong4_sat_rte(const double4& v);
ulong4 convert_ulong4_sat_rtz(const double4& v);
ulong4 convert_ulong4_sat_rtp(const double4& v);
ulong4 convert_ulong4_sat_rtn(const double4& v);

ulong8 convert_ulong8(const double8& v);
ulong8 convert_ulong8_rte(const double8& v);
ulong8 convert_ulong8_rtz(const double8& v);
ulong8 convert_ulong8_rtp(const double8& v);
ulong8 convert_ulong8_rtn(const double8& v);
ulong8 convert_ulong8_sat(const double8& v);
ulong8 convert_ulong8_sat_rte(const double8& v);
ulong8 convert_ulong8_sat_rtz(const double8& v);
ulong8 convert_ulong8_sat_rtp(const double8& v);
ulong8 convert_ulong8_sat_rtn(const double8& v);

ulong16 convert_ulong16(const double16& v);
ulong16 convert_ulong16_rte(const double16& v);
ulong16 convert_ulong16_rtz(const double16& v);
ulong16 convert_ulong16_rtp(const double16& v);
ulong16 convert_ulong16_rtn(const double16& v);
ulong16 convert_ulong16_sat(const double16& v);
ulong16 convert_ulong16_sat_rte(const double16& v);
ulong16 convert_ulong16_sat_rtz(const double16& v);
ulong16 convert_ulong16_sat_rtp(const double16& v);
ulong16 convert_ulong16_sat_rtn(const double16& v);

float convert_float(const char& v);
float2 convert_float2(const char2& v);
float3 convert_float3(const char3& v);
float4 convert_float4(const char4& v);
float8 convert_float8(const char8& v);
float16 convert_float16(const char16& v);

float convert_float(const uchar& v);
float2 convert_float2(const uchar2& v);
float3 convert_float3(const uchar3& v);
float4 convert_float4(const uchar4& v);
float8 convert_float8(const uchar8& v);
float16 convert_float16(const uchar16& v);

float convert_float(const short& v);
float2 convert_float2(const short2& v);
float3 convert_float3(const short3& v);
float4 convert_float4(const short4& v);
float8 convert_float8(const short8& v);
float16 convert_float16(const short16& v);

float convert_float(const ushort& v);
float2 convert_float2(const ushort2& v);
float3 convert_float3(const ushort3& v);
float4 convert_float4(const ushort4& v);
float8 convert_float8(const ushort8& v);
float16 convert_float16(const ushort16& v);

float convert_float(const int& v);
float2 convert_float2(const int2& v);
float3 convert_float3(const int3& v);
float4 convert_float4(const int4& v);
float8 convert_float8(const int8& v);
float16 convert_float16(const int16& v);

float convert_float(const uint& v);
float2 convert_float2(const uint2& v);
float3 convert_float3(const uint3& v);
float4 convert_float4(const uint4& v);
float8 convert_float8(const uint8& v);
float16 convert_float16(const uint16& v);

float convert_float(const long& v);
float2 convert_float2(const long2& v);
float3 convert_float3(const long3& v);
float4 convert_float4(const long4& v);
float8 convert_float8(const long8& v);
float16 convert_float16(const long16& v);

float convert_float(const ulong& v);
float2 convert_float2(const ulong2& v);
float3 convert_float3(const ulong3& v);
float4 convert_float4(const ulong4& v);
float8 convert_float8(const ulong8& v);
float16 convert_float16(const ulong16& v);

double convert_double(const char& v);
double2 convert_double2(const char2& v);
double3 convert_double3(const char3& v);
double4 convert_double4(const char4& v);
double8 convert_double8(const char8& v);
double16 convert_double16(const char16& v);

double convert_double(const uchar& v);
double2 convert_double2(const uchar2& v);
double3 convert_double3(const uchar3& v);
double4 convert_double4(const uchar4& v);
double8 convert_double8(const uchar8& v);
double16 convert_double16(const uchar16& v);

double convert_double(const short& v);
double2 convert_double2(const short2& v);
double3 convert_double3(const short3& v);
double4 convert_double4(const short4& v);
double8 convert_double8(const short8& v);
double16 convert_double16(const short16& v);

double convert_double(const ushort& v);
double2 convert_double2(const ushort2& v);
double3 convert_double3(const ushort3& v);
double4 convert_double4(const ushort4& v);
double8 convert_double8(const ushort8& v);
double16 convert_double16(const ushort16& v);

double convert_double(const int& v);
double2 convert_double2(const int2& v);
double3 convert_double3(const int3& v);
double4 convert_double4(const int4& v);
double8 convert_double8(const int8& v);
double16 convert_double16(const int16& v);

double convert_double(const uint& v);
double2 convert_double2(const uint2& v);
double3 convert_double3(const uint3& v);
double4 convert_double4(const uint4& v);
double8 convert_double8(const uint8& v);
double16 convert_double16(const uint16& v);

double convert_double(const long& v);
double2 convert_double2(const long2& v);
double3 convert_double3(const long3& v);
double4 convert_double4(const long4& v);
double8 convert_double8(const long8& v);
double16 convert_double16(const long16& v);

double convert_double(const ulong& v);
double2 convert_double2(const ulong2& v);
double3 convert_double3(const ulong3& v);
double4 convert_double4(const ulong4& v);
double8 convert_double8(const ulong8& v);
double16 convert_double16(const ulong16& v);

float convert_float(const double& v);
float convert_float_rte(const double& v);
float convert_float_rtz(const double& v);
float convert_float_rtp(const double& v);
float convert_float_rtn(const double& v);

float2 convert_float2(const double2& v);
float2 convert_float2_rte(const double2& v);
float2 convert_float2_rtz(const double2& v);
float2 convert_float2_rtp(const double2& v);
float2 convert_float2_rtn(const double2& v);

float3 convert_float3(const double3& v);
float3 convert_float3_rte(const double3& v);
float3 convert_float3_rtz(const double3& v);
float3 convert_float3_rtp(const double3& v);
float3 convert_float3_rtn(const double3& v);

float4 convert_float4(const double4& v);
float4 convert_float4_rte(const double4& v);
float4 convert_float4_rtz(const double4& v);
float4 convert_float4_rtp(const double4& v);
float4 convert_float4_rtn(const double4& v);

float8 convert_float8(const double8& v);
float8 convert_float8_rte(const double8& v);
float8 convert_float8_rtz(const double8& v);
float8 convert_float8_rtp(const double8& v);
float8 convert_float8_rtn(const double8& v);

float16 convert_float16(const double16& v);
float16 convert_float16_rte(const double16& v);
float16 convert_float16_rtz(const double16& v);
float16 convert_float16_rtp(const double16& v);
float16 convert_float16_rtn(const double16& v);

double convert_double(const float& v);
double convert_double_rte(const float& v);
double convert_double_rtz(const float& v);
double convert_double_rtp(const float& v);
double convert_double_rtn(const float& v);

double2 convert_double2(const float2& v);
double2 convert_double2_rte(const float2& v);
double2 convert_double2_rtz(const float2& v);
double2 convert_double2_rtp(const float2& v);
double2 convert_double2_rtn(const float2& v);

double3 convert_double3(const float3& v);
double3 convert_double3_rte(const float3& v);
double3 convert_double3_rtz(const float3& v);
double3 convert_double3_rtp(const float3& v);
double3 convert_double3_rtn(const float3& v);

double4 convert_double4(const float4& v);
double4 convert_double4_rte(const float4& v);
double4 convert_double4_rtz(const float4& v);
double4 convert_double4_rtp(const float4& v);
double4 convert_double4_rtn(const float4& v);

double8 convert_double8(const float8& v);
double8 convert_double8_rte(const float8& v);
double8 convert_double8_rtz(const float8& v);
double8 convert_double8_rtp(const float8& v);
double8 convert_double8_rtn(const float8& v);

double16 convert_double16(const float16& v);
double16 convert_double16_rte(const float16& v);
double16 convert_double16_rtz(const float16& v);
double16 convert_double16_rtp(const float16& v);
double16 convert_double16_rtn(const float16& v);

char2 operator+(const char2& a, const char2& b);
char2 operator+(const char2& a);
char2 operator-(const char2& a, const char2& b);
char2 operator-(const char2& a);
char2 operator*(const char2& a, const char2& b);
char2 operator/(const char2& a, const char2& b);
char3 operator+(const char3& a, const char3& b);
char3 operator+(const char3& a);
char3 operator-(const char3& a, const char3& b);
char3 operator-(const char3& a);
char3 operator*(const char3& a, const char3& b);
char3 operator/(const char3& a, const char3& b);
char4 operator+(const char4& a, const char4& b);
char4 operator+(const char4& a);
char4 operator-(const char4& a, const char4& b);
char4 operator-(const char4& a);
char4 operator*(const char4& a, const char4& b);
char4 operator/(const char4& a, const char4& b);
char8 operator+(const char8& a, const char8& b);
char8 operator+(const char8& a);
char8 operator-(const char8& a, const char8& b);
char8 operator-(const char8& a);
char8 operator*(const char8& a, const char8& b);
char8 operator/(const char8& a, const char8& b);
char16 operator+(const char16& a, const char16& b);
char16 operator+(const char16& a);
char16 operator-(const char16& a, const char16& b);
char16 operator-(const char16& a);
char16 operator*(const char16& a, const char16& b);
char16 operator/(const char16& a, const char16& b);

uchar2 operator+(const uchar2& a, const uchar2& b);
uchar2 operator+(const uchar2& a);
uchar2 operator-(const uchar2& a, const uchar2& b);
uchar2 operator-(const uchar2& a);
uchar2 operator*(const uchar2& a, const uchar2& b);
uchar2 operator/(const uchar2& a, const uchar2& b);
uchar3 operator+(const uchar3& a, const uchar3& b);
uchar3 operator+(const uchar3& a);
uchar3 operator-(const uchar3& a, const uchar3& b);
uchar3 operator-(const uchar3& a);
uchar3 operator*(const uchar3& a, const uchar3& b);
uchar3 operator/(const uchar3& a, const uchar3& b);
uchar4 operator+(const uchar4& a, const uchar4& b);
uchar4 operator+(const uchar4& a);
uchar4 operator-(const uchar4& a, const uchar4& b);
uchar4 operator-(const uchar4& a);
uchar4 operator*(const uchar4& a, const uchar4& b);
uchar4 operator/(const uchar4& a, const uchar4& b);
uchar8 operator+(const uchar8& a, const uchar8& b);
uchar8 operator+(const uchar8& a);
uchar8 operator-(const uchar8& a, const uchar8& b);
uchar8 operator-(const uchar8& a);
uchar8 operator*(const uchar8& a, const uchar8& b);
uchar8 operator/(const uchar8& a, const uchar8& b);
uchar16 operator+(const uchar16& a, const uchar16& b);
uchar16 operator+(const uchar16& a);
uchar16 operator-(const uchar16& a, const uchar16& b);
uchar16 operator-(const uchar16& a);
uchar16 operator*(const uchar16& a, const uchar16& b);
uchar16 operator/(const uchar16& a, const uchar16& b);

short2 operator+(const short2& a, const short2& b);
short2 operator+(const short2& a);
short2 operator-(const short2& a, const short2& b);
short2 operator-(const short2& a);
short2 operator*(const short2& a, const short2& b);
short2 operator/(const short2& a, const short2& b);
short3 operator+(const short3& a, const short3& b);
short3 operator+(const short3& a);
short3 operator-(const short3& a, const short3& b);
short3 operator-(const short3& a);
short3 operator*(const short3& a, const short3& b);
short3 operator/(const short3& a, const short3& b);
short4 operator+(const short4& a, const short4& b);
short4 operator+(const short4& a);
short4 operator-(const short4& a, const short4& b);
short4 operator-(const short4& a);
short4 operator*(const short4& a, const short4& b);
short4 operator/(const short4& a, const short4& b);
short8 operator+(const short8& a, const short8& b);
short8 operator+(const short8& a);
short8 operator-(const short8& a, const short8& b);
short8 operator-(const short8& a);
short8 operator*(const short8& a, const short8& b);
short8 operator/(const short8& a, const short8& b);
short16 operator+(const short16& a, const short16& b);
short16 operator+(const short16& a);
short16 operator-(const short16& a, const short16& b);
short16 operator-(const short16& a);
short16 operator*(const short16& a, const short16& b);
short16 operator/(const short16& a, const short16& b);

ushort2 operator+(const ushort2& a, const ushort2& b);
ushort2 operator+(const ushort2& a);
ushort2 operator-(const ushort2& a, const ushort2& b);
ushort2 operator-(const ushort2& a);
ushort2 operator*(const ushort2& a, const ushort2& b);
ushort2 operator/(const ushort2& a, const ushort2& b);
ushort3 operator+(const ushort3& a, const ushort3& b);
ushort3 operator+(const ushort3& a);
ushort3 operator-(const ushort3& a, const ushort3& b);
ushort3 operator-(const ushort3& a);
ushort3 operator*(const ushort3& a, const ushort3& b);
ushort3 operator/(const ushort3& a, const ushort3& b);
ushort4 operator+(const ushort4& a, const ushort4& b);
ushort4 operator+(const ushort4& a);
ushort4 operator-(const ushort4& a, const ushort4& b);
ushort4 operator-(const ushort4& a);
ushort4 operator*(const ushort4& a, const ushort4& b);
ushort4 operator/(const ushort4& a, const ushort4& b);
ushort8 operator+(const ushort8& a, const ushort8& b);
ushort8 operator+(const ushort8& a);
ushort8 operator-(const ushort8& a, const ushort8& b);
ushort8 operator-(const ushort8& a);
ushort8 operator*(const ushort8& a, const ushort8& b);
ushort8 operator/(const ushort8& a, const ushort8& b);
ushort16 operator+(const ushort16& a, const ushort16& b);
ushort16 operator+(const ushort16& a);
ushort16 operator-(const ushort16& a, const ushort16& b);
ushort16 operator-(const ushort16& a);
ushort16 operator*(const ushort16& a, const ushort16& b);
ushort16 operator/(const ushort16& a, const ushort16& b);

int2 operator+(const int2& a, const int2& b);
int2 operator+(const int2& a);
int2 operator-(const int2& a, const int2& b);
int2 operator-(const int2& a);
int2 operator*(const int2& a, const int2& b);
int2 operator/(const int2& a, const int2& b);
int3 operator+(const int3& a, const int3& b);
int3 operator+(const int3& a);
int3 operator-(const int3& a, const int3& b);
int3 operator-(const int3& a);
int3 operator*(const int3& a, const int3& b);
int3 operator/(const int3& a, const int3& b);
int4 operator+(const int4& a, const int4& b);
int4 operator+(const int4& a);
int4 operator-(const int4& a, const int4& b);
int4 operator-(const int4& a);
int4 operator*(const int4& a, const int4& b);
int4 operator/(const int4& a, const int4& b);
int8 operator+(const int8& a, const int8& b);
int8 operator+(const int8& a);
int8 operator-(const int8& a, const int8& b);
int8 operator-(const int8& a);
int8 operator*(const int8& a, const int8& b);
int8 operator/(const int8& a, const int8& b);
int16 operator+(const int16& a, const int16& b);
int16 operator+(const int16& a);
int16 operator-(const int16& a, const int16& b);
int16 operator-(const int16& a);
int16 operator*(const int16& a, const int16& b);
int16 operator/(const int16& a, const int16& b);

uint2 operator+(const uint2& a, const uint2& b);
uint2 operator+(const uint2& a);
uint2 operator-(const uint2& a, const uint2& b);
uint2 operator-(const uint2& a);
uint2 operator*(const uint2& a, const uint2& b);
uint2 operator/(const uint2& a, const uint2& b);
uint3 operator+(const uint3& a, const uint3& b);
uint3 operator+(const uint3& a);
uint3 operator-(const uint3& a, const uint3& b);
uint3 operator-(const uint3& a);
uint3 operator*(const uint3& a, const uint3& b);
uint3 operator/(const uint3& a, const uint3& b);
uint4 operator+(const uint4& a, const uint4& b);
uint4 operator+(const uint4& a);
uint4 operator-(const uint4& a, const uint4& b);
uint4 operator-(const uint4& a);
uint4 operator*(const uint4& a, const uint4& b);
uint4 operator/(const uint4& a, const uint4& b);
uint8 operator+(const uint8& a, const uint8& b);
uint8 operator+(const uint8& a);
uint8 operator-(const uint8& a, const uint8& b);
uint8 operator-(const uint8& a);
uint8 operator*(const uint8& a, const uint8& b);
uint8 operator/(const uint8& a, const uint8& b);
uint16 operator+(const uint16& a, const uint16& b);
uint16 operator+(const uint16& a);
uint16 operator-(const uint16& a, const uint16& b);
uint16 operator-(const uint16& a);
uint16 operator*(const uint16& a, const uint16& b);
uint16 operator/(const uint16& a, const uint16& b);

long2 operator+(const long2& a, const long2& b);
long2 operator+(const long2& a);
long2 operator-(const long2& a, const long2& b);
long2 operator-(const long2& a);
long2 operator*(const long2& a, const long2& b);
long2 operator/(const long2& a, const long2& b);
long3 operator+(const long3& a, const long3& b);
long3 operator+(const long3& a);
long3 operator-(const long3& a, const long3& b);
long3 operator-(const long3& a);
long3 operator*(const long3& a, const long3& b);
long3 operator/(const long3& a, const long3& b);
long4 operator+(const long4& a, const long4& b);
long4 operator+(const long4& a);
long4 operator-(const long4& a, const long4& b);
long4 operator-(const long4& a);
long4 operator*(const long4& a, const long4& b);
long4 operator/(const long4& a, const long4& b);
long8 operator+(const long8& a, const long8& b);
long8 operator+(const long8& a);
long8 operator-(const long8& a, const long8& b);
long8 operator-(const long8& a);
long8 operator*(const long8& a, const long8& b);
long8 operator/(const long8& a, const long8& b);
long16 operator+(const long16& a, const long16& b);
long16 operator+(const long16& a);
long16 operator-(const long16& a, const long16& b);
long16 operator-(const long16& a);
long16 operator*(const long16& a, const long16& b);
long16 operator/(const long16& a, const long16& b);

ulong2 operator+(const ulong2& a, const ulong2& b);
ulong2 operator+(const ulong2& a);
ulong2 operator-(const ulong2& a, const ulong2& b);
ulong2 operator-(const ulong2& a);
ulong2 operator*(const ulong2& a, const ulong2& b);
ulong2 operator/(const ulong2& a, const ulong2& b);
ulong3 operator+(const ulong3& a, const ulong3& b);
ulong3 operator+(const ulong3& a);
ulong3 operator-(const ulong3& a, const ulong3& b);
ulong3 operator-(const ulong3& a);
ulong3 operator*(const ulong3& a, const ulong3& b);
ulong3 operator/(const ulong3& a, const ulong3& b);
ulong4 operator+(const ulong4& a, const ulong4& b);
ulong4 operator+(const ulong4& a);
ulong4 operator-(const ulong4& a, const ulong4& b);
ulong4 operator-(const ulong4& a);
ulong4 operator*(const ulong4& a, const ulong4& b);
ulong4 operator/(const ulong4& a, const ulong4& b);
ulong8 operator+(const ulong8& a, const ulong8& b);
ulong8 operator+(const ulong8& a);
ulong8 operator-(const ulong8& a, const ulong8& b);
ulong8 operator-(const ulong8& a);
ulong8 operator*(const ulong8& a, const ulong8& b);
ulong8 operator/(const ulong8& a, const ulong8& b);
ulong16 operator+(const ulong16& a, const ulong16& b);
ulong16 operator+(const ulong16& a);
ulong16 operator-(const ulong16& a, const ulong16& b);
ulong16 operator-(const ulong16& a);
ulong16 operator*(const ulong16& a, const ulong16& b);
ulong16 operator/(const ulong16& a, const ulong16& b);

float2 operator+(const float2& a, const float2& b);
float2 operator+(const float2& a);
float2 operator-(const float2& a, const float2& b);
float2 operator-(const float2& a);
float2 operator*(const float2& a, const float2& b);
float2 operator/(const float2& a, const float2& b);
float3 operator+(const float3& a, const float3& b);
float3 operator+(const float3& a);
float3 operator-(const float3& a, const float3& b);
float3 operator-(const float3& a);
float3 operator*(const float3& a, const float3& b);
float3 operator/(const float3& a, const float3& b);
float4 operator+(const float4& a, const float4& b);
float4 operator+(const float4& a);
float4 operator-(const float4& a, const float4& b);
float4 operator-(const float4& a);
float4 operator*(const float4& a, const float4& b);
float4 operator/(const float4& a, const float4& b);
float8 operator+(const float8& a, const float8& b);
float8 operator+(const float8& a);
float8 operator-(const float8& a, const float8& b);
float8 operator-(const float8& a);
float8 operator*(const float8& a, const float8& b);
float8 operator/(const float8& a, const float8& b);
float16 operator+(const float16& a, const float16& b);
float16 operator+(const float16& a);
float16 operator-(const float16& a, const float16& b);
float16 operator-(const float16& a);
float16 operator*(const float16& a, const float16& b);
float16 operator/(const float16& a, const float16& b);

double2 operator+(const double2& a, const double2& b);
double2 operator+(const double2& a);
double2 operator-(const double2& a, const double2& b);
double2 operator-(const double2& a);
double2 operator*(const double2& a, const double2& b);
double2 operator/(const double2& a, const double2& b);
double3 operator+(const double3& a, const double3& b);
double3 operator+(const double3& a);
double3 operator-(const double3& a, const double3& b);
double3 operator-(const double3& a);
double3 operator*(const double3& a, const double3& b);
double3 operator/(const double3& a, const double3& b);
double4 operator+(const double4& a, const double4& b);
double4 operator+(const double4& a);
double4 operator-(const double4& a, const double4& b);
double4 operator-(const double4& a);
double4 operator*(const double4& a, const double4& b);
double4 operator/(const double4& a, const double4& b);
double8 operator+(const double8& a, const double8& b);
double8 operator+(const double8& a);
double8 operator-(const double8& a, const double8& b);
double8 operator-(const double8& a);
double8 operator*(const double8& a, const double8& b);
double8 operator/(const double8& a, const double8& b);
double16 operator+(const double16& a, const double16& b);
double16 operator+(const double16& a);
double16 operator-(const double16& a, const double16& b);
double16 operator-(const double16& a);
double16 operator*(const double16& a, const double16& b);
double16 operator/(const double16& a, const double16& b);

char2 operator==(const char2& a, const char2& b);
char2 operator!=(const char2& a, const char2& b);
char2 operator>(const char2& a, const char2& b);
char2 operator>=(const char2& a, const char2& b);
char2 operator<(const char2& a, const char2& b);
char2 operator<=(const char2& a, const char2& b);
char2 operator&&(const char2& a, const char2& b);
char2 operator||(const char2& a, const char2& b);
char2 operator!(const char2& a);
char3 operator==(const char3& a, const char3& b);
char3 operator!=(const char3& a, const char3& b);
char3 operator>(const char3& a, const char3& b);
char3 operator>=(const char3& a, const char3& b);
char3 operator<(const char3& a, const char3& b);
char3 operator<=(const char3& a, const char3& b);
char3 operator&&(const char3& a, const char3& b);
char3 operator||(const char3& a, const char3& b);
char3 operator!(const char3& a);
char4 operator==(const char4& a, const char4& b);
char4 operator!=(const char4& a, const char4& b);
char4 operator>(const char4& a, const char4& b);
char4 operator>=(const char4& a, const char4& b);
char4 operator<(const char4& a, const char4& b);
char4 operator<=(const char4& a, const char4& b);
char4 operator&&(const char4& a, const char4& b);
char4 operator||(const char4& a, const char4& b);
char4 operator!(const char4& a);
char8 operator==(const char8& a, const char8& b);
char8 operator!=(const char8& a, const char8& b);
char8 operator>(const char8& a, const char8& b);
char8 operator>=(const char8& a, const char8& b);
char8 operator<(const char8& a, const char8& b);
char8 operator<=(const char8& a, const char8& b);
char8 operator&&(const char8& a, const char8& b);
char8 operator||(const char8& a, const char8& b);
char8 operator!(const char8& a);
char16 operator==(const char16& a, const char16& b);
char16 operator!=(const char16& a, const char16& b);
char16 operator>(const char16& a, const char16& b);
char16 operator>=(const char16& a, const char16& b);
char16 operator<(const char16& a, const char16& b);
char16 operator<=(const char16& a, const char16& b);
char16 operator&&(const char16& a, const char16& b);
char16 operator||(const char16& a, const char16& b);
char16 operator!(const char16& a);

char2 operator==(const uchar2& a, const uchar2& b);
char2 operator!=(const uchar2& a, const uchar2& b);
char2 operator>(const uchar2& a, const uchar2& b);
char2 operator>=(const uchar2& a, const uchar2& b);
char2 operator<(const uchar2& a, const uchar2& b);
char2 operator<=(const uchar2& a, const uchar2& b);
char2 operator&&(const uchar2& a, const uchar2& b);
char2 operator||(const uchar2& a, const uchar2& b);
char2 operator!(const uchar2& a);
char3 operator==(const uchar3& a, const uchar3& b);
char3 operator!=(const uchar3& a, const uchar3& b);
char3 operator>(const uchar3& a, const uchar3& b);
char3 operator>=(const uchar3& a, const uchar3& b);
char3 operator<(const uchar3& a, const uchar3& b);
char3 operator<=(const uchar3& a, const uchar3& b);
char3 operator&&(const uchar3& a, const uchar3& b);
char3 operator||(const uchar3& a, const uchar3& b);
char3 operator!(const uchar3& a);
char4 operator==(const uchar4& a, const uchar4& b);
char4 operator!=(const uchar4& a, const uchar4& b);
char4 operator>(const uchar4& a, const uchar4& b);
char4 operator>=(const uchar4& a, const uchar4& b);
char4 operator<(const uchar4& a, const uchar4& b);
char4 operator<=(const uchar4& a, const uchar4& b);
char4 operator&&(const uchar4& a, const uchar4& b);
char4 operator||(const uchar4& a, const uchar4& b);
char4 operator!(const uchar4& a);
char8 operator==(const uchar8& a, const uchar8& b);
char8 operator!=(const uchar8& a, const uchar8& b);
char8 operator>(const uchar8& a, const uchar8& b);
char8 operator>=(const uchar8& a, const uchar8& b);
char8 operator<(const uchar8& a, const uchar8& b);
char8 operator<=(const uchar8& a, const uchar8& b);
char8 operator&&(const uchar8& a, const uchar8& b);
char8 operator||(const uchar8& a, const uchar8& b);
char8 operator!(const uchar8& a);
char16 operator==(const uchar16& a, const uchar16& b);
char16 operator!=(const uchar16& a, const uchar16& b);
char16 operator>(const uchar16& a, const uchar16& b);
char16 operator>=(const uchar16& a, const uchar16& b);
char16 operator<(const uchar16& a, const uchar16& b);
char16 operator<=(const uchar16& a, const uchar16& b);
char16 operator&&(const uchar16& a, const uchar16& b);
char16 operator||(const uchar16& a, const uchar16& b);
char16 operator!(const uchar16& a);

short2 operator==(const short2& a, const short2& b);
short2 operator!=(const short2& a, const short2& b);
short2 operator>(const short2& a, const short2& b);
short2 operator>=(const short2& a, const short2& b);
short2 operator<(const short2& a, const short2& b);
short2 operator<=(const short2& a, const short2& b);
short2 operator&&(const short2& a, const short2& b);
short2 operator||(const short2& a, const short2& b);
short2 operator!(const short2& a);
short3 operator==(const short3& a, const short3& b);
short3 operator!=(const short3& a, const short3& b);
short3 operator>(const short3& a, const short3& b);
short3 operator>=(const short3& a, const short3& b);
short3 operator<(const short3& a, const short3& b);
short3 operator<=(const short3& a, const short3& b);
short3 operator&&(const short3& a, const short3& b);
short3 operator||(const short3& a, const short3& b);
short3 operator!(const short3& a);
short4 operator==(const short4& a, const short4& b);
short4 operator!=(const short4& a, const short4& b);
short4 operator>(const short4& a, const short4& b);
short4 operator>=(const short4& a, const short4& b);
short4 operator<(const short4& a, const short4& b);
short4 operator<=(const short4& a, const short4& b);
short4 operator&&(const short4& a, const short4& b);
short4 operator||(const short4& a, const short4& b);
short4 operator!(const short4& a);
short8 operator==(const short8& a, const short8& b);
short8 operator!=(const short8& a, const short8& b);
short8 operator>(const short8& a, const short8& b);
short8 operator>=(const short8& a, const short8& b);
short8 operator<(const short8& a, const short8& b);
short8 operator<=(const short8& a, const short8& b);
short8 operator&&(const short8& a, const short8& b);
short8 operator||(const short8& a, const short8& b);
short8 operator!(const short8& a);
short16 operator==(const short16& a, const short16& b);
short16 operator!=(const short16& a, const short16& b);
short16 operator>(const short16& a, const short16& b);
short16 operator>=(const short16& a, const short16& b);
short16 operator<(const short16& a, const short16& b);
short16 operator<=(const short16& a, const short16& b);
short16 operator&&(const short16& a, const short16& b);
short16 operator||(const short16& a, const short16& b);
short16 operator!(const short16& a);

short2 operator==(const ushort2& a, const ushort2& b);
short2 operator!=(const ushort2& a, const ushort2& b);
short2 operator>(const ushort2& a, const ushort2& b);
short2 operator>=(const ushort2& a, const ushort2& b);
short2 operator<(const ushort2& a, const ushort2& b);
short2 operator<=(const ushort2& a, const ushort2& b);
short2 operator&&(const ushort2& a, const ushort2& b);
short2 operator||(const ushort2& a, const ushort2& b);
short2 operator!(const ushort2& a);
short3 operator==(const ushort3& a, const ushort3& b);
short3 operator!=(const ushort3& a, const ushort3& b);
short3 operator>(const ushort3& a, const ushort3& b);
short3 operator>=(const ushort3& a, const ushort3& b);
short3 operator<(const ushort3& a, const ushort3& b);
short3 operator<=(const ushort3& a, const ushort3& b);
short3 operator&&(const ushort3& a, const ushort3& b);
short3 operator||(const ushort3& a, const ushort3& b);
short3 operator!(const ushort3& a);
short4 operator==(const ushort4& a, const ushort4& b);
short4 operator!=(const ushort4& a, const ushort4& b);
short4 operator>(const ushort4& a, const ushort4& b);
short4 operator>=(const ushort4& a, const ushort4& b);
short4 operator<(const ushort4& a, const ushort4& b);
short4 operator<=(const ushort4& a, const ushort4& b);
short4 operator&&(const ushort4& a, const ushort4& b);
short4 operator||(const ushort4& a, const ushort4& b);
short4 operator!(const ushort4& a);
short8 operator==(const ushort8& a, const ushort8& b);
short8 operator!=(const ushort8& a, const ushort8& b);
short8 operator>(const ushort8& a, const ushort8& b);
short8 operator>=(const ushort8& a, const ushort8& b);
short8 operator<(const ushort8& a, const ushort8& b);
short8 operator<=(const ushort8& a, const ushort8& b);
short8 operator&&(const ushort8& a, const ushort8& b);
short8 operator||(const ushort8& a, const ushort8& b);
short8 operator!(const ushort8& a);
short16 operator==(const ushort16& a, const ushort16& b);
short16 operator!=(const ushort16& a, const ushort16& b);
short16 operator>(const ushort16& a, const ushort16& b);
short16 operator>=(const ushort16& a, const ushort16& b);
short16 operator<(const ushort16& a, const ushort16& b);
short16 operator<=(const ushort16& a, const ushort16& b);
short16 operator&&(const ushort16& a, const ushort16& b);
short16 operator||(const ushort16& a, const ushort16& b);
short16 operator!(const ushort16& a);

int2 operator==(const int2& a, const int2& b);
int2 operator!=(const int2& a, const int2& b);
int2 operator>(const int2& a, const int2& b);
int2 operator>=(const int2& a, const int2& b);
int2 operator<(const int2& a, const int2& b);
int2 operator<=(const int2& a, const int2& b);
int2 operator&&(const int2& a, const int2& b);
int2 operator||(const int2& a, const int2& b);
int2 operator!(const int2& a);
int3 operator==(const int3& a, const int3& b);
int3 operator!=(const int3& a, const int3& b);
int3 operator>(const int3& a, const int3& b);
int3 operator>=(const int3& a, const int3& b);
int3 operator<(const int3& a, const int3& b);
int3 operator<=(const int3& a, const int3& b);
int3 operator&&(const int3& a, const int3& b);
int3 operator||(const int3& a, const int3& b);
int3 operator!(const int3& a);
int4 operator==(const int4& a, const int4& b);
int4 operator!=(const int4& a, const int4& b);
int4 operator>(const int4& a, const int4& b);
int4 operator>=(const int4& a, const int4& b);
int4 operator<(const int4& a, const int4& b);
int4 operator<=(const int4& a, const int4& b);
int4 operator&&(const int4& a, const int4& b);
int4 operator||(const int4& a, const int4& b);
int4 operator!(const int4& a);
int8 operator==(const int8& a, const int8& b);
int8 operator!=(const int8& a, const int8& b);
int8 operator>(const int8& a, const int8& b);
int8 operator>=(const int8& a, const int8& b);
int8 operator<(const int8& a, const int8& b);
int8 operator<=(const int8& a, const int8& b);
int8 operator&&(const int8& a, const int8& b);
int8 operator||(const int8& a, const int8& b);
int8 operator!(const int8& a);
int16 operator==(const int16& a, const int16& b);
int16 operator!=(const int16& a, const int16& b);
int16 operator>(const int16& a, const int16& b);
int16 operator>=(const int16& a, const int16& b);
int16 operator<(const int16& a, const int16& b);
int16 operator<=(const int16& a, const int16& b);
int16 operator&&(const int16& a, const int16& b);
int16 operator||(const int16& a, const int16& b);
int16 operator!(const int16& a);

int2 operator==(const uint2& a, const uint2& b);
int2 operator!=(const uint2& a, const uint2& b);
int2 operator>(const uint2& a, const uint2& b);
int2 operator>=(const uint2& a, const uint2& b);
int2 operator<(const uint2& a, const uint2& b);
int2 operator<=(const uint2& a, const uint2& b);
int2 operator&&(const uint2& a, const uint2& b);
int2 operator||(const uint2& a, const uint2& b);
int2 operator!(const uint2& a);
int3 operator==(const uint3& a, const uint3& b);
int3 operator!=(const uint3& a, const uint3& b);
int3 operator>(const uint3& a, const uint3& b);
int3 operator>=(const uint3& a, const uint3& b);
int3 operator<(const uint3& a, const uint3& b);
int3 operator<=(const uint3& a, const uint3& b);
int3 operator&&(const uint3& a, const uint3& b);
int3 operator||(const uint3& a, const uint3& b);
int3 operator!(const uint3& a);
int4 operator==(const uint4& a, const uint4& b);
int4 operator!=(const uint4& a, const uint4& b);
int4 operator>(const uint4& a, const uint4& b);
int4 operator>=(const uint4& a, const uint4& b);
int4 operator<(const uint4& a, const uint4& b);
int4 operator<=(const uint4& a, const uint4& b);
int4 operator&&(const uint4& a, const uint4& b);
int4 operator||(const uint4& a, const uint4& b);
int4 operator!(const uint4& a);
int8 operator==(const uint8& a, const uint8& b);
int8 operator!=(const uint8& a, const uint8& b);
int8 operator>(const uint8& a, const uint8& b);
int8 operator>=(const uint8& a, const uint8& b);
int8 operator<(const uint8& a, const uint8& b);
int8 operator<=(const uint8& a, const uint8& b);
int8 operator&&(const uint8& a, const uint8& b);
int8 operator||(const uint8& a, const uint8& b);
int8 operator!(const uint8& a);
int16 operator==(const uint16& a, const uint16& b);
int16 operator!=(const uint16& a, const uint16& b);
int16 operator>(const uint16& a, const uint16& b);
int16 operator>=(const uint16& a, const uint16& b);
int16 operator<(const uint16& a, const uint16& b);
int16 operator<=(const uint16& a, const uint16& b);
int16 operator&&(const uint16& a, const uint16& b);
int16 operator||(const uint16& a, const uint16& b);
int16 operator!(const uint16& a);

long2 operator==(const long2& a, const long2& b);
long2 operator!=(const long2& a, const long2& b);
long2 operator>(const long2& a, const long2& b);
long2 operator>=(const long2& a, const long2& b);
long2 operator<(const long2& a, const long2& b);
long2 operator<=(const long2& a, const long2& b);
long2 operator&&(const long2& a, const long2& b);
long2 operator||(const long2& a, const long2& b);
long2 operator!(const long2& a);
long3 operator==(const long3& a, const long3& b);
long3 operator!=(const long3& a, const long3& b);
long3 operator>(const long3& a, const long3& b);
long3 operator>=(const long3& a, const long3& b);
long3 operator<(const long3& a, const long3& b);
long3 operator<=(const long3& a, const long3& b);
long3 operator&&(const long3& a, const long3& b);
long3 operator||(const long3& a, const long3& b);
long3 operator!(const long3& a);
long4 operator==(const long4& a, const long4& b);
long4 operator!=(const long4& a, const long4& b);
long4 operator>(const long4& a, const long4& b);
long4 operator>=(const long4& a, const long4& b);
long4 operator<(const long4& a, const long4& b);
long4 operator<=(const long4& a, const long4& b);
long4 operator&&(const long4& a, const long4& b);
long4 operator||(const long4& a, const long4& b);
long4 operator!(const long4& a);
long8 operator==(const long8& a, const long8& b);
long8 operator!=(const long8& a, const long8& b);
long8 operator>(const long8& a, const long8& b);
long8 operator>=(const long8& a, const long8& b);
long8 operator<(const long8& a, const long8& b);
long8 operator<=(const long8& a, const long8& b);
long8 operator&&(const long8& a, const long8& b);
long8 operator||(const long8& a, const long8& b);
long8 operator!(const long8& a);
long16 operator==(const long16& a, const long16& b);
long16 operator!=(const long16& a, const long16& b);
long16 operator>(const long16& a, const long16& b);
long16 operator>=(const long16& a, const long16& b);
long16 operator<(const long16& a, const long16& b);
long16 operator<=(const long16& a, const long16& b);
long16 operator&&(const long16& a, const long16& b);
long16 operator||(const long16& a, const long16& b);
long16 operator!(const long16& a);

long2 operator==(const ulong2& a, const ulong2& b);
long2 operator!=(const ulong2& a, const ulong2& b);
long2 operator>(const ulong2& a, const ulong2& b);
long2 operator>=(const ulong2& a, const ulong2& b);
long2 operator<(const ulong2& a, const ulong2& b);
long2 operator<=(const ulong2& a, const ulong2& b);
long2 operator&&(const ulong2& a, const ulong2& b);
long2 operator||(const ulong2& a, const ulong2& b);
long2 operator!(const ulong2& a);
long3 operator==(const ulong3& a, const ulong3& b);
long3 operator!=(const ulong3& a, const ulong3& b);
long3 operator>(const ulong3& a, const ulong3& b);
long3 operator>=(const ulong3& a, const ulong3& b);
long3 operator<(const ulong3& a, const ulong3& b);
long3 operator<=(const ulong3& a, const ulong3& b);
long3 operator&&(const ulong3& a, const ulong3& b);
long3 operator||(const ulong3& a, const ulong3& b);
long3 operator!(const ulong3& a);
long4 operator==(const ulong4& a, const ulong4& b);
long4 operator!=(const ulong4& a, const ulong4& b);
long4 operator>(const ulong4& a, const ulong4& b);
long4 operator>=(const ulong4& a, const ulong4& b);
long4 operator<(const ulong4& a, const ulong4& b);
long4 operator<=(const ulong4& a, const ulong4& b);
long4 operator&&(const ulong4& a, const ulong4& b);
long4 operator||(const ulong4& a, const ulong4& b);
long4 operator!(const ulong4& a);
long8 operator==(const ulong8& a, const ulong8& b);
long8 operator!=(const ulong8& a, const ulong8& b);
long8 operator>(const ulong8& a, const ulong8& b);
long8 operator>=(const ulong8& a, const ulong8& b);
long8 operator<(const ulong8& a, const ulong8& b);
long8 operator<=(const ulong8& a, const ulong8& b);
long8 operator&&(const ulong8& a, const ulong8& b);
long8 operator||(const ulong8& a, const ulong8& b);
long8 operator!(const ulong8& a);
long16 operator==(const ulong16& a, const ulong16& b);
long16 operator!=(const ulong16& a, const ulong16& b);
long16 operator>(const ulong16& a, const ulong16& b);
long16 operator>=(const ulong16& a, const ulong16& b);
long16 operator<(const ulong16& a, const ulong16& b);
long16 operator<=(const ulong16& a, const ulong16& b);
long16 operator&&(const ulong16& a, const ulong16& b);
long16 operator||(const ulong16& a, const ulong16& b);
long16 operator!(const ulong16& a);

int2 operator==(const float2& a, const float2& b);
int2 operator!=(const float2& a, const float2& b);
int2 operator>(const float2& a, const float2& b);
int2 operator>=(const float2& a, const float2& b);
int2 operator<(const float2& a, const float2& b);
int2 operator<=(const float2& a, const float2& b);
int2 operator&&(const float2& a, const float2& b);
int2 operator||(const float2& a, const float2& b);
int2 operator!(const float2& a);
int3 operator==(const float3& a, const float3& b);
int3 operator!=(const float3& a, const float3& b);
int3 operator>(const float3& a, const float3& b);
int3 operator>=(const float3& a, const float3& b);
int3 operator<(const float3& a, const float3& b);
int3 operator<=(const float3& a, const float3& b);
int3 operator&&(const float3& a, const float3& b);
int3 operator||(const float3& a, const float3& b);
int3 operator!(const float3& a);
int4 operator==(const float4& a, const float4& b);
int4 operator!=(const float4& a, const float4& b);
int4 operator>(const float4& a, const float4& b);
int4 operator>=(const float4& a, const float4& b);
int4 operator<(const float4& a, const float4& b);
int4 operator<=(const float4& a, const float4& b);
int4 operator&&(const float4& a, const float4& b);
int4 operator||(const float4& a, const float4& b);
int4 operator!(const float4& a);
int8 operator==(const float8& a, const float8& b);
int8 operator!=(const float8& a, const float8& b);
int8 operator>(const float8& a, const float8& b);
int8 operator>=(const float8& a, const float8& b);
int8 operator<(const float8& a, const float8& b);
int8 operator<=(const float8& a, const float8& b);
int8 operator&&(const float8& a, const float8& b);
int8 operator||(const float8& a, const float8& b);
int8 operator!(const float8& a);
int16 operator==(const float16& a, const float16& b);
int16 operator!=(const float16& a, const float16& b);
int16 operator>(const float16& a, const float16& b);
int16 operator>=(const float16& a, const float16& b);
int16 operator<(const float16& a, const float16& b);
int16 operator<=(const float16& a, const float16& b);
int16 operator&&(const float16& a, const float16& b);
int16 operator||(const float16& a, const float16& b);
int16 operator!(const float16& a);

long2 operator==(const double2& a, const double2& b);
long2 operator!=(const double2& a, const double2& b);
long2 operator>(const double2& a, const double2& b);
long2 operator>=(const double2& a, const double2& b);
long2 operator<(const double2& a, const double2& b);
long2 operator<=(const double2& a, const double2& b);
long2 operator&&(const double2& a, const double2& b);
long2 operator||(const double2& a, const double2& b);
long2 operator!(const double2& a);
long3 operator==(const double3& a, const double3& b);
long3 operator!=(const double3& a, const double3& b);
long3 operator>(const double3& a, const double3& b);
long3 operator>=(const double3& a, const double3& b);
long3 operator<(const double3& a, const double3& b);
long3 operator<=(const double3& a, const double3& b);
long3 operator&&(const double3& a, const double3& b);
long3 operator||(const double3& a, const double3& b);
long3 operator!(const double3& a);
long4 operator==(const double4& a, const double4& b);
long4 operator!=(const double4& a, const double4& b);
long4 operator>(const double4& a, const double4& b);
long4 operator>=(const double4& a, const double4& b);
long4 operator<(const double4& a, const double4& b);
long4 operator<=(const double4& a, const double4& b);
long4 operator&&(const double4& a, const double4& b);
long4 operator||(const double4& a, const double4& b);
long4 operator!(const double4& a);
long8 operator==(const double8& a, const double8& b);
long8 operator!=(const double8& a, const double8& b);
long8 operator>(const double8& a, const double8& b);
long8 operator>=(const double8& a, const double8& b);
long8 operator<(const double8& a, const double8& b);
long8 operator<=(const double8& a, const double8& b);
long8 operator&&(const double8& a, const double8& b);
long8 operator||(const double8& a, const double8& b);
long8 operator!(const double8& a);
long16 operator==(const double16& a, const double16& b);
long16 operator!=(const double16& a, const double16& b);
long16 operator>(const double16& a, const double16& b);
long16 operator>=(const double16& a, const double16& b);
long16 operator<(const double16& a, const double16& b);
long16 operator<=(const double16& a, const double16& b);
long16 operator&&(const double16& a, const double16& b);
long16 operator||(const double16& a, const double16& b);
long16 operator!(const double16& a);

char2 operator%(const char2& a, const char2& b);
char2 operator&(const char2& a, const char2& b);
char2 operator|(const char2& a, const char2& b);
char2 operator^(const char2& a, const char2& b);
char2 operator~(const char2& a);
char2 operator++(const char2& a);
char2 operator++(const char2& a, int dummy);
char2 operator--(const char2& a);
char2 operator--(const char2& a, int dummy);
char2 operator<<(const char2& a, int bits);
char2 operator>>(const char2& a, int bits);
char3 operator%(const char3& a, const char3& b);
char3 operator&(const char3& a, const char3& b);
char3 operator|(const char3& a, const char3& b);
char3 operator^(const char3& a, const char3& b);
char3 operator~(const char3& a);
char3 operator++(const char3& a);
char3 operator++(const char3& a, int dummy);
char3 operator--(const char3& a);
char3 operator--(const char3& a, int dummy);
char3 operator<<(const char3& a, int bits);
char3 operator>>(const char3& a, int bits);
char4 operator%(const char4& a, const char4& b);
char4 operator&(const char4& a, const char4& b);
char4 operator|(const char4& a, const char4& b);
char4 operator^(const char4& a, const char4& b);
char4 operator~(const char4& a);
char4 operator++(const char4& a);
char4 operator++(const char4& a, int dummy);
char4 operator--(const char4& a);
char4 operator--(const char4& a, int dummy);
char4 operator<<(const char4& a, int bits);
char4 operator>>(const char4& a, int bits);
char8 operator%(const char8& a, const char8& b);
char8 operator&(const char8& a, const char8& b);
char8 operator|(const char8& a, const char8& b);
char8 operator^(const char8& a, const char8& b);
char8 operator~(const char8& a);
char8 operator++(const char8& a);
char8 operator++(const char8& a, int dummy);
char8 operator--(const char8& a);
char8 operator--(const char8& a, int dummy);
char8 operator<<(const char8& a, int bits);
char8 operator>>(const char8& a, int bits);
char16 operator%(const char16& a, const char16& b);
char16 operator&(const char16& a, const char16& b);
char16 operator|(const char16& a, const char16& b);
char16 operator^(const char16& a, const char16& b);
char16 operator~(const char16& a);
char16 operator++(const char16& a);
char16 operator++(const char16& a, int dummy);
char16 operator--(const char16& a);
char16 operator--(const char16& a, int dummy);
char16 operator<<(const char16& a, int bits);
char16 operator>>(const char16& a, int bits);

uchar2 operator%(const uchar2& a, const uchar2& b);
uchar2 operator&(const uchar2& a, const uchar2& b);
uchar2 operator|(const uchar2& a, const uchar2& b);
uchar2 operator^(const uchar2& a, const uchar2& b);
uchar2 operator~(const uchar2& a);
uchar2 operator++(const uchar2& a);
uchar2 operator++(const uchar2& a, int dummy);
uchar2 operator--(const uchar2& a);
uchar2 operator--(const uchar2& a, int dummy);
uchar2 operator<<(const uchar2& a, int bits);
uchar2 operator>>(const uchar2& a, int bits);
uchar3 operator%(const uchar3& a, const uchar3& b);
uchar3 operator&(const uchar3& a, const uchar3& b);
uchar3 operator|(const uchar3& a, const uchar3& b);
uchar3 operator^(const uchar3& a, const uchar3& b);
uchar3 operator~(const uchar3& a);
uchar3 operator++(const uchar3& a);
uchar3 operator++(const uchar3& a, int dummy);
uchar3 operator--(const uchar3& a);
uchar3 operator--(const uchar3& a, int dummy);
uchar3 operator<<(const uchar3& a, int bits);
uchar3 operator>>(const uchar3& a, int bits);
uchar4 operator%(const uchar4& a, const uchar4& b);
uchar4 operator&(const uchar4& a, const uchar4& b);
uchar4 operator|(const uchar4& a, const uchar4& b);
uchar4 operator^(const uchar4& a, const uchar4& b);
uchar4 operator~(const uchar4& a);
uchar4 operator++(const uchar4& a);
uchar4 operator++(const uchar4& a, int dummy);
uchar4 operator--(const uchar4& a);
uchar4 operator--(const uchar4& a, int dummy);
uchar4 operator<<(const uchar4& a, int bits);
uchar4 operator>>(const uchar4& a, int bits);
uchar8 operator%(const uchar8& a, const uchar8& b);
uchar8 operator&(const uchar8& a, const uchar8& b);
uchar8 operator|(const uchar8& a, const uchar8& b);
uchar8 operator^(const uchar8& a, const uchar8& b);
uchar8 operator~(const uchar8& a);
uchar8 operator++(const uchar8& a);
uchar8 operator++(const uchar8& a, int dummy);
uchar8 operator--(const uchar8& a);
uchar8 operator--(const uchar8& a, int dummy);
uchar8 operator<<(const uchar8& a, int bits);
uchar8 operator>>(const uchar8& a, int bits);
uchar16 operator%(const uchar16& a, const uchar16& b);
uchar16 operator&(const uchar16& a, const uchar16& b);
uchar16 operator|(const uchar16& a, const uchar16& b);
uchar16 operator^(const uchar16& a, const uchar16& b);
uchar16 operator~(const uchar16& a);
uchar16 operator++(const uchar16& a);
uchar16 operator++(const uchar16& a, int dummy);
uchar16 operator--(const uchar16& a);
uchar16 operator--(const uchar16& a, int dummy);
uchar16 operator<<(const uchar16& a, int bits);
uchar16 operator>>(const uchar16& a, int bits);

short2 operator%(const short2& a, const short2& b);
short2 operator&(const short2& a, const short2& b);
short2 operator|(const short2& a, const short2& b);
short2 operator^(const short2& a, const short2& b);
short2 operator~(const short2& a);
short2 operator++(const short2& a);
short2 operator++(const short2& a, int dummy);
short2 operator--(const short2& a);
short2 operator--(const short2& a, int dummy);
short2 operator<<(const short2& a, int bits);
short2 operator>>(const short2& a, int bits);
short3 operator%(const short3& a, const short3& b);
short3 operator&(const short3& a, const short3& b);
short3 operator|(const short3& a, const short3& b);
short3 operator^(const short3& a, const short3& b);
short3 operator~(const short3& a);
short3 operator++(const short3& a);
short3 operator++(const short3& a, int dummy);
short3 operator--(const short3& a);
short3 operator--(const short3& a, int dummy);
short3 operator<<(const short3& a, int bits);
short3 operator>>(const short3& a, int bits);
short4 operator%(const short4& a, const short4& b);
short4 operator&(const short4& a, const short4& b);
short4 operator|(const short4& a, const short4& b);
short4 operator^(const short4& a, const short4& b);
short4 operator~(const short4& a);
short4 operator++(const short4& a);
short4 operator++(const short4& a, int dummy);
short4 operator--(const short4& a);
short4 operator--(const short4& a, int dummy);
short4 operator<<(const short4& a, int bits);
short4 operator>>(const short4& a, int bits);
short8 operator%(const short8& a, const short8& b);
short8 operator&(const short8& a, const short8& b);
short8 operator|(const short8& a, const short8& b);
short8 operator^(const short8& a, const short8& b);
short8 operator~(const short8& a);
short8 operator++(const short8& a);
short8 operator++(const short8& a, int dummy);
short8 operator--(const short8& a);
short8 operator--(const short8& a, int dummy);
short8 operator<<(const short8& a, int bits);
short8 operator>>(const short8& a, int bits);
short16 operator%(const short16& a, const short16& b);
short16 operator&(const short16& a, const short16& b);
short16 operator|(const short16& a, const short16& b);
short16 operator^(const short16& a, const short16& b);
short16 operator~(const short16& a);
short16 operator++(const short16& a);
short16 operator++(const short16& a, int dummy);
short16 operator--(const short16& a);
short16 operator--(const short16& a, int dummy);
short16 operator<<(const short16& a, int bits);
short16 operator>>(const short16& a, int bits);

ushort2 operator%(const ushort2& a, const ushort2& b);
ushort2 operator&(const ushort2& a, const ushort2& b);
ushort2 operator|(const ushort2& a, const ushort2& b);
ushort2 operator^(const ushort2& a, const ushort2& b);
ushort2 operator~(const ushort2& a);
ushort2 operator++(const ushort2& a);
ushort2 operator++(const ushort2& a, int dummy);
ushort2 operator--(const ushort2& a);
ushort2 operator--(const ushort2& a, int dummy);
ushort2 operator<<(const ushort2& a, int bits);
ushort2 operator>>(const ushort2& a, int bits);
ushort3 operator%(const ushort3& a, const ushort3& b);
ushort3 operator&(const ushort3& a, const ushort3& b);
ushort3 operator|(const ushort3& a, const ushort3& b);
ushort3 operator^(const ushort3& a, const ushort3& b);
ushort3 operator~(const ushort3& a);
ushort3 operator++(const ushort3& a);
ushort3 operator++(const ushort3& a, int dummy);
ushort3 operator--(const ushort3& a);
ushort3 operator--(const ushort3& a, int dummy);
ushort3 operator<<(const ushort3& a, int bits);
ushort3 operator>>(const ushort3& a, int bits);
ushort4 operator%(const ushort4& a, const ushort4& b);
ushort4 operator&(const ushort4& a, const ushort4& b);
ushort4 operator|(const ushort4& a, const ushort4& b);
ushort4 operator^(const ushort4& a, const ushort4& b);
ushort4 operator~(const ushort4& a);
ushort4 operator++(const ushort4& a);
ushort4 operator++(const ushort4& a, int dummy);
ushort4 operator--(const ushort4& a);
ushort4 operator--(const ushort4& a, int dummy);
ushort4 operator<<(const ushort4& a, int bits);
ushort4 operator>>(const ushort4& a, int bits);
ushort8 operator%(const ushort8& a, const ushort8& b);
ushort8 operator&(const ushort8& a, const ushort8& b);
ushort8 operator|(const ushort8& a, const ushort8& b);
ushort8 operator^(const ushort8& a, const ushort8& b);
ushort8 operator~(const ushort8& a);
ushort8 operator++(const ushort8& a);
ushort8 operator++(const ushort8& a, int dummy);
ushort8 operator--(const ushort8& a);
ushort8 operator--(const ushort8& a, int dummy);
ushort8 operator<<(const ushort8& a, int bits);
ushort8 operator>>(const ushort8& a, int bits);
ushort16 operator%(const ushort16& a, const ushort16& b);
ushort16 operator&(const ushort16& a, const ushort16& b);
ushort16 operator|(const ushort16& a, const ushort16& b);
ushort16 operator^(const ushort16& a, const ushort16& b);
ushort16 operator~(const ushort16& a);
ushort16 operator++(const ushort16& a);
ushort16 operator++(const ushort16& a, int dummy);
ushort16 operator--(const ushort16& a);
ushort16 operator--(const ushort16& a, int dummy);
ushort16 operator<<(const ushort16& a, int bits);
ushort16 operator>>(const ushort16& a, int bits);

int2 operator%(const int2& a, const int2& b);
int2 operator&(const int2& a, const int2& b);
int2 operator|(const int2& a, const int2& b);
int2 operator^(const int2& a, const int2& b);
int2 operator~(const int2& a);
int2 operator++(const int2& a);
int2 operator++(const int2& a, int dummy);
int2 operator--(const int2& a);
int2 operator--(const int2& a, int dummy);
int2 operator<<(const int2& a, int bits);
int2 operator>>(const int2& a, int bits);
int3 operator%(const int3& a, const int3& b);
int3 operator&(const int3& a, const int3& b);
int3 operator|(const int3& a, const int3& b);
int3 operator^(const int3& a, const int3& b);
int3 operator~(const int3& a);
int3 operator++(const int3& a);
int3 operator++(const int3& a, int dummy);
int3 operator--(const int3& a);
int3 operator--(const int3& a, int dummy);
int3 operator<<(const int3& a, int bits);
int3 operator>>(const int3& a, int bits);
int4 operator%(const int4& a, const int4& b);
int4 operator&(const int4& a, const int4& b);
int4 operator|(const int4& a, const int4& b);
int4 operator^(const int4& a, const int4& b);
int4 operator~(const int4& a);
int4 operator++(const int4& a);
int4 operator++(const int4& a, int dummy);
int4 operator--(const int4& a);
int4 operator--(const int4& a, int dummy);
int4 operator<<(const int4& a, int bits);
int4 operator>>(const int4& a, int bits);
int8 operator%(const int8& a, const int8& b);
int8 operator&(const int8& a, const int8& b);
int8 operator|(const int8& a, const int8& b);
int8 operator^(const int8& a, const int8& b);
int8 operator~(const int8& a);
int8 operator++(const int8& a);
int8 operator++(const int8& a, int dummy);
int8 operator--(const int8& a);
int8 operator--(const int8& a, int dummy);
int8 operator<<(const int8& a, int bits);
int8 operator>>(const int8& a, int bits);
int16 operator%(const int16& a, const int16& b);
int16 operator&(const int16& a, const int16& b);
int16 operator|(const int16& a, const int16& b);
int16 operator^(const int16& a, const int16& b);
int16 operator~(const int16& a);
int16 operator++(const int16& a);
int16 operator++(const int16& a, int dummy);
int16 operator--(const int16& a);
int16 operator--(const int16& a, int dummy);
int16 operator<<(const int16& a, int bits);
int16 operator>>(const int16& a, int bits);

uint2 operator%(const uint2& a, const uint2& b);
uint2 operator&(const uint2& a, const uint2& b);
uint2 operator|(const uint2& a, const uint2& b);
uint2 operator^(const uint2& a, const uint2& b);
uint2 operator~(const uint2& a);
uint2 operator++(const uint2& a);
uint2 operator++(const uint2& a, int dummy);
uint2 operator--(const uint2& a);
uint2 operator--(const uint2& a, int dummy);
uint2 operator<<(const uint2& a, int bits);
uint2 operator>>(const uint2& a, int bits);
uint3 operator%(const uint3& a, const uint3& b);
uint3 operator&(const uint3& a, const uint3& b);
uint3 operator|(const uint3& a, const uint3& b);
uint3 operator^(const uint3& a, const uint3& b);
uint3 operator~(const uint3& a);
uint3 operator++(const uint3& a);
uint3 operator++(const uint3& a, int dummy);
uint3 operator--(const uint3& a);
uint3 operator--(const uint3& a, int dummy);
uint3 operator<<(const uint3& a, int bits);
uint3 operator>>(const uint3& a, int bits);
uint4 operator%(const uint4& a, const uint4& b);
uint4 operator&(const uint4& a, const uint4& b);
uint4 operator|(const uint4& a, const uint4& b);
uint4 operator^(const uint4& a, const uint4& b);
uint4 operator~(const uint4& a);
uint4 operator++(const uint4& a);
uint4 operator++(const uint4& a, int dummy);
uint4 operator--(const uint4& a);
uint4 operator--(const uint4& a, int dummy);
uint4 operator<<(const uint4& a, int bits);
uint4 operator>>(const uint4& a, int bits);
uint8 operator%(const uint8& a, const uint8& b);
uint8 operator&(const uint8& a, const uint8& b);
uint8 operator|(const uint8& a, const uint8& b);
uint8 operator^(const uint8& a, const uint8& b);
uint8 operator~(const uint8& a);
uint8 operator++(const uint8& a);
uint8 operator++(const uint8& a, int dummy);
uint8 operator--(const uint8& a);
uint8 operator--(const uint8& a, int dummy);
uint8 operator<<(const uint8& a, int bits);
uint8 operator>>(const uint8& a, int bits);
uint16 operator%(const uint16& a, const uint16& b);
uint16 operator&(const uint16& a, const uint16& b);
uint16 operator|(const uint16& a, const uint16& b);
uint16 operator^(const uint16& a, const uint16& b);
uint16 operator~(const uint16& a);
uint16 operator++(const uint16& a);
uint16 operator++(const uint16& a, int dummy);
uint16 operator--(const uint16& a);
uint16 operator--(const uint16& a, int dummy);
uint16 operator<<(const uint16& a, int bits);
uint16 operator>>(const uint16& a, int bits);

long2 operator%(const long2& a, const long2& b);
long2 operator&(const long2& a, const long2& b);
long2 operator|(const long2& a, const long2& b);
long2 operator^(const long2& a, const long2& b);
long2 operator~(const long2& a);
long2 operator++(const long2& a);
long2 operator++(const long2& a, int dummy);
long2 operator--(const long2& a);
long2 operator--(const long2& a, int dummy);
long2 operator<<(const long2& a, int bits);
long2 operator>>(const long2& a, int bits);
long3 operator%(const long3& a, const long3& b);
long3 operator&(const long3& a, const long3& b);
long3 operator|(const long3& a, const long3& b);
long3 operator^(const long3& a, const long3& b);
long3 operator~(const long3& a);
long3 operator++(const long3& a);
long3 operator++(const long3& a, int dummy);
long3 operator--(const long3& a);
long3 operator--(const long3& a, int dummy);
long3 operator<<(const long3& a, int bits);
long3 operator>>(const long3& a, int bits);
long4 operator%(const long4& a, const long4& b);
long4 operator&(const long4& a, const long4& b);
long4 operator|(const long4& a, const long4& b);
long4 operator^(const long4& a, const long4& b);
long4 operator~(const long4& a);
long4 operator++(const long4& a);
long4 operator++(const long4& a, int dummy);
long4 operator--(const long4& a);
long4 operator--(const long4& a, int dummy);
long4 operator<<(const long4& a, int bits);
long4 operator>>(const long4& a, int bits);
long8 operator%(const long8& a, const long8& b);
long8 operator&(const long8& a, const long8& b);
long8 operator|(const long8& a, const long8& b);
long8 operator^(const long8& a, const long8& b);
long8 operator~(const long8& a);
long8 operator++(const long8& a);
long8 operator++(const long8& a, int dummy);
long8 operator--(const long8& a);
long8 operator--(const long8& a, int dummy);
long8 operator<<(const long8& a, int bits);
long8 operator>>(const long8& a, int bits);
long16 operator%(const long16& a, const long16& b);
long16 operator&(const long16& a, const long16& b);
long16 operator|(const long16& a, const long16& b);
long16 operator^(const long16& a, const long16& b);
long16 operator~(const long16& a);
long16 operator++(const long16& a);
long16 operator++(const long16& a, int dummy);
long16 operator--(const long16& a);
long16 operator--(const long16& a, int dummy);
long16 operator<<(const long16& a, int bits);
long16 operator>>(const long16& a, int bits);

ulong2 operator%(const ulong2& a, const ulong2& b);
ulong2 operator&(const ulong2& a, const ulong2& b);
ulong2 operator|(const ulong2& a, const ulong2& b);
ulong2 operator^(const ulong2& a, const ulong2& b);
ulong2 operator~(const ulong2& a);
ulong2 operator++(const ulong2& a);
ulong2 operator++(const ulong2& a, int dummy);
ulong2 operator--(const ulong2& a);
ulong2 operator--(const ulong2& a, int dummy);
ulong2 operator<<(const ulong2& a, int bits);
ulong2 operator>>(const ulong2& a, int bits);
ulong3 operator%(const ulong3& a, const ulong3& b);
ulong3 operator&(const ulong3& a, const ulong3& b);
ulong3 operator|(const ulong3& a, const ulong3& b);
ulong3 operator^(const ulong3& a, const ulong3& b);
ulong3 operator~(const ulong3& a);
ulong3 operator++(const ulong3& a);
ulong3 operator++(const ulong3& a, int dummy);
ulong3 operator--(const ulong3& a);
ulong3 operator--(const ulong3& a, int dummy);
ulong3 operator<<(const ulong3& a, int bits);
ulong3 operator>>(const ulong3& a, int bits);
ulong4 operator%(const ulong4& a, const ulong4& b);
ulong4 operator&(const ulong4& a, const ulong4& b);
ulong4 operator|(const ulong4& a, const ulong4& b);
ulong4 operator^(const ulong4& a, const ulong4& b);
ulong4 operator~(const ulong4& a);
ulong4 operator++(const ulong4& a);
ulong4 operator++(const ulong4& a, int dummy);
ulong4 operator--(const ulong4& a);
ulong4 operator--(const ulong4& a, int dummy);
ulong4 operator<<(const ulong4& a, int bits);
ulong4 operator>>(const ulong4& a, int bits);
ulong8 operator%(const ulong8& a, const ulong8& b);
ulong8 operator&(const ulong8& a, const ulong8& b);
ulong8 operator|(const ulong8& a, const ulong8& b);
ulong8 operator^(const ulong8& a, const ulong8& b);
ulong8 operator~(const ulong8& a);
ulong8 operator++(const ulong8& a);
ulong8 operator++(const ulong8& a, int dummy);
ulong8 operator--(const ulong8& a);
ulong8 operator--(const ulong8& a, int dummy);
ulong8 operator<<(const ulong8& a, int bits);
ulong8 operator>>(const ulong8& a, int bits);
ulong16 operator%(const ulong16& a, const ulong16& b);
ulong16 operator&(const ulong16& a, const ulong16& b);
ulong16 operator|(const ulong16& a, const ulong16& b);
ulong16 operator^(const ulong16& a, const ulong16& b);
ulong16 operator~(const ulong16& a);
ulong16 operator++(const ulong16& a);
ulong16 operator++(const ulong16& a, int dummy);
ulong16 operator--(const ulong16& a);
ulong16 operator--(const ulong16& a, int dummy);
ulong16 operator<<(const ulong16& a, int bits);
ulong16 operator>>(const ulong16& a, int bits);

char as_char(const uchar& x);
char2 as_char2(const uchar2& x);
char2 as_char2(const short& x);
char2 as_char2(const ushort& x);
char3 as_char3(const char4& x);
char3 as_char3(const uchar3& x);
char3 as_char3(const uchar4& x);
char4 as_char4(const uchar4& x);
char4 as_char4(const short2& x);
char4 as_char4(const ushort2& x);
char4 as_char4(const int& x);
char4 as_char4(const uint& x);
char4 as_char4(const float& x);
char8 as_char8(const uchar8& x);
char8 as_char8(const short4& x);
char8 as_char8(const ushort4& x);
char8 as_char8(const int2& x);
char8 as_char8(const uint2& x);
char8 as_char8(const long& x);
char8 as_char8(const ulong& x);
char8 as_char8(const float2& x);
char8 as_char8(const double& x);
char16 as_char16(const uchar16& x);
char16 as_char16(const short8& x);
char16 as_char16(const ushort8& x);
char16 as_char16(const int4& x);
char16 as_char16(const uint4& x);
char16 as_char16(const long2& x);
char16 as_char16(const ulong2& x);
char16 as_char16(const float4& x);
char16 as_char16(const double2& x);

uchar as_uchar(const char& x);
uchar2 as_uchar2(const char2& x);
uchar2 as_uchar2(const short& x);
uchar2 as_uchar2(const ushort& x);
uchar3 as_uchar3(const char3& x);
uchar3 as_uchar3(const char4& x);
uchar3 as_uchar3(const uchar4& x);
uchar4 as_uchar4(const char4& x);
uchar4 as_uchar4(const short2& x);
uchar4 as_uchar4(const ushort2& x);
uchar4 as_uchar4(const int& x);
uchar4 as_uchar4(const uint& x);
uchar4 as_uchar4(const float& x);
uchar8 as_uchar8(const char8& x);
uchar8 as_uchar8(const short4& x);
uchar8 as_uchar8(const ushort4& x);
uchar8 as_uchar8(const int2& x);
uchar8 as_uchar8(const uint2& x);
uchar8 as_uchar8(const long& x);
uchar8 as_uchar8(const ulong& x);
uchar8 as_uchar8(const float2& x);
uchar8 as_uchar8(const double& x);
uchar16 as_uchar16(const char16& x);
uchar16 as_uchar16(const short8& x);
uchar16 as_uchar16(const ushort8& x);
uchar16 as_uchar16(const int4& x);
uchar16 as_uchar16(const uint4& x);
uchar16 as_uchar16(const long2& x);
uchar16 as_uchar16(const ulong2& x);
uchar16 as_uchar16(const float4& x);
uchar16 as_uchar16(const double2& x);

short as_short(const char2& x);
short as_short(const uchar2& x);
short as_short(const ushort& x);
short2 as_short2(const char4& x);
short2 as_short2(const uchar4& x);
short2 as_short2(const ushort2& x);
short2 as_short2(const int& x);
short2 as_short2(const uint& x);
short2 as_short2(const float& x);
short3 as_short3(const short4& x);
short3 as_short3(const ushort3& x);
short3 as_short3(const ushort4& x);
short4 as_short4(const char8& x);
short4 as_short4(const uchar8& x);
short4 as_short4(const ushort4& x);
short4 as_short4(const int2& x);
short4 as_short4(const uint2& x);
short4 as_short4(const long& x);
short4 as_short4(const ulong& x);
short4 as_short4(const float2& x);
short4 as_short4(const double& x);
short8 as_short8(const char16& x);
short8 as_short8(const uchar16& x);
short8 as_short8(const ushort8& x);
short8 as_short8(const int4& x);
short8 as_short8(const uint4& x);
short8 as_short8(const long2& x);
short8 as_short8(const ulong2& x);
short8 as_short8(const float4& x);
short8 as_short8(const double2& x);
short16 as_short16(const ushort16& x);
short16 as_short16(const int8& x);
short16 as_short16(const uint8& x);
short16 as_short16(const long4& x);
short16 as_short16(const ulong4& x);
short16 as_short16(const float8& x);
short16 as_short16(const double4& x);

ushort as_ushort(const char2& x);
ushort as_ushort(const uchar2& x);
ushort as_ushort(const short& x);
ushort2 as_ushort2(const char4& x);
ushort2 as_ushort2(const uchar4& x);
ushort2 as_ushort2(const short2& x);
ushort2 as_ushort2(const int& x);
ushort2 as_ushort2(const uint& x);
ushort2 as_ushort2(const float& x);
ushort3 as_ushort3(const short3& x);
ushort3 as_ushort3(const short4& x);
ushort3 as_ushort3(const ushort4& x);
ushort4 as_ushort4(const char8& x);
ushort4 as_ushort4(const uchar8& x);
ushort4 as_ushort4(const short4& x);
ushort4 as_ushort4(const int2& x);
ushort4 as_ushort4(const uint2& x);
ushort4 as_ushort4(const long& x);
ushort4 as_ushort4(const ulong& x);
ushort4 as_ushort4(const float2& x);
ushort4 as_ushort4(const double& x);
ushort8 as_ushort8(const char16& x);
ushort8 as_ushort8(const uchar16& x);
ushort8 as_ushort8(const short8& x);
ushort8 as_ushort8(const int4& x);
ushort8 as_ushort8(const uint4& x);
ushort8 as_ushort8(const long2& x);
ushort8 as_ushort8(const ulong2& x);
ushort8 as_ushort8(const float4& x);
ushort8 as_ushort8(const double2& x);
ushort16 as_ushort16(const short16& x);
ushort16 as_ushort16(const int8& x);
ushort16 as_ushort16(const uint8& x);
ushort16 as_ushort16(const long4& x);
ushort16 as_ushort16(const ulong4& x);
ushort16 as_ushort16(const float8& x);
ushort16 as_ushort16(const double4& x);

int as_int(const char4& x);
int as_int(const uchar4& x);
int as_int(const short2& x);
int as_int(const ushort2& x);
int as_int(const uint& x);
int as_int(const float& x);
int2 as_int2(const char8& x);
int2 as_int2(const uchar8& x);
int2 as_int2(const short4& x);
int2 as_int2(const ushort4& x);
int2 as_int2(const uint2& x);
int2 as_int2(const long& x);
int2 as_int2(const ulong& x);
int2 as_int2(const float2& x);
int2 as_int2(const double& x);
int3 as_int3(const int4& x);
int3 as_int3(const uint3& x);
int3 as_int3(const uint4& x);
int3 as_int3(const float3& x);
int3 as_int3(const float4& x);
int4 as_int4(const char16& x);
int4 as_int4(const uchar16& x);
int4 as_int4(const short8& x);
int4 as_int4(const ushort8& x);
int4 as_int4(const uint4& x);
int4 as_int4(const long2& x);
int4 as_int4(const ulong2& x);
int4 as_int4(const float4& x);
int4 as_int4(const double2& x);
int8 as_int8(const short16& x);
int8 as_int8(const ushort16& x);
int8 as_int8(const uint8& x);
int8 as_int8(const long4& x);
int8 as_int8(const ulong4& x);
int8 as_int8(const float8& x);
int8 as_int8(const double4& x);
int16 as_int16(const uint16& x);
int16 as_int16(const long8& x);
int16 as_int16(const ulong8& x);
int16 as_int16(const float16& x);
int16 as_int16(const double8& x);

uint as_uint(const char4& x);
uint as_uint(const uchar4& x);
uint as_uint(const short2& x);
uint as_uint(const ushort2& x);
uint as_uint(const int& x);
uint as_uint(const float& x);
uint2 as_uint2(const char8& x);
uint2 as_uint2(const uchar8& x);
uint2 as_uint2(const short4& x);
uint2 as_uint2(const ushort4& x);
uint2 as_uint2(const int2& x);
uint2 as_uint2(const long& x);
uint2 as_uint2(const ulong& x);
uint2 as_uint2(const float2& x);
uint2 as_uint2(const double& x);
uint3 as_uint3(const int3& x);
uint3 as_uint3(const int4& x);
uint3 as_uint3(const uint4& x);
uint3 as_uint3(const float3& x);
uint3 as_uint3(const float4& x);
uint4 as_uint4(const char16& x);
uint4 as_uint4(const uchar16& x);
uint4 as_uint4(const short8& x);
uint4 as_uint4(const ushort8& x);
uint4 as_uint4(const int4& x);
uint4 as_uint4(const long2& x);
uint4 as_uint4(const ulong2& x);
uint4 as_uint4(const float4& x);
uint4 as_uint4(const double2& x);
uint8 as_uint8(const short16& x);
uint8 as_uint8(const ushort16& x);
uint8 as_uint8(const int8& x);
uint8 as_uint8(const long4& x);
uint8 as_uint8(const ulong4& x);
uint8 as_uint8(const float8& x);
uint8 as_uint8(const double4& x);
uint16 as_uint16(const int16& x);
uint16 as_uint16(const long8& x);
uint16 as_uint16(const ulong8& x);
uint16 as_uint16(const float16& x);
uint16 as_uint16(const double8& x);

long as_long(const char8& x);
long as_long(const uchar8& x);
long as_long(const short4& x);
long as_long(const ushort4& x);
long as_long(const int2& x);
long as_long(const uint2& x);
long as_long(const ulong& x);
long as_long(const float2& x);
long as_long(const double& x);
long2 as_long2(const char16& x);
long2 as_long2(const uchar16& x);
long2 as_long2(const short8& x);
long2 as_long2(const ushort8& x);
long2 as_long2(const int4& x);
long2 as_long2(const uint4& x);
long2 as_long2(const ulong2& x);
long2 as_long2(const float4& x);
long2 as_long2(const double2& x);
long3 as_long3(const long4& x);
long3 as_long3(const ulong3& x);
long3 as_long3(const ulong4& x);
long3 as_long3(const double3& x);
long3 as_long3(const double4& x);
long4 as_long4(const short16& x);
long4 as_long4(const ushort16& x);
long4 as_long4(const int8& x);
long4 as_long4(const uint8& x);
long4 as_long4(const ulong4& x);
long4 as_long4(const float8& x);
long4 as_long4(const double4& x);
long8 as_long8(const int16& x);
long8 as_long8(const uint16& x);
long8 as_long8(const ulong8& x);
long8 as_long8(const float16& x);
long8 as_long8(const double8& x);
long16 as_long16(const ulong16& x);
long16 as_long16(const double16& x);

ulong as_ulong(const char8& x);
ulong as_ulong(const uchar8& x);
ulong as_ulong(const short4& x);
ulong as_ulong(const ushort4& x);
ulong as_ulong(const int2& x);
ulong as_ulong(const uint2& x);
ulong as_ulong(const long& x);
ulong as_ulong(const float2& x);
ulong as_ulong(const double& x);
ulong2 as_ulong2(const char16& x);
ulong2 as_ulong2(const uchar16& x);
ulong2 as_ulong2(const short8& x);
ulong2 as_ulong2(const ushort8& x);
ulong2 as_ulong2(const int4& x);
ulong2 as_ulong2(const uint4& x);
ulong2 as_ulong2(const long2& x);
ulong2 as_ulong2(const float4& x);
ulong2 as_ulong2(const double2& x);
ulong3 as_ulong3(const long3& x);
ulong3 as_ulong3(const long4& x);
ulong3 as_ulong3(const ulong4& x);
ulong3 as_ulong3(const double3& x);
ulong3 as_ulong3(const double4& x);
ulong4 as_ulong4(const short16& x);
ulong4 as_ulong4(const ushort16& x);
ulong4 as_ulong4(const int8& x);
ulong4 as_ulong4(const uint8& x);
ulong4 as_ulong4(const long4& x);
ulong4 as_ulong4(const float8& x);
ulong4 as_ulong4(const double4& x);
ulong8 as_ulong8(const int16& x);
ulong8 as_ulong8(const uint16& x);
ulong8 as_ulong8(const long8& x);
ulong8 as_ulong8(const float16& x);
ulong8 as_ulong8(const double8& x);
ulong16 as_ulong16(const long16& x);
ulong16 as_ulong16(const double16& x);

float as_float(const char4& x);
float as_float(const uchar4& x);
float as_float(const short2& x);
float as_float(const ushort2& x);
float as_float(const int& x);
float as_float(const uint& x);
float2 as_float2(const char8& x);
float2 as_float2(const uchar8& x);
float2 as_float2(const short4& x);
float2 as_float2(const ushort4& x);
float2 as_float2(const int2& x);
float2 as_float2(const uint2& x);
float2 as_float2(const long& x);
float2 as_float2(const ulong& x);
float2 as_float2(const double& x);
float3 as_float3(const int3& x);
float3 as_float3(const int4& x);
float3 as_float3(const uint3& x);
float3 as_float3(const uint4& x);
float3 as_float3(const float4& x);
float4 as_float4(const char16& x);
float4 as_float4(const uchar16& x);
float4 as_float4(const short8& x);
float4 as_float4(const ushort8& x);
float4 as_float4(const int4& x);
float4 as_float4(const uint4& x);
float4 as_float4(const long2& x);
float4 as_float4(const ulong2& x);
float4 as_float4(const double2& x);
float8 as_float8(const short16& x);
float8 as_float8(const ushort16& x);
float8 as_float8(const int8& x);
float8 as_float8(const uint8& x);
float8 as_float8(const long4& x);
float8 as_float8(const ulong4& x);
float8 as_float8(const double4& x);
float16 as_float16(const int16& x);
float16 as_float16(const uint16& x);
float16 as_float16(const long8& x);
float16 as_float16(const ulong8& x);
float16 as_float16(const double8& x);

double as_double(const char8& x);
double as_double(const uchar8& x);
double as_double(const short4& x);
double as_double(const ushort4& x);
double as_double(const int2& x);
double as_double(const uint2& x);
double as_double(const long& x);
double as_double(const ulong& x);
double as_double(const float2& x);
double2 as_double2(const char16& x);
double2 as_double2(const uchar16& x);
double2 as_double2(const short8& x);
double2 as_double2(const ushort8& x);
double2 as_double2(const int4& x);
double2 as_double2(const uint4& x);
double2 as_double2(const long2& x);
double2 as_double2(const ulong2& x);
double2 as_double2(const float4& x);
double3 as_double3(const long3& x);
double3 as_double3(const long4& x);
double3 as_double3(const ulong3& x);
double3 as_double3(const ulong4& x);
double3 as_double3(const double4& x);
double4 as_double4(const short16& x);
double4 as_double4(const ushort16& x);
double4 as_double4(const int8& x);
double4 as_double4(const uint8& x);
double4 as_double4(const long4& x);
double4 as_double4(const ulong4& x);
double4 as_double4(const float8& x);
double8 as_double8(const int16& x);
double8 as_double8(const uint16& x);
double8 as_double8(const long8& x);
double8 as_double8(const ulong8& x);
double8 as_double8(const float16& x);
double16 as_double16(const long16& x);
double16 as_double16(const ulong16& x);

#endif /* OPENCLKERNEL_HPP_ */
