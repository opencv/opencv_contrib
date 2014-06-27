#pragma once

#include <opencv2/core/types_c.h>

//! computes the absolute difference of two unsigned char values
static inline size_t absdiff_uchar(uchar a, uchar b) {
	return (size_t)abs((int)a-(int)b); // should return the same as (a<b?b-a:a-b), but faster when properly optimized
}

//! computes the L1 distance between two unsigned char vectors (RGB)
static inline size_t L1dist_uchar(const uchar* a, const uchar* b) {
	return absdiff_uchar(a[0],b[0])+absdiff_uchar(a[1],b[1])+absdiff_uchar(a[2],b[2]);
}

//! computes the color distortion between two unsigned char vectors (RGB)
static inline size_t cdist_uchar(const uchar* curr, const uchar* bg) {
	size_t curr_int_sqr = curr[0]*curr[0] + curr[1]*curr[1] + curr[2]*curr[2];
	if(bg[0] || bg[1] || bg[2]) {
		size_t bg_int_sqr = bg[0]*bg[0] + bg[1]*bg[1] + bg[2]*bg[2];
		float mix_int_sqr = std::pow((float)(curr[0]*bg[0] + curr[1]*bg[1] + curr[2]*bg[2]),2);
		return (size_t)sqrt(curr_int_sqr-(mix_int_sqr/bg_int_sqr));
	}
	else
		return (size_t)sqrt((float)curr_int_sqr);
}

//! computes the L1 distance between two opencv unsigned char vectors (RGB)
static inline size_t L1dist_vec3b(const cv::Vec3b& a, const cv::Vec3b& b) {
	const uchar a_array[3] = {a[0],a[1],a[2]};
	const uchar b_array[3] = {b[0],b[1],b[2]};
	return L1dist_uchar(a_array,b_array);
}

//! computes the squared L2 distance between two unsigned char vectors (RGB)
static inline size_t L2sqrdist_uchar(const uchar* a, const uchar* b) {
	return (absdiff_uchar(a[0],b[0])^2)+(absdiff_uchar(a[1],b[1])^2)+(absdiff_uchar(a[2],b[2])^2);
}

//! computes the L2 distance between two unsigned char vectors (RGB)
static inline float L2dist_uchar(const uchar* a, const uchar* b) {
	return sqrt((float)L2sqrdist_uchar(a,b));
}

//! computes the squared L2 distance between two opencv unsigned char vectors (RGB)
static inline size_t L2sqrdist_vec3b(const cv::Vec3b& a, const cv::Vec3b& b) {
	const uchar a_array[3] = {a[0],a[1],a[2]};
	const uchar b_array[3] = {b[0],b[1],b[2]};
	return L2sqrdist_uchar(a_array,b_array);
}

//! computes the squared L2 distance between two opencv unsigned char vectors (RGB)
static inline float L2dist_vec3b(const cv::Vec3b& a, const cv::Vec3b& b) {
	return sqrt((float)L2sqrdist_vec3b(a,b));
}

//! popcount LUT for 8bit vectors
static const uchar popcount_LUT8[256] = {
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

//! computes the population count of a 16bit vector using an 8bit popcount LUT (min=0, max=48)
static inline uchar popcount_ushort_8bitsLUT(ushort x) {
	return popcount_LUT8[(uchar)x] + popcount_LUT8[(uchar)(x>>8)];
}

//! computes the population count of 3x16bit vectors using an 8bit popcount LUT (min=0, max=48)
static inline uchar popcount_ushort_8bitsLUT(const ushort* x) {
	return	popcount_LUT8[(uchar)x[0]] + popcount_LUT8[(uchar)(x[0]>>8)]
		  + popcount_LUT8[(uchar)x[1]] + popcount_LUT8[(uchar)(x[1]>>8)]
		  + popcount_LUT8[(uchar)x[2]] + popcount_LUT8[(uchar)(x[2]>>8)];
}

//! computes the hamming distance between two 16bit vectors (min=0, max=16)
static inline size_t hdist_ushort_8bitLUT(ushort a, ushort b) {
	return popcount_ushort_8bitsLUT(a^b);
}

//! computes the sum of hamming distances between two 3x16 bits vectors (min=0, max=48)
static inline size_t hdist_ushort_8bitLUT(const ushort* a, const ushort* b) {
	return popcount_ushort_8bitsLUT(a[0]^b[0])+popcount_ushort_8bitsLUT(a[1]^b[1])+popcount_ushort_8bitsLUT(a[2]^b[2]);
}

//! computes the gradient magnitude distance between two 16 bits vectors (min=0, max=16)
static inline size_t gdist_ushort_8bitLUT(ushort a, ushort b) {
	return (size_t)abs((int)popcount_ushort_8bitsLUT(a)-(int)popcount_ushort_8bitsLUT(b));
}

//! computes the sum of gradient magnitude distances between two 3x16 bits vectors (min=0, max=48)
static inline size_t gdist_ushort_8bitLUT(const ushort* a, const ushort* b) {
	return (size_t)abs((int)popcount_ushort_8bitsLUT(a)-(int)popcount_ushort_8bitsLUT(b));
}
