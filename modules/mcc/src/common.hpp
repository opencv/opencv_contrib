#ifndef _MCC_COMMON_HPP
#define _MCC_COMMON_HPP

#include "precomp.hpp"

using namespace std;
namespace cv{
namespace mcc {

	cv::Mat poly2mask(const std::vector<cv::Point2f> &poly, cv::Size size);

	template <typename T>
	void circshift(std::vector<T> &A, int shiff)
	{
		if (A.empty() || shiff < 1) return;
		int n = A.size();

		if (shiff >= n) return;

		std::vector<T> Tmp(n);
		for (int i = shiff; i < n + shiff; i++) Tmp[(i%n)] = A[i - shiff];

		A = Tmp;
	}

	float perimeter(const std::vector<cv::Point2f> &ps);

	cv::Point2f mace_center(const std::vector<cv::Point2f> &ps);

	template<typename T>
	void unique(const std::vector<T> &A, std::vector<T> &U)
	{

		int n = A.size();
		std::vector<T> Tm = A;

		std::sort(Tm.begin(), Tm.end());

		U.clear();
		U.push_back(Tm[0]);
		for (int i = 1; i < n; i++)
			if (Tm[i] != Tm[i - 1]) U.push_back(Tm[i]);

	}

	template<typename T>
	void polyanticlockwise(std::vector<T> &points)
	{
		// Sort the points in anti-clockwise order
		// Trace a line between the first and second point.
		// If the third point is at the right side, then the points are anti-clockwise
		cv::Point v1 = points[1] - points[0];
		cv::Point v2 = points[2] - points[0];

		double o = (v1.x * v2.y) - (v1.y * v2.x);

		if (o < 0.0)	//if the third point is in the left side, then sort in anti-clockwise order
			std::swap(points[1], points[3]);

	}
	template<typename T>
	void polyclockwise(std::vector<T> &points)
	{
		// Sort the points in clockwise order
		// Trace a line between the first and second point.
		// If the third point is at the right side, then the points are clockwise
		cv::Point v1 = points[1] - points[0];
		cv::Point v2 = points[2] - points[0];

		double o = (v1.x * v2.y) - (v1.y * v2.x);

		if (o > 0.0)	//if the third point is in the left side, then sort in clockwise order
			std::swap(points[1], points[3]);

	}
	// Does lexical cast of the input argument to string
	template <typename T>
	std::string ToString(const T& value)
	{
		std::ostringstream stream;
		stream << value;
		return stream.str();
	}

	template<typename T>
	void change(T &a, T &b) { T c = a;  a = b;  b = c; }

	template<typename T>
	void sort(std::vector<T> &A, std::vector<int> &idx, bool ord = true)
	{
		size_t N = A.size();
		if (N == 0) return;

		idx.clear(); idx.resize(N);
		for (size_t i = 0; i < N; i++) idx[i] = i;

		for (size_t i = 0; i < N - 1; i++)
		{

			size_t k = i; T valor = A[i];
			for (size_t j = i + 1; j < N; j++)
			{
				if ((A[j] < valor && ord) || (A[j] > valor && !ord))
				{
					valor = A[j];
					k = j;
				}
			}

			if (k == i) continue;

			change(A[i], A[k]);
			change(idx[i], idx[k]);

		}
	}


}
}

#endif //_MCC_COMMON_HPP
