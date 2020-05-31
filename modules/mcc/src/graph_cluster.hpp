#ifndef _MCC_GRAPH_CLUSTERS_H
#define _MCC_GRAPH_CLUSTERS_H

#include "core.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace cv{
namespace mcc{

class CB0cluster
{
public:

	CB0cluster();
	~CB0cluster();

	inline void setVertex(const std::vector<cv::Point> &V) { X = V; }
	inline void setB0(const std::vector<float> &b0) { B0 = b0;  }
	inline void setWeight(const std::vector<float> &Weight) { W = Weight; }

	void group();

	void getGroup(std::vector<int> &g) { g = G; }

private:

	//entrada
	std::vector<cv::Point> X;
	std::vector<float> B0;
	std::vector<float> W;

	//salida
	std::vector<int> G;


private:

	template<typename T>
	void find(const std::vector<T> &A, std::vector<int> &indx)
	{
		indx.clear();
		for (size_t i = 0; i < A.size(); i++)
		if (A[i])indx.push_back(i);

	}

};

}
}

#endif //_MCC_GRAPH_CLUSTERS_H
