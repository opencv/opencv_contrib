#ifndef _MCC_GRAPH_CLUSTERS_HPP
#define _MCC_GRAPH_CLUSTERS_HPP

#include "precomp.hpp"

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
	inline void setB0(const std::vector<double> &b0) { B0 = b0;  }
	inline void setWeight(const std::vector<double> &Weight) { W = Weight; }

	void group();

	void getGroup(std::vector<int> &g) { g = G; }

private:

	//entrada
	std::vector<cv::Point> X;
	std::vector<double> B0;
	std::vector<double> W;

	//salida
	std::vector<int> G;


private:

	template<typename T>
	void find(const std::vector<T> &A, std::vector<int> &indx)
	{
		indx.clear();
		for (int i = 0; i < (int)A.size(); i++)
		if (A[i])indx.push_back(i);

	}

};

}
}

#endif //_MCC_GRAPH_CLUSTERS_HPP
