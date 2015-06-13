/*
 * sphereview3d.hpp
 *
 *  Created on: Jun 13, 2015
 *      Author: wangyida
 */

#ifndef SPHEREVIEW3D_HPP_
#define SPHEREVIEW3D_HPP_

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;
namespace cv{ namespace sphereview3d{

class IcoSphere {


	private:
		float X;
		float Z;

	public:


		std::vector<float>* vertexNormalsList = new std::vector<float>;
		std::vector<float>* vertexList = new std::vector<float>;
		std::vector<cv::Point3d>* CameraPos = new std::vector<cv::Point3d>;
		float radius;
		IcoSphere(float radius_in, int depth_in);
		void norm(float v[]);
		void add(float v[]);
		void subdivide(float v1[], float v2[], float v3[], int depth);

};

}}



#endif /* SPHEREVIEW3D_HPP_ */
