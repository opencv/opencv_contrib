/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <opencv2/cnn_3dobj.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;
using namespace cv::cnn_3dobj;
Point3d getCenter(string plymodel)
{
	char* path_model=(char*)plymodel.data();
	int numPoint = 5841;
	ifstream ifs(path_model);
	string str;
	for(size_t i = 0; i < 15; ++i)
		getline(ifs, str);
	float temp_x, temp_y, temp_z;
	Point3f data;
	float dummy1, dummy2, dummy3, dummy4, dummy5, dummy6;
	for(int i = 0; i < numPoint; ++i)
	{
		ifs >> temp_x >> temp_y >> temp_z >> dummy1 >> dummy2 >> dummy3 >> dummy4 >> dummy5 >> dummy6;
		data.x += temp_x;
		data.y += temp_y;
		data.z += temp_z;
	}
	data.x = data.x/numPoint;
	data.y = data.y/numPoint;
	data.z = data.z/numPoint;
	return data;
};
int main(int argc, char *argv[]){
	const String keys = "{help | | demo :$ ./sphereview_test -radius=250 -ite_depth=1 -plymodel=../ape.ply -imagedir=../data/images_ape/ -labeldir=../data/label_ape.txt, then press 'q' to run the demo for images generation when you see the gray background and a coordinate.}"
			     "{radius | 250 | Distanse from camera to object, used for adjust view for the reason that differet scale of .ply model.}"
			     "{ite_depth | 1 | Iteration of sphere generation, we add points on the middle of lines of sphere and adjust the radius suit for the original radius.}"
			     "{plymodel | ../ape.ply | path of the '.ply' file for image rendering. }"
			     "{imagedir | ../data/images_ape/ | path of the generated images for one particular .ply model. }"
			     "{labeldir | ../data/label_ape.txt | path of the generated images for one particular .ply model. }";
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Demo for Sphere View data generation");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	float radius = parser.get<float>("radius");
	int ite_depth = parser.get<int>("ite_depth");
	string plymodel = parser.get<string>("plymodel");
	string imagedir = parser.get<string>("imagedir");
	string labeldir = parser.get<string>("labeldir");
	cv::cnn_3dobj::IcoSphere ViewSphere(10,ite_depth);
	std::vector<cv::Point3d> campos = ViewSphere.CameraPos;
	std::fstream imglabel;
	char* p=(char*)labeldir.data();
	imglabel.open(p);
	//IcoSphere ViewSphere(16,0);
	//std::vector<cv::Point3d>* campos = ViewSphere.CameraPos;
	bool camera_pov = (true);
	/// Create a window
	viz::Viz3d myWindow("Coordinate Frame");
	/// Add coordinate axes
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	myWindow.setBackgroundColor(viz::Color::gray());
	myWindow.spin();
	/// Set background color
	/// Let's assume camera has the following properties
	Point3d cam_focal_point = getCenter(plymodel);
	Point3d cam_y_dir(0.0f,0.0f,1.0f);
	for(int pose = 0; pose < (int)campos.size(); pose++){
		imglabel << campos.at(pose).x << ' ' << campos.at(pose).y << ' ' << campos.at(pose).z << endl;
		/// We can get the pose of the cam using makeCameraPoses
		Affine3f cam_pose = viz::makeCameraPose(campos.at(pose)*radius, cam_focal_point, cam_y_dir);
		//Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
		/// We can get the transformation matrix from camera coordinate system to global using
		/// - makeTransformToGlobal. We need the axes of the camera
		Affine3f transform = viz::makeTransformToGlobal(Vec3f(1.0f,0.0f,0.0f), Vec3f(0.0f,1.0f,0.0f), Vec3f(0.0f,0.0f,1.0f), campos.at(pose));
		/// Create a cloud widget.
		viz::Mesh objmesh = viz::Mesh::load(plymodel);
		viz::WMesh mesh_widget(objmesh);
		/// Pose of the widget in camera frame
		Affine3f cloud_pose = Affine3f().translate(Vec3f(1.0f,1.0f,1.0f));
		/// Pose of the widget in global frame
		Affine3f cloud_pose_global = transform * cloud_pose;
		/// Visualize camera frame
		if (!camera_pov)
		{
			viz::WCameraPosition cpw(1); // Coordinate axes
			viz::WCameraPosition cpw_frustum(Vec2f(0.5, 0.5)); // Camera frustum
			myWindow.showWidget("CPW", cpw, cam_pose);
			myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
		}

		/// Visualize widget
		mesh_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
		myWindow.showWidget("ape", mesh_widget, cloud_pose_global);

	    /*viz::WLine axis(cam_focal_point, campos->at(pose)*23);
	    axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
	    myWindow.showWidget("Line Widget", axis);*/

		/// Set the viewer pose to that of camera
		if (camera_pov)
			myWindow.setViewerPose(cam_pose);
		char* temp = new char;
		sprintf (temp,"%d",pose);
		string filename = temp;
		filename = imagedir + filename;
		filename += ".png";
		myWindow.saveScreenshot(filename);
	}
	return 1;
};
