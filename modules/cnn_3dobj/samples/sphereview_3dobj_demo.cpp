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
int main(int argc, char *argv[]){
	float radius = atof(argv[1]);
	int ite_depth = argv[2][0] - '0';
	cv::cnn_3dobj::IcoSphere ViewSphere(10,ite_depth);
	std::vector<cv::Point3d> campos = ViewSphere.CameraPos;
	std::fstream imglabel;
	std::string plymodel = argv[3];
	imglabel.open("../data/label_ape.txt");
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
	Point3d cam_focal_point(0.0f,0.0f,0.0f), cam_y_dir(-0.0f,-0.0f,-1.0f);
	for(int pose = 0; pose < (int)campos.size(); pose++){
		imglabel << campos.at(pose).x << ' ' << campos.at(pose).y << ' ' << campos.at(pose).z << endl;
		/// We can get the pose of the cam using makeCameraPoses
		Affine3f cam_pose = viz::makeCameraPose(campos.at(pose)*radius, cam_focal_point, cam_y_dir);
		//Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
		/// We can get the transformation matrix from camera coordinate system to global using
		/// - makeTransformToGlobal. We need the axes of the camera
		Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f,-1.0f,0.0f), Vec3f(-1.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-1.0f), campos.at(pose));
		/// Create a cloud widget.
		viz::Mesh objmesh = viz::Mesh::load(plymodel);
		viz::WMesh mesh_widget(objmesh);
		/// Pose of the widget in camera frame
		Affine3f cloud_pose = Affine3f().translate(Vec3f(3.0f,3.0f,3.0f));
		/// Pose of the widget in global frame
		Affine3f cloud_pose_global = transform * cloud_pose;
		/// Visualize camera frame
		if (!camera_pov)
		{
			viz::WCameraPosition cpw(1); // Coordinate axes
			viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
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
		filename = "../data/images_ape/" + filename;
		filename += ".png";
		myWindow.saveScreenshot(filename);
	}
	return 1;
};
