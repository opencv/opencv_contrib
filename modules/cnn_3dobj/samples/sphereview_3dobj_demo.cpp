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
#define HAVE_CAFFE
#include <opencv2/cnn_3dobj.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <iostream>
#include <stdlib.h>
using namespace cv;
using namespace std;
using namespace cv::cnn_3dobj;
int main(int argc, char *argv[])
{
    const String keys = "{help | | demo :$ ./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/ape.ply -imagedir=../data/images_ape/ -labeldir=../data/label_ape.txt -num_class=4 -label_class=0, then press 'q' to run the demo for images generation when you see the gray background and a coordinate.}"
"{ite_depth | 2 | Iteration of sphere generation.}"
"{plymodel | ../3Dmodel/ape.ply | path of the '.ply' file for image rendering. }"
"{imagedir | ../data/images_all/ | path of the generated images for one particular .ply model. }"
"{labeldir | ../data/label_all.txt | path of the generated images for one particular .ply model. }"
"{num_class | 4 | total number of classes of models}"
"{label_class | 0 | class label of current .ply model}"
"{rgb_use | 0 | use RGB image or grayscale}";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Demo for Sphere View data generation");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int ite_depth = parser.get<int>("ite_depth");
    string plymodel = parser.get<string>("plymodel");
    string imagedir = parser.get<string>("imagedir");
    string labeldir = parser.get<string>("labeldir");
    int num_class = parser.get<int>("num_class");
    int label_class = parser.get<int>("label_class");
    int rgb_use = parser.get<int>("rgb_use");
    cv::cnn_3dobj::icoSphere ViewSphere(10,ite_depth);
    std::vector<cv::Point3d> campos = ViewSphere.CameraPos;
    std::fstream imglabel;
    char* p=(char*)labeldir.data();
    imglabel.open(p, fstream::app|fstream::out);
    bool camera_pov = (true);
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");
    myWindow.setWindowSize(Size(64,64));
    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    myWindow.setBackgroundColor(viz::Color::gray());
    myWindow.spin();
    /// Set background color
    /// Let's assume camera has the following properties
    /// Create a cloud widget.
    viz::Mesh objmesh = viz::Mesh::load(plymodel);
    Point3d cam_focal_point = ViewSphere.getCenter(objmesh.cloud);
    float radius = ViewSphere.getRadius(objmesh.cloud, cam_focal_point);
    Point3d cam_y_dir(0.0f,0.0f,1.0f);
    const char* headerPath = "./header_for_";
    const char* binaryPath = "./binary_";
    ViewSphere.createHeader((int)campos.size(), 64, 64, headerPath);
    for(int pose = 0; pose < (int)campos.size(); pose++){
        char* temp = new char;
        sprintf (temp,"%d",label_class);
        string filename = temp;
        filename += "_";
        sprintf (temp,"%d",pose);
        filename += temp;
        filename += ".png";
        imglabel << filename << ' ' << (int)(campos.at(pose).x*100) << ' ' << (int)(campos.at(pose).y*100) << ' ' << (int)(campos.at(pose).z*100) << endl;
        filename = imagedir + filename;
        /// We can get the pose of the cam using makeCameraPoses
        Affine3f cam_pose = viz::makeCameraPose(campos.at(pose)*radius+cam_focal_point, cam_focal_point, cam_y_dir*radius+cam_focal_point);
        /// We can get the transformation matrix from camera coordinate system to global using
        /// - makeTransformToGlobal. We need the axes of the camera
        Affine3f transform = viz::makeTransformToGlobal(Vec3f(1.0f,0.0f,0.0f), Vec3f(0.0f,1.0f,0.0f), Vec3f(0.0f,0.0f,1.0f), campos.at(pose));
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

        /// Set the viewer pose to that of camera
        if (camera_pov)
            myWindow.setViewerPose(cam_pose);
        myWindow.saveScreenshot(filename);
        ViewSphere.writeBinaryfile(filename, binaryPath, headerPath,(int)campos.size()*num_class, label_class, (int)(campos.at(pose).x*100), (int)(campos.at(pose).y*100), (int)(campos.at(pose).z*100), rgb_use);
    }
    imglabel.close();
    return 1;
};
