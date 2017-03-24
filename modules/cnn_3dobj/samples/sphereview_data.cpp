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
/**
 * @file demo_sphereview_data.cpp
 * @brief Generating training data for CNN with triplet loss.
 * @author Yida Wang
 */
#include <opencv2/cnn_3dobj.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace cv;
using namespace std;
using namespace cv::cnn_3dobj;

/**
 * @function listDir
 * @brief Making all files names under a directory into a list
 */
static void listDir(const char *path, std::vector<String>& files, bool r)
{
    DIR *pDir;
    struct dirent *ent;
    char childpath[512];
    pDir = opendir(path);
    memset(childpath, 0, sizeof(childpath));
    while ((ent = readdir(pDir)) != NULL)
    {
        if (ent->d_type & DT_DIR)
        {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0 || strcmp(ent->d_name, ".DS_Store") == 0)
            {
                continue;
            }
            if (r)
            {
                sprintf(childpath, "%s/%s", path, ent->d_name);
                listDir(childpath,files,false);
            }
        }
        else
        {
            if (strcmp(ent->d_name, ".DS_Store") != 0)
                files.push_back(ent->d_name);
        }
    }
    sort(files.begin(),files.end());
};

int main(int argc, char *argv[])
{
    const String keys = "{help | | demo :$ ./sphereview_test -ite_depth=2 -plymodel=../data/3Dmodel/ape.ply -imagedir=../data/images_all/ -labeldir=../data/label_all.txt -num_class=6 -label_class=0, then press 'q' to run the demo for images generation when you see the gray background and a coordinate.}"
    "{ite_depth | 3 | Iteration of sphere generation.}"
    "{plymodel | ../data/3Dmodel/ape.ply | Path of the '.ply' file for image rendering. }"
    "{imagedir | ../data/images_all/ | Path of the generated images for one particular .ply model. }"
    "{labeldir | ../data/label_all.txt | Path of the generated images for one particular .ply model. }"
    "{bakgrdir | | Path of the backgroud images sets. }"
    "{cam_head_x | 0 | Head of the camera. }"
    "{cam_head_y | 0 | Head of the camera. }"
    "{cam_head_z | -1 | Head of the camera. }"
    "{semisphere | 1 | Camera only has positions on half of the whole sphere. }"
    "{z_range | 0.6 | Maximum camera position on z axis. }"
    "{center_gen | 0 | Find center from all points. }"
    "{image_size | 128 | Size of captured images. }"
    "{label_class |  | Class label of current .ply model. }"
    "{label_item |  | Item label of current .ply model. }"
    "{rgb_use | 0 | Use RGB image or grayscale. }"
    "{num_class | 6 | Total number of classes of models. }"
    "{binary_out | 0 | Produce binaryfiles for images and label. }"
    "{view_region | 0 | Take a special view of front or back angle}";
    /* Get parameters from comand line. */
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Generating training data for CNN with triplet loss");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int ite_depth = parser.get<int>("ite_depth");
    String plymodel = parser.get<String>("plymodel");
    String imagedir = parser.get<String>("imagedir");
    String labeldir = parser.get<String>("labeldir");
    String bakgrdir = parser.get<String>("bakgrdir");
    int label_class = parser.get<int>("label_class");
    int label_item = parser.get<int>("label_item");
    float cam_head_x = parser.get<float>("cam_head_x");
    float cam_head_y = parser.get<float>("cam_head_y");
    float cam_head_z = parser.get<float>("cam_head_z");
    int semisphere = parser.get<int>("semisphere");
    float z_range = parser.get<float>("z_range");
    int center_gen = parser.get<int>("center_gen");
    int image_size = parser.get<int>("image_size");
    int rgb_use = parser.get<int>("rgb_use");
    int num_class = parser.get<int>("num_class");
    int binary_out = parser.get<int>("binary_out");
    int view_region = parser.get<int>("view_region");
    double obj_dist, bg_dist, y_range;
    if (view_region == 1 || view_region == 2)
    {
        /* Set for TV */
        if (label_class == 12)
            obj_dist = 340;
        else
            obj_dist = 250;
        ite_depth = ite_depth + 1;
        bg_dist = 700;
        y_range = 0.85;
    }
    else if (view_region == 0)
    {
        obj_dist = 370;
        bg_dist = 400;
    }
    if (label_class == 5 || label_class == 10 || label_class == 11 || label_class == 12)
        ite_depth = ite_depth + 1;
    cv::cnn_3dobj::icoSphere ViewSphere(10,ite_depth);
    std::vector<cv::Point3d> campos;
    std::vector<cv::Point3d> campos_temp = ViewSphere.CameraPos;
    /* Regular objects on the ground using a semisphere view system */
    if (semisphere == 1)
    {
        if (view_region == 1)
        {
            for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
            {
                if (campos_temp.at(pose).z >= 0 && campos_temp.at(pose).z < z_range && campos_temp.at(pose).y < -y_range)
                    campos.push_back(campos_temp.at(pose));
            }
        }
        else if (view_region == 2)
        {
            for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
            {
                if (campos_temp.at(pose).z >= 0 && campos_temp.at(pose).z < z_range && campos_temp.at(pose).y > y_range)
                campos.push_back(campos_temp.at(pose));
            }
        }
        else
        {
            /* Set for sofa */
            if (label_class == 10)
            {
                for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
                {
                    if (campos_temp.at(pose).z >= 0 && campos_temp.at(pose).z < z_range && campos_temp.at(pose).y < -0.4)
                    campos.push_back(campos_temp.at(pose));
                }
            }
            else
            {
                for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
                {
                    if (campos_temp.at(pose).z >= 0 && campos_temp.at(pose).z < z_range)
                        campos.push_back(campos_temp.at(pose));
                }
            }
        }
    }
    /* Special object such as plane using a full space of view sphere */
    else
    {
        if (view_region == 1)
        {
            for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
            {
                if (campos_temp.at(pose).z < 0.2 && campos_temp.at(pose).z > -0.2 && campos_temp.at(pose).y < -y_range)
                    campos.push_back(campos_temp.at(pose));
            }
        }
        else if (view_region == 2)
        {
            for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
            {
                if (campos_temp.at(pose).z < 0.2 && campos_temp.at(pose).z > -0.2 && campos_temp.at(pose).y > y_range)
                campos.push_back(campos_temp.at(pose));
            }
        }
        else
        {
            for (int pose = 0; pose < static_cast<int>(campos_temp.size()); pose++)
            {
                if (campos_temp.at(pose).z < 0.2 && campos_temp.at(pose).z > -0.6)
                    campos.push_back(campos_temp.at(pose));
            }
        }
    }
    std::fstream imglabel;
    imglabel.open(labeldir.c_str(), fstream::app|fstream::out);
    bool camera_pov = true;
    /* Create a window using viz. */
    viz::Viz3d myWindow("Coordinate Frame");
    /* Set window size. */
    myWindow.setWindowSize(Size(image_size,image_size));
    /* Set background color. */
    myWindow.setBackgroundColor(viz::Color::gray());
    myWindow.spinOnce();
    /* Create a Mesh widget, loading .ply models. */
    viz::Mesh objmesh = viz::Mesh::load(plymodel);
    /* Get the center of the generated mesh widget, cause some .ply files, this could be ignored if you are using PASCAL database*/
    Point3d cam_focal_point;
    if (center_gen)
        cam_focal_point = ViewSphere.getCenter(objmesh.cloud);
    else
        cam_focal_point = Point3d(0,0,0);
    const char* headerPath = "../data/header_for_";
    const char* binaryPath = "../data/binary_";
    if (binary_out)
    {
        ViewSphere.createHeader(static_cast<int>(campos.size()), image_size, image_size, headerPath);
    }
    float radius = ViewSphere.getRadius(objmesh.cloud, cam_focal_point);
    objmesh.cloud = objmesh.cloud/radius*100;
    cam_focal_point = cam_focal_point/radius*100;
    Point3d cam_y_dir;
    cam_y_dir.x = cam_head_x;
    cam_y_dir.y = cam_head_y;
    cam_y_dir.z = cam_head_z;
    char temp[1024];
    std::vector<String> name_bkg;
    if (bakgrdir.size() != 0)
    {
        /* List the file names under a given path */
        listDir(bakgrdir.c_str(), name_bkg, false);
        for (unsigned int i = 0; i < name_bkg.size(); i++)
        {
            name_bkg.at(i) = bakgrdir + name_bkg.at(i);
        }
    }
    /* Images will be saved as .png files. */
    size_t cnt_img;
    srand((int)time(0));
    do
    {
        cnt_img = 0;
        for(int pose = 0; pose < static_cast<int>(campos.size()); pose++){
            /* Add light. */
            // double alpha1 = rand()%(314/2)/100;
            // double alpha2 = rand()%(314*2)/100;
            // printf("%f %f %f/n", ceil(10000*sqrt(1 - sin(alpha1)*sin(alpha1))*sin(alpha2)), 10000*sqrt(1 - sin(alpha1)*sin(alpha1))*cos(alpha2), sin(alpha1)*10000);
            // myWindow.addLight(Vec3d(10000*sqrt(1 - sin(alpha1)*sin(alpha1))*sin(alpha2),10000*sqrt(1 - sin(alpha1)*sin(alpha1))*cos(alpha2),sin(alpha1)*10000), Vec3d(0,0,0), viz::Color::white(), viz::Color::white(), viz::Color::black(), viz::Color::white());
            int label_x, label_y, label_z;
            label_x = static_cast<int>(campos.at(pose).x*100);
            label_y = static_cast<int>(campos.at(pose).y*100);
            label_z = static_cast<int>(campos.at(pose).z*100);
            sprintf (temp,"%02i_%02i_%04i_%04i_%04i_%02i", label_class, label_item, label_x, label_y, label_z, static_cast<int>(obj_dist/100));
            String filename = temp;
            filename += ".png";
            imglabel << filename << ' ' << label_class << endl;
            filename = imagedir + filename;
            /* Get the pose of the camera using makeCameraPoses. */
            if (view_region != 0)
            {
                cam_focal_point.x = cam_focal_point.y - label_x/5;
            }
            Affine3f cam_pose = viz::makeCameraPose(campos.at(pose)*obj_dist+cam_focal_point, cam_focal_point, cam_y_dir*obj_dist+cam_focal_point);
            /* Get the transformation matrix from camera coordinate system to global. */
            Affine3f transform = viz::makeTransformToGlobal(Vec3f(1.0f,0.0f,0.0f), Vec3f(0.0f,1.0f,0.0f), Vec3f(0.0f,0.0f,1.0f), campos.at(pose));
            viz::WMesh mesh_widget(objmesh);
            /* Pose of the widget in camera frame. */
            Affine3f cloud_pose = Affine3f().translate(Vec3f(1.0f,1.0f,1.0f));
            /* Pose of the widget in global frame. */
            Affine3f cloud_pose_global = transform * cloud_pose;
            /* Visualize camera frame. */
            if (!camera_pov)
            {
                viz::WCameraPosition cpw(1); // Coordinate axes
                viz::WCameraPosition cpw_frustum(Vec2f(0.5, 0.5)); // Camera frustum
                myWindow.showWidget("CPW", cpw, cam_pose);
                myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
            }

            /* Visualize widget. */
            if (bakgrdir.size() != 0)
            {
                cv::Mat img_bg = cv::imread(name_bkg.at(rand()%name_bkg.size()));
                /* Back ground images has a distance of 2 times of radius of camera view distance */
                cv::viz::WImage3D background_widget(img_bg, Size2d(image_size*4.2, image_size*4.2), Vec3d(-campos.at(pose)*bg_dist+cam_focal_point), Vec3d(campos.at(pose)*bg_dist-cam_focal_point), Vec3d(0,0,-1)*bg_dist+Vec3d(0,2*cam_focal_point.y,0));
                myWindow.showWidget("bgwidget", background_widget, cloud_pose_global);
            }
            // mesh_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
            myWindow.showWidget("targetwidget", mesh_widget, cloud_pose_global);

            /* Set the viewer pose to that of camera. */
            if (camera_pov)
                myWindow.setViewerPose(cam_pose);
            /* Save screen shot as images. */
            myWindow.saveScreenshot(filename);
            if (binary_out)
            {
            /* Write images into binary files for further using in CNN training. */
                ViewSphere.writeBinaryfile(filename, binaryPath, headerPath,static_cast<int>(campos.size())*num_class, label_class, static_cast<int>(campos.at(pose).x*100), static_cast<int>(campos.at(pose).y*100), static_cast<int>(campos.at(pose).z*100), rgb_use);
            }
            cnt_img++;
        }
    } while (cnt_img != campos.size());
    imglabel.close();
    return 1;
};
