// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "face_alignmentimpl.hpp"
#include <fstream>
#include <ctime>

using namespace std;
namespace cv{
namespace face{
bool FacemarkKazemiImpl :: findNearestLandmarks( vector< vector<int> >& nearest){
    if(meanshape.empty()||loaded_pixel_coordinates.empty()){
        String error_message = "Model not loaded properly.Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    nearest.resize(loaded_pixel_coordinates.size());
    for(unsigned long i=0 ; i< loaded_pixel_coordinates.size(); i++){
        for(unsigned long j = 0;j<loaded_pixel_coordinates[i].size();j++){
            nearest[i].push_back(getNearestLandmark(loaded_pixel_coordinates[i][j]));
        }
    }
    return true;
}
void FacemarkKazemiImpl :: readSplit(ifstream& is, splitr &vec)
{
    is.read((char*)&vec, sizeof(splitr));
}
void FacemarkKazemiImpl :: readLeaf(ifstream& is, vector<Point2f> &leaf)
{
    uint64_t size;
    is.read((char*)&size, sizeof(size));
    leaf.resize((size_t)size);
    is.read((char*)&leaf[0], leaf.size() * sizeof(Point2f));
}
void FacemarkKazemiImpl :: readPixels(ifstream& is,uint64_t index)
{
    is.read((char*)&loaded_pixel_coordinates[(unsigned long)index][0], loaded_pixel_coordinates[(unsigned long)index].size() * sizeof(Point2f));
}
void FacemarkKazemiImpl :: loadModel(String filename){
    if(filename.empty()){
        String error_message = "No filename found.Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    ifstream f(filename.c_str(),ios::binary);
    if(!f.is_open()){
        String error_message = "No file with given name found.Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    uint64_t len;
    f.read((char*)&len, sizeof(len));
    char* temp = new char[(size_t)len+1];
    f.read(temp, len);
    temp[len] = '\0';
    string s(temp);
    delete [] temp;
    if(s.compare("cascade_depth")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    uint64_t cascade_size;
    f.read((char*)&cascade_size,sizeof(cascade_size));
    loaded_forests.resize((unsigned long)cascade_size);
    f.read((char*)&len, sizeof(len));
    temp = new char[(unsigned long)len+1];
    f.read(temp, len);
    temp[len] = '\0';
    s = string(temp);
    delete [] temp;
    if(s.compare("pixel_coordinates")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    loaded_pixel_coordinates.resize((unsigned long)cascade_size);
    uint64_t num_pixels;
    f.read((char*)&num_pixels,sizeof(num_pixels));
    for(unsigned long i=0 ; i < cascade_size ; i++){
        loaded_pixel_coordinates[i].resize((unsigned long)num_pixels);
        readPixels(f,i);
    }
    f.read((char*)&len, sizeof(len));
    temp = new char[(unsigned long)len+1];
    f.read(temp, len);
    temp[len] = '\0';
    s = string(temp);
    delete [] temp;
    if(s.compare("mean_shape")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    uint64_t mean_shape_size;
    f.read((char*)&mean_shape_size,sizeof(mean_shape_size));
    meanshape.resize((unsigned long)mean_shape_size);
    f.read((char*)&meanshape[0], meanshape.size() * sizeof(Point2f));
    if(!setMeanExtreme())
        exit(0);
    f.read((char*)&len, sizeof(len));
    temp = new char[(unsigned long)len+1];
    f.read(temp, len);
    temp[len] = '\0';
    s = string(temp);
    delete [] temp;
    if(s.compare("num_trees")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    uint64_t num_trees;
    f.read((char*)&num_trees,sizeof(num_trees));
    for(unsigned long i=0;i<cascade_size;i++){
        for(unsigned long j=0;j<num_trees;j++){
            regtree tree;
            f.read((char*)&len, sizeof(len));
            char* temp2 = new char[(unsigned long)len+1];
            f.read(temp2, len);
            temp2[len] = '\0';
            s =string(temp2);
            delete [] temp2;
            if(s.compare("num_nodes")!=0){
                String error_message = "Data not saved properly.Aborting.....";
                CV_Error(Error::StsBadArg, error_message);
                return ;
            }
            uint64_t num_nodes;
            f.read((char*)&num_nodes,sizeof(num_nodes));
            tree.nodes.resize((unsigned long)num_nodes+1);
            for(unsigned long k=0; k < num_nodes ; k++){
                f.read((char*)&len, sizeof(len));
                char* temp3 = new char[(unsigned long)len+1];
                f.read(temp3, len);
                temp3[len] = '\0';
                s =string(temp3);
                delete [] temp3;
                tree_node node;
                if(s.compare("split")==0){
                    splitr split;
                    readSplit(f,split);
                    node.split = split;
                    node.leaf.clear();
                }
                else if(s.compare("leaf")==0){
                    vector<Point2f> leaf;
                    readLeaf(f,leaf);
                    node.leaf = leaf;
                }
                else{
                    String error_message = "Data not saved properly.Aborting.....";
                    CV_Error(Error::StsBadArg, error_message);
                    return ;
                }
                tree.nodes[k]=node;
            }
            loaded_forests[i].push_back(tree);
        }
    }
    f.close();
    isModelLoaded = true;
}
bool FacemarkKazemiImpl::fit(InputArray img, InputArray roi, OutputArrayOfArrays landmarks){
    if(!isModelLoaded){
        String error_message = "No model loaded. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    Mat image  = img.getMat();
    std::vector<Rect> & faces = *(std::vector<Rect>*)roi.getObj();
    std::vector<std::vector<Point2f> > & shapes = *(std::vector<std::vector<Point2f> >*) landmarks.getObj();
    shapes.resize(faces.size());

    if(image.empty()){
        String error_message = "No image found.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(faces.empty()){
        String error_message = "No faces found.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(meanshape.empty()||loaded_forests.empty()||loaded_pixel_coordinates.empty()){
        String error_message = "Model not loaded properly.Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(loaded_forests.size()==0){
        String error_message = "Model not loaded properly.Aboerting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(loaded_pixel_coordinates.size()==0){
        String error_message = "Model not loaded properly.Aboerting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    vector< vector<int> > nearest_landmarks;
    findNearestLandmarks(nearest_landmarks);
    tree_node curr_node;
    vector<Point2f> pixel_relative;
    vector<int> pixel_intensity;
    Mat warp_mat;
    for(size_t e=0;e<faces.size();e++){
        shapes[e]=meanshape;
        convertToActual(faces[e],warp_mat);
        for(size_t i=0;i<loaded_forests.size();i++){
            pixel_intensity.clear();
            pixel_relative = loaded_pixel_coordinates[i];
            getRelativePixels(shapes[e],pixel_relative,nearest_landmarks[i]);
            getPixelIntensities(image,pixel_relative,pixel_intensity,faces[e]);
            for(size_t j=0;j<loaded_forests[i].size();j++){
                regtree tree = loaded_forests[i][j];
                curr_node = tree.nodes[0];
                unsigned long curr_node_index = 0;
                while(curr_node.leaf.size()==0)
                {
                    if ((float)pixel_intensity[(unsigned long)curr_node.split.index1] - (float)pixel_intensity[(unsigned long)curr_node.split.index2] > curr_node.split.thresh)
                    {
                        curr_node_index=left(curr_node_index);
                    }                    else
                        curr_node_index=right(curr_node_index);
                    curr_node = tree.nodes[curr_node_index];
                }
                for(size_t p=0;p<curr_node.leaf.size();p++){
                    shapes[e][p]=shapes[e][p] + curr_node.leaf[p];
                }
            }
        }
        for(unsigned long j=0;j<shapes[e].size();j++){
                Mat C = (Mat_<double>(3,1) << shapes[e][j].x, shapes[e][j].y, 1);
                Mat D = warp_mat*C;
                shapes[e][j].x=float(D.at<double>(0,0));
                shapes[e][j].y=float(D.at<double>(1,0));
        }
    }
    return true;
}
}//cv
}//face
