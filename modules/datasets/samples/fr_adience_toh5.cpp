/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/datasets/fr_adience.hpp"
#include "opencv2/datasets/util.hpp"
#include "opencv2/datasets/util_hdf5.hpp"

#include <opencv2/core.hpp>

#include <cstdio>

#include <string>
#include <vector>
#include <set>

#ifdef HAVE_HDF5
#include <hdf5.h>

using namespace std;
using namespace cv;
using namespace cv::datasets;

struct objToWrite
{
    objToWrite() {}
    objToWrite(FR_adienceObj *curr) :
        face_id(curr->face_id),
        gender(curr->gender),
        x(curr->x),
        y(curr->y),
        dx(curr->dx),
        dy(curr->dy),
        tilt_ang(curr->tilt_ang),
        fiducial_yaw_angle(curr->fiducial_yaw_angle),
        fiducial_score(curr->fiducial_score) {}

    static hid_t createH5Type()
    {
        hid_t tid = H5Tcreate(H5T_COMPOUND, sizeof(objToWrite));
        H5Tinsert(tid, "ref", HOFFSET(objToWrite, ref), H5T_STD_REF_OBJ);
        H5Tinsert(tid, "face_id", HOFFSET(objToWrite, face_id), H5T_NATIVE_INT);
        H5Tinsert(tid, "gender", HOFFSET(objToWrite, gender), H5T_NATIVE_INT);
        H5Tinsert(tid, "x", HOFFSET(objToWrite, x), H5T_NATIVE_INT);
        H5Tinsert(tid, "y", HOFFSET(objToWrite, y), H5T_NATIVE_INT);
        H5Tinsert(tid, "dx", HOFFSET(objToWrite, dx), H5T_NATIVE_INT);
        H5Tinsert(tid, "dy", HOFFSET(objToWrite, dy), H5T_NATIVE_INT);
        H5Tinsert(tid, "tilt_ang", HOFFSET(objToWrite, tilt_ang), H5T_NATIVE_INT);
        H5Tinsert(tid, "fiducial_yaw_angle", HOFFSET(objToWrite, fiducial_yaw_angle), H5T_NATIVE_INT);
        H5Tinsert(tid, "fiducial_score", HOFFSET(objToWrite, fiducial_score), H5T_NATIVE_INT);

        return tid;
    }

    hobj_ref_t ref;
    int face_id;
    //std::string age;
    int gender;
    int x;
    int y;
    int dx;
    int dy;
    int tilt_ang;
    int fiducial_yaw_angle;
    int fiducial_score;
};

void save(const std::string &path, Ptr<FR_adience> &dataset, const std::string &name); // suppress warning
void save(const std::string &path, Ptr<FR_adience> &dataset, const std::string &name)
{
    unsigned int numObj = 0;

    hid_t file_id = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    hid_t grp_data = H5Gcreate(file_id, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t grp_split = H5Gcreate(file_id, "splits", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(grp_split);
    hid_t grp_split_train = H5Gcreate(file_id, "splits/train", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t grp_split_test = H5Gcreate(file_id, "splits/test", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t grp_split_validation = H5Gcreate(file_id, "splits/validation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    set<string> groups;
    set<string> imageNames;
    vector<objToWrite> objs;
    for (int s=0; s<3; ++s) // train test validation
    {
        unsigned int numSplits = 0;

        for (int i=0; i<dataset->getNumSplits(); ++i)
        {
            vector<int> objRefs;

            vector< Ptr<Object> > *currSplit;
            hid_t currSplitGroup;
            if (0 == s)
            {
                currSplit = &dataset->getTrain(i);
                currSplitGroup = grp_split_train;
            } else
            if (1 == s)
            {
                currSplit = &dataset->getTest(i);
                currSplitGroup = grp_split_test;
            } else
            {
                currSplit = &dataset->getValidation(i);
                currSplitGroup = grp_split_validation;
            }
            for (vector< Ptr<Object> >::iterator it=currSplit->begin(); it!=currSplit->end(); ++it)
            {
                FR_adienceObj *curr = static_cast<FR_adienceObj *>((*it).get());
                string imageName = curr->user_id + "/" + curr->original_image;
                if (imageNames.find(imageName) == imageNames.end())
                {
                    imageNames.insert(imageName);

                    // write data
                    hid_t grp_id;
                    if (groups.find(curr->user_id) == groups.end())
                    {
                        groups.insert(curr->user_id);
                        grp_id = H5Gcreate(grp_data, curr->user_id.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    } else
                    {
                        grp_id = H5Gopen(grp_data, curr->user_id.c_str(), H5P_DEFAULT);
                    }

                    writeFileToH5(path + "faces/" + imageName, curr->original_image, grp_id);

                    H5Gclose(grp_id);
                }

                // prepare to write object
                objToWrite obj(curr);
                H5Rcreate(&obj.ref, file_id, ("data/"+curr->user_id+"/"+curr->original_image).c_str(), H5R_OBJECT, -1);

                // prepare to write split
                objRefs.push_back(numObj);
                objs.push_back(obj);
                numObj++;
            }

            // write split
            if (objRefs.size() > 0)
            {
                string numSplitStr;
                numberToString(numSplits, numSplitStr);
                numSplits++;
                write1DToH5(currSplitGroup, H5T_STD_U32LE, numSplitStr, &objRefs[0], objRefs.size());
            }
        }
    }

    // write objects
    hid_t tid = objToWrite::createH5Type();
    hid_t grp_obj = H5Gcreate(file_id, "objects", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write1DToH5(grp_obj, tid, "objects", &objs[0], objs.size());
    H5Gclose(grp_obj);
    H5Tclose(tid);

    H5Gclose(grp_split_validation);
    H5Gclose(grp_split_test);
    H5Gclose(grp_split_train);
    H5Gclose(grp_data);
    H5Fclose(file_id);
}

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset folder and splits }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<FR_adience> dataset = FR_adience::create();
    dataset->load(path);

    string h5FileName("./adience.h5");
    printf("\nsave to hdf5: %s\n", h5FileName.c_str());
    save(path, dataset, h5FileName);

    return 0;
}
#else
int main(int argc, char *argv[])
{
    if (1==argc) // suppress warning
    {
        printf("%s\n", argv[0]);
    }
    return 0;
}
#endif
