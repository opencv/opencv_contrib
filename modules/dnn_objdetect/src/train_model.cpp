////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <fstream>
#include <sstream>

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/layer.hpp"

#include <opencv2/core.hpp>

using namespace caffe;
using namespace cv;

void train(std::string &solver, const char* snapshot, int gpu_indx) {
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(solver, &solver_param);

    // set mode of training
    if (gpu_indx != -1) {
        solver_param.set_device_id(gpu_indx);
        Caffe::SetDevice(gpu_indx);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(1);
        std::cout << "Using GPU " << gpu_indx << " to train the model\n";
    } else {
        Caffe::set_mode(Caffe::CPU);
        std::cout << "Using CPU to train the model\n";
    }

    boost::shared_ptr<caffe::Solver<float> > net_solver(
        caffe::SolverRegistry<float>::CreateSolver(solver_param));

    if (strlen(snapshot)) {
        net_solver->Restore(snapshot);
    }

    // Begin the training
    for (size_t iter = 0; iter < solver_param.max_iter(); ++iter) {
        net_solver->Step(1);
    }
}

int main(int argc, const char** argv) {
    // Process the input arguemnts
    CommandLineParser parser(argc, argv,
        "{ help h | | show this message and exit }"
        "{ solver S | | (required) solver for the model }"
        "{ snapshot s | | snapshot of the model to resume training }"
        "{ gpu g | -1 | (optional) ID of the gpu to use for training. If not specified, cpu will be used }"
        );
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    string solver(parser.get<string>("solver"));
    string snapshot(parser.get<string>("snapshot"));
    int gpu_indx(parser.get<int>("gpu"));

    if (solver.empty()) {
        parser.printMessage();
        return -1;
    } else {
        if (snapshot.empty()) {
            train(solver, "", gpu_indx);
        } else {
            train(solver, snapshot.c_str(), gpu_indx);
        }
    }
    return 0;
}
