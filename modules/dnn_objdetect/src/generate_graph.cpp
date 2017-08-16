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
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

using conv    = convolutional_layer;
using pool    = max_pooling_layer;
using relu    = relu_layer;
using drop    = dropout_layer;
using soft    = softmax_layer;
using concat  = concat_layer;
using sigmoid = sigmoid_layer;

void make_fire(std::shared_ptr<conv> &fire_squeeze,
               std::shared_ptr<relu> &rect_fire_squeeze,
               std::shared_ptr<conv> &fire_expand_1x1,
               std::shared_ptr<relu> &rect_fire_expand_1x1,
               std::shared_ptr<conv> &fire_expand_3x3,
               std::shared_ptr<relu> &rect_fire_expand_3x3,
               std::shared_ptr<concat> &fire,
               size_t input_x, size_t squeeze_filters, size_t expand_filters,
               size_t input_filters) {

  fire_squeeze         = std::make_shared<conv>(input_x, input_x, 1, input_filters,
                                    squeeze_filters, padding::valid, true, 1, 1);
  rect_fire_squeeze    = std::make_shared<relu>();
  fire_expand_1x1      = std::make_shared<conv>(input_x, input_x, 1, squeeze_filters,
                                    expand_filters, padding::valid, true, 1, 1);
  rect_fire_expand_1x1 = std::make_shared<relu>();
  fire_expand_3x3      = std::make_shared<conv>(input_x, input_x, 3, squeeze_filters,
                                    expand_filters, padding::same, true, 1, 1);
  rect_fire_expand_3x3 = std::make_shared<relu>();
  fire                 = std::make_shared<concat>(std::initializer_list<shape3d>
                                      {{input_x, input_x, expand_filters},
                                       {input_x, input_x, expand_filters}});
}


void generate_graph() {

  // declare the nodes
  // TODO : Add negative slope in relu layer
  // TODO : In concat layer, what is the channel order

  auto input = std::make_shared<input_layer>(shape3d(3, 416, 416));
  auto conv1 = std::make_shared<conv>(416, 416, 7, 3, 96,
                                      padding::valid, true, 2, 2);
  auto rect_conv1 = std::make_shared<relu>();
  auto pool1 = std::make_shared<pool>(205, 205, 96, 3, 3, 2, 2, padding::valid);

  // fire2
  std::shared_ptr<conv> fire2_squeeze;
  std::shared_ptr<relu> rect_fire2_squeeze;
  std::shared_ptr<conv> fire2_expand_1x1;
  std::shared_ptr<relu> rect_fire2_expand_1x1;
  std::shared_ptr<conv> fire2_expand_3x3;
  std::shared_ptr<relu> rect_fire2_expand_3x3;
  std::shared_ptr<concat> fire2;
  make_fire(fire2_squeeze, rect_fire2_squeeze,
            fire2_expand_1x1, rect_fire2_expand_1x1,
            fire2_expand_3x3, rect_fire2_expand_3x3,
            fire2,
            102, 16, 64, 96);

  // fire3
  std::shared_ptr<conv> fire3_squeeze;
  std::shared_ptr<relu> rect_fire3_squeeze;
  std::shared_ptr<conv> fire3_expand_1x1;
  std::shared_ptr<relu> rect_fire3_expand_1x1;
  std::shared_ptr<conv> fire3_expand_3x3;
  std::shared_ptr<relu> rect_fire3_expand_3x3;
  std::shared_ptr<concat> fire3;
  make_fire(fire3_squeeze, rect_fire3_squeeze,
            fire3_expand_1x1, rect_fire3_expand_1x1,
            fire3_expand_3x3, rect_fire3_expand_3x3,
            fire3,
            102, 16, 64, 128);

  // fire4
  std::shared_ptr<conv> fire4_squeeze;
  std::shared_ptr<relu> rect_fire4_squeeze;
  std::shared_ptr<conv> fire4_expand_1x1;
  std::shared_ptr<relu> rect_fire4_expand_1x1;
  std::shared_ptr<conv> fire4_expand_3x3;
  std::shared_ptr<relu> rect_fire4_expand_3x3;
  std::shared_ptr<concat> fire4;
  make_fire(fire4_squeeze, rect_fire4_squeeze,
            fire4_expand_1x1, rect_fire4_expand_1x1,
            fire4_expand_3x3, rect_fire4_expand_3x3,
            fire4,
            102, 32, 128, 128);

  // 2nd pool layer
  auto pool4 = std::make_shared<pool>(102, 102, 256, 3, 3, 2, 2, padding::same);

  // fire5
  std::shared_ptr<conv> fire5_squeeze;
  std::shared_ptr<relu> rect_fire5_squeeze;
  std::shared_ptr<conv> fire5_expand_1x1;
  std::shared_ptr<relu> rect_fire5_expand_1x1;
  std::shared_ptr<conv> fire5_expand_3x3;
  std::shared_ptr<relu> rect_fire5_expand_3x3;
  std::shared_ptr<concat> fire5;
  make_fire(fire5_squeeze, rect_fire5_squeeze,
            fire5_expand_1x1, rect_fire5_expand_1x1,
            fire5_expand_3x3, rect_fire5_expand_3x3,
            fire5,
            51, 32, 128, 256);

  // fire6
  std::shared_ptr<conv> fire6_squeeze;
  std::shared_ptr<relu> rect_fire6_squeeze;
  std::shared_ptr<conv> fire6_expand_1x1;
  std::shared_ptr<relu> rect_fire6_expand_1x1;
  std::shared_ptr<conv> fire6_expand_3x3;
  std::shared_ptr<relu> rect_fire6_expand_3x3;
  std::shared_ptr<concat> fire6;
  make_fire(fire6_squeeze, rect_fire6_squeeze,
            fire6_expand_1x1, rect_fire6_expand_1x1,
            fire6_expand_3x3, rect_fire6_expand_3x3,
            fire6,
            51, 48, 192, 256);

  // fire7
  std::shared_ptr<conv> fire7_squeeze;
  std::shared_ptr<relu> rect_fire7_squeeze;
  std::shared_ptr<conv> fire7_expand_1x1;
  std::shared_ptr<relu> rect_fire7_expand_1x1;
  std::shared_ptr<conv> fire7_expand_3x3;
  std::shared_ptr<relu> rect_fire7_expand_3x3;
  std::shared_ptr<concat> fire7;
  make_fire(fire7_squeeze, rect_fire7_squeeze,
            fire7_expand_1x1, rect_fire7_expand_1x1,
            fire7_expand_3x3, rect_fire7_expand_3x3,
            fire7,
            51, 48, 192, 384);

  // fire8
  std::shared_ptr<conv> fire8_squeeze;
  std::shared_ptr<relu> rect_fire8_squeeze;
  std::shared_ptr<conv> fire8_expand_1x1;
  std::shared_ptr<relu> rect_fire8_expand_1x1;
  std::shared_ptr<conv> fire8_expand_3x3;
  std::shared_ptr<relu> rect_fire8_expand_3x3;
  std::shared_ptr<concat> fire8;
  make_fire(fire8_squeeze, rect_fire8_squeeze,
            fire8_expand_1x1, rect_fire8_expand_1x1,
            fire8_expand_3x3, rect_fire8_expand_3x3,
            fire8,
            51, 64, 256, 384);

  // 3rd pool layer
  auto pool8 = std::make_shared<pool>(51, 51, 512, 3, 3, 2, 2, padding::valid);

  // fire9
  std::shared_ptr<conv> fire9_squeeze;
  std::shared_ptr<relu> rect_fire9_squeeze;
  std::shared_ptr<conv> fire9_expand_1x1;
  std::shared_ptr<relu> rect_fire9_expand_1x1;
  std::shared_ptr<conv> fire9_expand_3x3;
  std::shared_ptr<relu> rect_fire9_expand_3x3;
  std::shared_ptr<concat> fire9;
  make_fire(fire9_squeeze, rect_fire9_squeeze,
            fire9_expand_1x1, rect_fire9_expand_1x1,
            fire9_expand_3x3, rect_fire9_expand_3x3,
            fire9,
            25, 64, 256, 512);

  auto conv10 = std::make_shared<conv>(25, 25, 1, 512, 1000,
                                      padding::same, true, 1, 1);

  // fire10
  std::shared_ptr<conv> fire10_squeeze;
  std::shared_ptr<relu> rect_fire10_squeeze;
  std::shared_ptr<conv> fire10_expand_1x1;
  std::shared_ptr<relu> rect_fire10_expand_1x1;
  std::shared_ptr<conv> fire10_expand_3x3;
  std::shared_ptr<relu> rect_fire10_expand_3x3;
  std::shared_ptr<concat> fire10;
  make_fire(fire10_squeeze, rect_fire10_squeeze,
            fire10_expand_1x1, rect_fire10_expand_1x1,
            fire10_expand_3x3, rect_fire10_expand_3x3,
            fire10,
            25, 96, 384, 1000);

  // fire11
  std::shared_ptr<conv> fire11_squeeze;
  std::shared_ptr<relu> rect_fire11_squeeze;
  std::shared_ptr<conv> fire11_expand_1x1;
  std::shared_ptr<relu> rect_fire11_expand_1x1;
  std::shared_ptr<conv> fire11_expand_3x3;
  std::shared_ptr<relu> rect_fire11_expand_3x3;
  std::shared_ptr<concat> fire11;
  make_fire(fire11_squeeze, rect_fire11_squeeze,
            fire11_expand_1x1, rect_fire11_expand_1x1,
            fire11_expand_3x3, rect_fire11_expand_3x3,
            fire11,
            25, 96, 384, 768);

  auto conv11 = std::make_shared<conv>(25, 25, 3, 768, 225,
                                      padding::same, true, 1, 1);

  // Add the permute layer and the rest of the layers here

  // Construct the graph
  input << conv1
        << rect_conv1
        << pool1;

  // fire2
  pool1 << fire2_squeeze
        << rect_fire2_squeeze;
  rect_fire2_squeeze << fire2_expand_1x1
                     << rect_fire2_expand_1x1;
  rect_fire2_squeeze << fire2_expand_3x3
                     << rect_fire2_expand_3x3;
  (rect_fire2_expand_1x1, rect_fire2_expand_3x3) << fire2;

  // fire3
  fire2 << fire3_squeeze
        << rect_fire3_squeeze;
  rect_fire3_squeeze << fire3_expand_1x1
                     << rect_fire3_expand_1x1;
  rect_fire3_squeeze << fire3_expand_3x3
                     << rect_fire3_expand_3x3;
  (rect_fire3_expand_1x1, rect_fire3_expand_3x3) << fire3;

  // fire4
  fire3 << fire4_squeeze
        << rect_fire4_squeeze;
  rect_fire4_squeeze << fire4_expand_1x1
                     << rect_fire4_expand_1x1;
  rect_fire4_squeeze << fire4_expand_3x3
                     << rect_fire4_expand_3x3;
  (rect_fire4_expand_1x1, rect_fire4_expand_3x3) << fire4;

  // pool
  fire4 << pool4;
  
  // fire5
  pool4 << fire5_squeeze
        << rect_fire5_squeeze;
  rect_fire5_squeeze << fire5_expand_1x1
                     << rect_fire5_expand_1x1;
  rect_fire5_squeeze << fire5_expand_3x3
                     << rect_fire5_expand_3x3;
  (rect_fire5_expand_1x1, rect_fire5_expand_3x3) << fire5;

  // fire6
  fire5 << fire6_squeeze
        << rect_fire6_squeeze;
  rect_fire6_squeeze << fire6_expand_1x1
                     << rect_fire6_expand_1x1;
  rect_fire6_squeeze << fire6_expand_3x3
                     << rect_fire6_expand_3x3;
  (rect_fire6_expand_1x1, rect_fire6_expand_3x3) << fire6;

  // fire7
  fire6 << fire7_squeeze
        << rect_fire7_squeeze;
  rect_fire7_squeeze << fire7_expand_1x1
                     << rect_fire7_expand_1x1;
  rect_fire7_squeeze << fire7_expand_3x3
                     << rect_fire7_expand_3x3;
  (rect_fire7_expand_1x1, rect_fire7_expand_3x3) << fire7;

  // fire8
  fire7 << fire8_squeeze
        << rect_fire8_squeeze;
  rect_fire8_squeeze << fire8_expand_1x1
                     << rect_fire8_expand_1x1;
  rect_fire8_squeeze << fire8_expand_3x3
                     << rect_fire8_expand_3x3;
  (rect_fire8_expand_1x1, rect_fire8_expand_3x3) << fire8;

  // pool
  fire8 << pool8;
  
  // fire9
  pool8 << fire9_squeeze
        << rect_fire9_squeeze;
  rect_fire9_squeeze << fire9_expand_1x1
                     << rect_fire9_expand_1x1;
  rect_fire9_squeeze << fire9_expand_3x3
                     << rect_fire9_expand_3x3;
  (rect_fire9_expand_1x1, rect_fire9_expand_3x3) << fire9;

  // conv
  fire9 << conv10;

  // fire10
  conv10 << fire10_squeeze
         << rect_fire10_squeeze;
  rect_fire10_squeeze << fire10_expand_1x1
                      << rect_fire10_expand_1x1;
  rect_fire10_squeeze << fire10_expand_3x3
                      << rect_fire10_expand_3x3;
  (rect_fire10_expand_1x1, rect_fire10_expand_3x3) << fire10;

  // fire11
  fire10 << fire11_squeeze
         << rect_fire11_squeeze;
  rect_fire11_squeeze << fire11_expand_1x1
                      << rect_fire11_expand_1x1;
  rect_fire11_squeeze << fire11_expand_3x3
                      << rect_fire11_expand_3x3;
  (rect_fire11_expand_1x1, rect_fire11_expand_3x3) << fire11;

  // conv
  fire11 << conv11;

  network<graph> net;
  construct_graph(net, {input}, {conv11});

}

int main() {
  std::cout << "Generating the graph !\n";
  generate_graph();
}
