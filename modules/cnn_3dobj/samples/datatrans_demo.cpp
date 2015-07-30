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
using namespace cv;
using namespace std;
using namespace cv::cnn_3dobj;
int main(int argc, char* argv[])
{
	const String keys = "{help | | this demo will convert a set of images in a particular path into leveldb database for feature extraction using Caffe.}"
		     "{src_dir | ../data/images_all | Source direction of the images ready for being converted to leveldb dataset.}"
		     "{src_dst | ../data/dbfile | Aim direction of the converted to leveldb dataset. }"
		     "{attach_dir | ../data/dbfile | Path for saving additional files which describe the transmission results. }"
		     "{channel | 1 | Channel of the images. }"
		     "{width | 64 | Width of images}"
		     "{height | 64 | Height of images}";
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Demo for Sphere View data generation");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	string src_dir = parser.get<string>("src_dir");
	string src_dst = parser.get<string>("src_dst");
	string attach_dir = parser.get<string>("attach_dir");
	int channel = parser.get<int>("channel");
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
	cv::cnn_3dobj::DataTrans Trans;
	Trans.convert(src_dir,src_dst,attach_dir,channel,width,height);
	std::cout << std::endl << "All featrues of images in: " << std::endl << src_dir << std::endl << "have been converted to levelDB data in: " << std::endl << src_dst << std::endl << "for extracting feature of gallery images in classification efficiently, this convertion is not needed in feature extraction of test image" << std::endl;
}
