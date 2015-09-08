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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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
 //M*/

#include "precomp.hpp"
#include <ctime>
#include <opencv2/highgui.hpp>
#include <fstream>

namespace cv
{
	namespace saliency
	{

		/**
		 * EdgeBoxes Objectness
		 */

		ObjectnessEdgeBoxes::ObjectnessEdgeBoxes()
		{

			className = "ObjectnessEdgeBoxes";
		}

		ObjectnessEdgeBoxes::~ObjectnessEdgeBoxes()
		{

		}


		bool ObjectnessEdgeBoxes::computeSaliency(InputArray image, OutputArray saliencyMap){

			Mat inputImg = image.getMat();
			
			//compute edges
			Mat edgeImage = Mat(inputImg.rows, inputImg.cols, CV_64F);

			Mat orientationImage = Mat(edgeImage.rows, edgeImage.cols, CV_64F);
			getOrientationImage(edgeImage, orientationImage);



			//do everything else
			std::vector<Vec4i> box_list;
			std::vector<double> score_list;
			getBoxScores(edgeImage,orientationImage,box_list, score_list);
			saliencyMap.setTo(_box_list);
			return true;
		}


		bool ObjectnessEdgeBoxes::computeSaliency(InputArray image, OutputArray saliencyMap, const int mode){

			Mat inputImg = image.getMat();
			Mat edgeImage;

			switch(mode)
			{

			case 1:
					edgeImage = inputImg;
					edgeImage.convertTo(edgeImage, CV_64F, 1 / 255.0f);
						break;

			case 0:
						//todo
						break;

					default: 
						//todo
						break;
			}


			Mat orientationImage = Mat(edgeImage.rows, edgeImage.cols, CV_64F);

			getOrientationImage(edgeImage, orientationImage);


			//do everything else
			std::vector<Vec4i> box_list;
			std::vector<double> score_list;
			getBoxScores(edgeImage, orientationImage, box_list, score_list);
			saliencyMap.setTo(_box_list);
			return true;
		}



		std::vector<float> ObjectnessEdgeBoxes::getobjectnessValues()
		{

			return _score_list;
		}


		void ObjectnessEdgeBoxes::setBBResDir(std::string resultsDir)
		{
			_resultsDir = resultsDir;
		}

		bool ObjectnessEdgeBoxes::computeSaliencyImpl(InputArray image, OutputArray objectnessBoundingBox)
		{
			return true;
		}



		void ObjectnessEdgeBoxes::computeEdgeWeights(std::vector<int> &edge_idx_list,
			Mat &affinity_matrix, 
			Vec4i &box, 
			std::vector<Vec2i> &group_position, 
			std::vector<double> &edge_weight_list){

			int N = group_position.size();

			for (int i = 0; i < N; i++){
				
				Vec2i position = group_position.at(i);
				if (position[0] > box[0] & position[0] < box[2] & position[1] > box[1] & position[1] & box[3]){
					edge_weight_list.push_back(1.0f);
				}
				else{
					edge_weight_list.push_back(0.0f);
				}

			}

			for (int i = 0; i < edge_idx_list.size(); i++){
				edge_weight_list.at(edge_idx_list.at(i) - 1) = 0;
			}


			//flip them:
			for (int i = 0; i < N; i++){
				edge_weight_list.at(i) = 1 - edge_weight_list.at(i);
			}

			if (box[0] == 333 & box[1] == 64 & box[2] == 378 & box[3] == 104){
				int d = 5;
			}

			std::vector<int> todo_list;
			int todo_counter;
			for (int i = 0; i < edge_idx_list.size(); i++){
				todo_counter = 0;
				todo_list.clear();
				todo_list.push_back(edge_idx_list.at(i)-1); //source equals straddling node

				for (int tt = 0; tt < todo_list.size(); tt++){
					double source_affinity = 1 - edge_weight_list.at(todo_list.at(tt));
					for (int nn = 0; nn < N; nn++){ //for all possible nodes
						if (nn != tt){ 
							//find affinity between source and current node:
							double source_to_node_affinity = affinity_matrix.at<double>(Point(nn,todo_list.at(tt)));

							//if node is in the box:
							Vec2i position = group_position.at(nn);
							if (position[0] > box[0] & position[0] < box[2] & position[1] > box[1] & position[1] & box[3]){

								if (source_to_node_affinity > 0.015){
									//if source*source2node affinity > node affinity, replace node affinity
									double node_affinity = 1 - edge_weight_list.at(nn);
									if (source_affinity * source_to_node_affinity > node_affinity){
										edge_weight_list.at(nn) = 1 - (source_affinity * source_to_node_affinity);
										todo_list.push_back(nn);
									}  
								}

							}
							else{ // not in box
								edge_weight_list.at(nn) = 1;
							}
						}
					}
					todo_counter++;
				}

			}

			//flip them:
			for (int i = 0; i < N; i++){
				edge_weight_list.at(i) = 1 - edge_weight_list.at(i);
			}
		}



		double ObjectnessEdgeBoxes::scoreBox(Vec4i &box,
			std::vector<std::vector<int>> &row_intersection_list,
			Mat &row_intersection_img,
			std::vector<std::vector<int>> &column_intersection_list,
			Mat &column_intersection_img,
			Mat &affinity_matrix,
			std::vector<Vec2i> &group_position,
			std::vector<double> &group_sum_magnitude){

			int r0 = box[0];
			int r1 = box[2];
			int c0 = box[1];
			int c1 = box[3];

			if (r0 < 0 || r0 > row_intersection_img.rows - 1){return 0.0;}

			if (r1 < 0 || r1 > row_intersection_img.rows - 1){ return 0.0; }

			if (c0 < 0 || c0 > row_intersection_img.cols - 1){ return 0.0; }

			if (c1 < 0 || c1 > row_intersection_img.cols - 1){ return 0.0; }

			std::vector<int> edge_idx_list, list;
			int start_idx, end_idx;

			//top:
			list = row_intersection_list.at(r0);
			start_idx = row_intersection_img.at<int>(Point(c0,r0));
			end_idx = row_intersection_img.at<int>(Point(c1, r0));
			for (int kk = start_idx; kk < end_idx; kk++){
				if (list.at(kk) != 0){
					edge_idx_list.push_back(list.at(kk));
				}
			}


			//bottom:
			list = row_intersection_list.at(r1);
			start_idx = row_intersection_img.at<int>(Point(c0, r1));
			end_idx = row_intersection_img.at<int>(Point(c1, r1));
			for (int kk = start_idx; kk < end_idx; kk++){
				if (list.at(kk) != 0){
					edge_idx_list.push_back(list.at(kk));
				}
			}

			//right:
			list = column_intersection_list.at(c1);
			start_idx = column_intersection_img.at<int>(Point(c1, r0));
			end_idx = column_intersection_img.at<int>(Point(c1, r1));
			for (int kk = start_idx; kk < end_idx; kk++){
				if (list.at(kk) != 0){
					edge_idx_list.push_back(list.at(kk));
				}
			}

			//left:
			list = column_intersection_list.at(c0);
			start_idx = column_intersection_img.at<int>(Point(c0, r0));
			end_idx = column_intersection_img.at<int>(Point(c0, r1));
			for (int kk = start_idx; kk < end_idx; kk++){
				if (list.at(kk) != 0){
					edge_idx_list.push_back(list.at(kk));
				}
			}


			std::vector<double> edge_weight_list;
			computeEdgeWeights(edge_idx_list, affinity_matrix, box, group_position, edge_weight_list);

			double kappa = 1.0f;
			double score = 0;
			for (int i = 0; i < edge_weight_list.size(); i++){

				score += (edge_weight_list.at(i)*group_sum_magnitude.at(i)) / (pow(2 * ((c1-c0)+(c1-r0)), kappa));
			}




			return score;

		}


		double ObjectnessEdgeBoxes::scoreBoxParams(Vec4i &box){
			return scoreBox(box,
				_params.row_intersection_list,
				_params.row_intersection_img,
				_params.column_intersection_list,
				_params.column_intersection_img,
				_params.affinity_matrix,
				_params.group_position,
				_params.group_sum_magnitude);
		}

		void ObjectnessEdgeBoxes::computeIntersectionDataStructure(Mat &group_idx_img,
			std::vector<std::vector<int>> &row_intersection_list,
			Mat &row_intersection_img,
			std::vector<std::vector<int>> &column_intersection_list,
			Mat &column_intersection_img){

			row_intersection_img = Mat(Size(group_idx_img.cols, group_idx_img.rows), CV_32SC1, Scalar(0));
			column_intersection_img = Mat(Size(group_idx_img.cols, group_idx_img.rows), CV_32SC1, Scalar(0));

			for (int rr = 0; rr < group_idx_img.rows; rr++){
				std::vector<int> intersection_list;
				int last_idx = group_idx_img.at<int>(Point(0, rr));
				intersection_list.push_back(last_idx);
				row_intersection_img.at<int>(Point(0, rr)) = intersection_list.size();

				for (int cc = 1; cc < group_idx_img.cols; cc++){
					int current_idx = group_idx_img.at<int>(Point(cc,rr));
					if (intersection_list.at(intersection_list.size() - 1) != current_idx){
						intersection_list.push_back(current_idx);
						row_intersection_img.at<int>(Point(cc,rr)) = intersection_list.size();
					}
					else{
						row_intersection_img.at<int>(Point(cc, rr)) = intersection_list.size();
					}
				}
				row_intersection_list.push_back(intersection_list);
			}

			for (int cc = 0; cc < group_idx_img.cols; cc++){
				std::vector<int> intersection_list;
				int last_idx = group_idx_img.at<int>(Point(cc, 0));
				intersection_list.push_back(last_idx);
				row_intersection_img.at<int>(Point(cc, 0)) = intersection_list.size();

				for (int rr = 1; rr < group_idx_img.rows; rr++){
					int current_idx = group_idx_img.at<int>(Point(cc, rr));
					if (intersection_list.at(intersection_list.size() - 1) != current_idx){
						intersection_list.push_back(current_idx);
						column_intersection_img.at<int>(Point(cc, rr)) = intersection_list.size();
					}
					else{
						column_intersection_img.at<int>(Point(cc, rr)) = intersection_list.size();
					}
				}
				column_intersection_list.push_back(intersection_list);
			}
		}

		void ObjectnessEdgeBoxes::computeAffinity(std::vector<double> &group_mean_orientation, Mat &group_idx_img, Mat &affinity_matrix){
			
			int N = group_mean_orientation.size();
			affinity_matrix = Mat(Size(N, N), CV_64FC1, Scalar(-1.0f));
			double gamma = 2;
			int rad = 2;

			for (int rr = 0; rr < group_idx_img.rows; rr++){
				for (int cc = 0; cc < group_idx_img.cols; cc++){

					int idx = group_idx_img.at<int>(Point(cc,rr));
					if (idx != 0){

						for (int rk = rr - rad; rk < rr + rad + 1; rk++){
							for (int ck = cc - rad; ck < cc + rad + 1; ck++){
								int col = min(max(ck, 1), group_idx_img.cols-1);
								int row = min(max(rk, 1), group_idx_img.rows-1);

								if (col != 0 && row != 0){
									int test_idx = group_idx_img.at<int>(Point(col, row));
									if (test_idx != 0 & (test_idx != idx)){
										if (affinity_matrix.at<double>(Point(idx-1, test_idx-1)) == -1){

											double idx_orientation = group_mean_orientation.at(idx-1);
											double test_orientation = group_mean_orientation.at(test_idx-1);
											double affinity = cos(idx_orientation - (idx_orientation - test_orientation))*
												cos(test_orientation - (idx_orientation - test_orientation));
											affinity = abs(affinity);
											affinity = pow(affinity, gamma);
											affinity_matrix.at<double>(Point(idx-1, test_idx-1)) = (double)affinity;
											affinity_matrix.at<double>(Point(test_idx-1, idx-1)) = (double)affinity;

										}
									}
								}

							}
						}

					}
				}
			}
			

			for (int rr = 0; rr < affinity_matrix.rows; rr++){
				for (int cc = 0; cc < affinity_matrix.cols; cc++){
					if (affinity_matrix.at<double>(rr, cc) == -1){
						affinity_matrix.at<double>(rr, cc) = 0;
					}
				}
			}
			
			
			//return group_idx_img;
		}

		bool ObjectnessEdgeBoxes::getNextComponent(Mat &edgeImage, int &current_row, int &current_col, Mat &isProcessed, int &next_row, int &next_col){

			for (int rr = current_row - 1; rr < current_row + 2; rr++){
				for (int cc = current_col - 1; cc < current_col + 2; cc++){
					if (rr >= 0 && rr < edgeImage.rows && cc >= 0 && cc < edgeImage.cols){ //if in bounds:
						if (rr != current_row || cc != current_col){ //if i'm not evaluating myself:
							if (isProcessed.at<bool>(Point(cc, rr)) == false && edgeImage.at<double>(Point(cc, rr)) > 0.1){
								next_row = rr;
								next_col = cc;
								return true;
							}
						}
					}
				}
			}
			return false;
		}


		bool ObjectnessEdgeBoxes::clusterEdges(Mat &edgeImage,
			Mat &orientationImage,
			Mat &group_idx_img,
			std::vector<double> &group_mean_orientation,
			std::vector<double> &group_mean_magnitude,
			std::vector<double> &group_sum_magnitude,
			std::vector<Vec2i> &group_position){

			cv::namedWindow("debug window", WINDOW_AUTOSIZE);// Create a window for display.


			std::vector<std::vector<double>> all_orientations;
			std::vector<std::vector<double>> all_magnitudes;
			std::vector<std::vector<Vec2i>> all_positions;


			float cumulative_error = 0.0f;
			int current_col, current_row, next_col, next_row = 0;
			double current_orientation, current_magnitude, new_orientation, new_magnitude;
			std::vector<double> this_group_orientation_list;
			std::vector<double> this_group_magnitude_list;
			std::vector<Vec2i> this_group_position_list;


			Mat isProcessed = Mat(Size(edgeImage.cols, edgeImage.rows), CV_8U, Scalar(0));
			group_idx_img = Mat(Size(edgeImage.cols, edgeImage.rows), CV_32S, Scalar(0));
			int segment_count = 0;
			for (int rr = 0; rr < edgeImage.rows; rr++){
				for (int cc = 0; cc < edgeImage.cols; cc++){
					if (isProcessed.at<bool>(Point(cc, rr))==false){
						double value = edgeImage.at<double>(Point(cc, rr));
						if (edgeImage.at<double>(Point(cc, rr)) > 0.1){

							std::vector<float> tree_orientation_list;
							std::vector<Vec2i> trunk_list;
							int trunk_col = cc;
							int trunk_row = rr;
							segment_count++;
							cumulative_error = 0.0f;

							this_group_magnitude_list.clear();
							this_group_orientation_list.clear();
							this_group_position_list.clear();

							current_orientation = edgeImage.at<double>(Point(cc, rr));
							current_magnitude = orientationImage.at<double>(Point(cc, rr));

							tree_orientation_list.push_back(current_orientation);
							double latest_orientation = current_orientation;
							trunk_list.push_back(Vec2i(rr, cc));


							this_group_magnitude_list.push_back(current_magnitude);
							this_group_orientation_list.push_back(current_orientation);
							this_group_position_list.push_back(Vec2i(rr, cc));
							group_idx_img.at<int>(rr, cc) = segment_count;
							isProcessed.at<bool>(rr, cc) = true;

							bool keep_growing = true;
							while (keep_growing){
								std::vector<Vec2i> new_trunk_list;
								for (int tt = 0; tt < trunk_list.size(); tt++){ //go through each trunk point

									Vec2i trunk_point = trunk_list.at(tt);

									//get new branches
									for (int rd = -1; rd < 2; rd++){
										for (int cd = -1; cd < 2; cd++){
											if (rd != 0 || cd != 0){
												Vec2i branch_point = Vec2i(trunk_point[0] + rd, trunk_point[1] + cd);
												if (isProcessed.at<bool>(branch_point[0], branch_point[1]) == false){
													double branch_orientation = orientationImage.at<double>(branch_point[0], branch_point[1]);
													double branch_magnitude = edgeImage.at<double>(branch_point[0], branch_point[1]);
													
													if (branch_magnitude > 0.1){
														//if we add it to the group and it's under threshold 
														double potential_error = cumulative_error + fabs(latest_orientation - branch_orientation);
														if (potential_error < 3.14 / 2.0f){
															new_trunk_list.push_back(branch_point);
															latest_orientation = branch_orientation;
															isProcessed.at<bool>(branch_point[0], branch_point[1]) = true;
															group_idx_img.at<int>(branch_point[0], branch_point[1]) = segment_count;
															cumulative_error = potential_error;
															this_group_magnitude_list.push_back(branch_magnitude);
															this_group_orientation_list.push_back(branch_orientation);
															this_group_position_list.push_back(Vec2i(branch_point[0], branch_point[1]));
														} // potential error
													}
													else{
														isProcessed.at<bool>(branch_point[0], branch_point[1]) == true;
														group_idx_img.at<int>(branch_point[0], branch_point[1]) = 0;
													}
												} // is branch processed

											}
										}
									} // looping over potential branches

									trunk_list = new_trunk_list;
									new_trunk_list.clear();
									if (trunk_list.size() == 0){
										keep_growing = 0;
									}
								} // for each trunk point
							} // while keep growing

							all_magnitudes.push_back(this_group_magnitude_list);
							all_orientations.push_back(this_group_orientation_list);
							all_positions.push_back(this_group_position_list);

							if (all_positions.size() == 32){
								int d = 5;
							}

							printf("Adding segment %d, length %d \n", segment_count, this_group_magnitude_list.size());
							if (segment_count == 105){
								int d = 5;
							}
						}
						else{
							isProcessed.at<bool>(Point(cc, rr)) == true;
							group_idx_img.at<int>(Point(cc, rr)) = 0;
						}



						
					} //is processed					
				}
			}


			std::ofstream group_img_file;
			group_img_file.open("C:/Users/hisham/Dropbox/trud/edges-master/cv_seg_img.txt");
			for (int r = 0; r < group_idx_img.rows; r++){
				for (int c = 0; c < group_idx_img.cols; c++){
					group_img_file << group_idx_img.at<int>(r, c) << " ";
				}
				group_img_file << std::endl;
			}
			group_img_file.close();




			for (int i = 0; i < all_orientations.size(); i++){
				double orientation = 0;
				std::vector<double> orientation_list = all_orientations.at(i);
				for (int j = 0; j < orientation_list.size(); j++){
					orientation += orientation_list.at(j);
				}
				group_mean_orientation.push_back(orientation / ((double)orientation_list.size()));
			}

			for (int i = 0; i < all_magnitudes.size(); i++){
				double magnitude = 0;
				std::vector<double> magnitude_list = all_magnitudes.at(i);
				for (int j = 0; j < magnitude_list.size(); j++){
					magnitude += magnitude_list.at(j);
				}
				group_sum_magnitude.push_back(magnitude);
				group_mean_magnitude.push_back(magnitude / ((double)magnitude_list.size()));
			}

			for (int i = 0; i < all_positions.size(); i++){
				//printf("loading positions %d \n", i);
				if (i == 23){
					int d = 5;
				}
				std::vector<Vec2i> position_list = all_positions.at(i);
				group_position.push_back(position_list.at(0)); //just grab the first position

			}

			return true;


		}



		void ObjectnessEdgeBoxes::get_window_list(std::vector<Vec4i> &window_list,
			std::vector<double> &score_list, 
			std::vector<float> &aspect_list,
			std::vector<float> &width_list, 
			float &iou, 
			float &thresh, 
			float &start_t, 
			float &end_t, 
			float &num_t, 
			float &start_width, 
			float &end_width, 
			float &num_width){

			int img_width = _params.width;
			int img_height = _params.height;

			window_list.empty();
			aspect_list.empty();
			width_list.empty();
			score_list.empty();

			int counter = 0;

			std::ofstream box_file;
			box_file.open("C:/Users/hisham/Dropbox/trud/edges-master/box_list.txt");

			int width_step = (end_width - start_width) / num_width;
			float aspect_step = ((1 / (start_t)) - start_t ) / num_t;
			for (int width = start_width; width < end_width; width += width_step){
				for (float aspect = (start_t); aspect <1 / (start_t); aspect += aspect_step){

					//printf("Finished aspect ratio: %f, width: %d \n", aspect, width);

					int height = width / aspect;

					int x_step = std::max(floor((width -  width*iou) / (1 + iou)), 1.0f);
					int y_step = std::max( floor((height -  height*iou) / (1 + iou)), 1.0f);

					printf("Diag: [%d, %f] \n", width, aspect);

					//create a sliding window:
					for (int xx = 0; xx < img_width - width; xx += x_step){
						for (int yy = 0; yy < img_height - height; yy += y_step){
							counter++;

							//if ((xx == 207) & yy == 406 & width = 44 & abs(aspect - .333) < .1){
							if ((xx == 64) & (yy == 333) & (width == 40) & (abs(aspect - .870061) < .1)){
								int d = 5;
							}
							//printf("xx = %d, yy=%d, width = %d, aspect = %f \n", xx, yy, width, aspect);
							
							Vec4i box(yy,xx,yy+height,xx+width);


							int r0 = box[0];
							int r1 = box[2];
							int c0 = box[1];
							int c1 = box[3];



							box_file << r0 << " " << r1 << " " << c0 << " " << c1 << std::endl;

							




							//box[0] = 153;
							//box[2] = 191;
							//box[1] = 84;
							//box[3] = 124;


							//c1 = 124;

							bool in_bounds = true;

							if (r0 < 0 || r0 > _params.height - 1){ in_bounds = false; }

							if (r1 < 0 || r1 > _params.height - 1){ in_bounds = false; }

							if (c0 < 0 || c0 > _params.width - 1){ in_bounds = false; }

							if (c1 < 0 || c1 > _params.width - 1){ in_bounds = false; }

							if (in_bounds){
								std::clock_t start = std::clock();
								double score = scoreBoxParams(box);
								//double score = 0.0;
								//printf("Elapsed time for score: %03d \n", (int)(std::clock() - start));
								//printf("Diag: [%d,%f]  [%d,%d,%d,%d] \n", width, aspect, box[0], box[1], box[2], box[3]);


								if (score > 0.0f){
									int d = 5;
								}

								if (score > thresh){

									window_list.push_back(box);
									aspect_list.push_back(aspect);
									width_list.push_back(width);
									score_list.push_back(score);

									printf("Adding box: %d,%d,%d,%d aspect: %f, width: %d, score: %f, list_size: %d \n", box[0], box[1], box[2], box[3], aspect, width, (float)score, window_list.size());

									if (window_list.size() == 24){
										int d = 5;
									}
								}
							}
							
							
						}
					}


				}
			}

			printf("Num windows iterated over: %d \n", counter);
			box_file.close();

			return;

		}

		Vec4i ObjectnessEdgeBoxes::local_optimum_box(Vec4i &box, float &aspect, float &width){

			float variation = 0.01; //look within 1% of the values

			//determine step sizes for greedy gradient descent:
			int next_width = width*variation;
			float next_aspect = aspect*variation;
			std::vector<float> aspect_step_list;
			aspect_step_list.push_back(next_aspect);
			std::vector<int> width_step_list;
			width_step_list.push_back(next_width);
			int next_position = width*variation;
			std::vector<int> position_step_list;
			position_step_list.push_back(next_position);


			while (next_width / 2 > 2){
				next_width = next_width / 2;
				next_aspect = next_aspect / 2.0f;
				next_position = next_position / 2;
				width_step_list.push_back(next_width);
				aspect_step_list.push_back(next_aspect);
				position_step_list.push_back(next_position);
			}




			for (int i = 0; i < width_step_list.size(); i++){

				int width_step = width_step_list[i];
				float aspect_step = aspect_step_list[i];
				int height_step = width_step / aspect_step;
				int position_step = position_step_list[i];

				//generate boxes with a shift in all dimensions:
				std::vector<Vec4i> potential_boxes;
				potential_boxes.push_back(box);

				for (int j = 0; j < 16; j++){ //
					int flip = j;
					int coeff0 = 2*( (flip >> 1) & 0x01) -1;
					int coeff1 = 2 * ((flip >> 1) & 0x01) - 1;
					int coeff2 = 2 * ((flip >> 1) & 0x01) - 1;
					int coeff3 = 2 * ((flip >> 1) & 0x01) - 1;

					
					int new_start_y = box[0] + coeff0*position_step;
					int new_start_x = box[1] + coeff1*position_step;
					int new_end_y = new_start_y + coeff3*height_step;
					int new_end_x = new_start_x + coeff3*width_step;

					Vec4i new_box(new_start_y, new_start_x, new_end_y, new_end_x);
					potential_boxes.push_back(new_box);
				}

				//select best one
				double max_score = 0;
				int max_idx = 0;
				for (int bb = 0; bb < potential_boxes.size(); bb++){
					double score = scoreBoxParams(potential_boxes.at(bb));
					if (score > max_score){
						max_idx = bb;
						max_score = score;
					}
				}
				box = potential_boxes.at(max_idx);

				
			}


			return box;
		}


		float ObjectnessEdgeBoxes::calculateIOU(Vec4i &box1, Vec4i &box2){

			float center1_x = (box1[3] - box1[1]) / 2.0f + box1[1];
			float center1_y = (box1[2] - box1[0]) / 2.0f + box1[0];
			float center2_x = (box2[3] - box2[1]) / 2.0f + box2[1];
			float center2_y = (box2[2] - box2[0]) / 2.0f + box2[0];
			float diff_center_x = abs(center1_x - center2_x);
			float diff_center_y = abs(center1_y - center2_y);
			float width1 = abs(box1[3] - box1[1]);
			float width2 = abs(box2[3] - box2[1]);
			float height1 = abs(box1[2] - box1[0]);
			float height2 = abs(box2[2] - box2[0]);


			float intersection_start_x = 0.0f;
			float intersection_width = 0.0f;
			float intersection_start_y = 0.0f;
			float intersection_height = 0.0f;


			if (diff_center_x - (width1 / 2.0f + width2 / 2.0f) < 0.0f){ //they intersect
				if (box1[1] < box2[1]){
					intersection_start_x = box2[1];
				}
				else{
					intersection_start_x = box1[1];
				}

				if (box1[3] < box2[3]){
					intersection_width = intersection_start_x - box1[3];
				}
				else{
					intersection_width = intersection_start_x - box2[3];
				}
			}else{ //they don't intersect
				return 0.0f;
			}


			if (diff_center_y - (height1 / 2.0f + height2 / 2.0f) < 0.0f){ //they intersect
				if (box1[0] < box2[0]){
					intersection_start_y = box2[0];
				}
				else{
					intersection_start_y = box1[0];
				}

				if (box1[2] < box2[2]){
					intersection_height = intersection_start_y - box1[2];
				}
				else{
					intersection_height = intersection_start_y - box2[2];
				}
			}
			else{ //they don't intersect
				return 0.0f;
			}

			float intersection_area = intersection_width*intersection_height;
			float union_area = width1*height1 + width2*height2 - intersection_area;

			if (union_area == 0.0f){
				return 0.0f;
			}

			float IOU = intersection_area / union_area;
			if (IOU > 1.0f){
				int d = 5;
			}

			return intersection_area/union_area;
		}


		std::vector<Vec4i> ObjectnessEdgeBoxes::non_maximal_suppression(std::vector<Vec4i> &window_list, std::vector<double> &score_list){

			std::vector<Vec4i> sparse_window_list;
			for (int i = 0; i < window_list.size(); i++){
				float maxIOU = 0.0f;
				bool valid = true;
				for (int j = 0; j < sparse_window_list.size(); j++){

					maxIOU = std::max(maxIOU, calculateIOU(window_list.at(i), sparse_window_list.at(j)));
					if (maxIOU > 0.8f){
						valid = false;
					}
				}


				printf("Window [%d %d %d %d] IOU: %f \n", window_list.at(i)[0], window_list.at(i)[1], window_list.at(i)[2], window_list.at(i)[3], maxIOU);
				if (valid){ 
					sparse_window_list.push_back(window_list.at(i));
				}


			}

			return sparse_window_list;
		}


		void ObjectnessEdgeBoxes::initializeDataStructures(Mat &edgeImage, Mat &orientationImage){

			_params.width = edgeImage.cols;
			_params.height = edgeImage.rows;

			Mat group_idx_img;
			std::vector<double> group_mean_orientation, group_mean_magnitude, group_sum_magnitude;
			std::vector<Vec2i> group_position;


			clusterEdges(edgeImage,
				orientationImage,
				group_idx_img,
				group_mean_orientation,
				group_mean_magnitude,
				group_sum_magnitude,
				group_position);

			Mat affinity_matrix;
			computeAffinity(group_mean_orientation, group_idx_img, affinity_matrix);


			std::vector<std::vector<int>> row_intersection_list, column_intersection_list;
			Mat row_intersection_img, column_intersection_img;
			computeIntersectionDataStructure(group_idx_img,
				row_intersection_list,
				row_intersection_img,
				column_intersection_list,
				column_intersection_img);

			_params.row_intersection_list = row_intersection_list;
			_params.row_intersection_img = row_intersection_img;
			_params.column_intersection_list = column_intersection_list;
			_params.column_intersection_img = column_intersection_img;
			_params.affinity_matrix = affinity_matrix;
			_params.group_position = group_position;
			_params.group_sum_magnitude = group_sum_magnitude;
			_params.group_idx_img = group_idx_img;


			return;
		}


		void ObjectnessEdgeBoxes::gradient_x(Mat &input, Mat &output){
			//middle
			for (int i = 0; i < input.rows; i++){
				for (int j = 1; j < input.cols - 1; j++){
					output.at<double>(i, j) = 0.5*(input.at<double>(i, j + 1) - input.at<double>(i, j - 1));
				}

				output.at<double>(i, 0) = (input.at<double>(i, 1) - input.at<double>(i, 0));
				output.at<double>(i, input.cols - 1) = (input.at<double>(i, input.cols - 1) - input.at<double>(i, input.cols - 2));
			}

		}

		void ObjectnessEdgeBoxes::gradient_y(Mat &input, Mat &output){
			//middle
			for (int i = 1; i < input.rows-1; i++){
				for (int j = 0; j < input.cols ; j++){
					output.at<double>(i, j) = 0.5*(input.at<double>(i+1, j) - input.at<double>(i-1, j));
				}
			}

			//top and bottom rows
			for (int j = 0; j < input.cols; j++){
				output.at<double>(0, j) = (input.at<double>(1, j) - input.at<double>(0, j));
				output.at<double>(input.rows - 1, j) = (input.at<double>(input.rows - 1, j) - input.at<double>(input.rows - 2, j));
			}
		}

		bool ObjectnessEdgeBoxes::computeSaliencyDiagnostic(Mat &edgeImage, Mat &orientationImage, Mat &resultImage){
	



			int width = edgeImage.cols;
			int height = edgeImage.rows;
			resultImage = Mat(Size(width, height), CV_64FC1, Scalar(0.0f));

			//getOrientationImage(edgeImage, orientationImage);

			initializeDataStructures(edgeImage, orientationImage);


			std::vector<Vec4i> window_list;
			std::vector<double> score_list;
			

			int x_step = 20;
			int y_step = 20;

			for (int i = 0; i < height; i+=y_step){
				for (int j = 0; j < width; j += x_step){

					Vec4i box = Vec4i(i, j, std::min(i + y_step, height - 1), std::min(j + x_step, width - 1));
					window_list.push_back(box);
					score_list.push_back(scoreBoxParams(box));

				}
			}

		

			//evaluate all windows:
			for (int bb = 0; bb < window_list.size(); bb++){
				Vec4i box = window_list[bb];
				double score = score_list[bb];



				//assign all pixels in window to score value:
				printf("Filling in box: %d %d %d %d \n", box[0], box[1], box[2], box[3]);
				for (int rb = box[0]; rb < box[2]; rb++){
					for (int cb = box[1]; cb < box[3]; cb++){
						resultImage.at<double>(Point(cb, rb)) = score;
						//resultImage.at<double>(Point(cb, rb)) = rb;
						//printf("Value is now: %f \n", (float)resultImage.at<double>(Point(cb, rb)));
					}
				}
			}
	
		
			return true;
		}




		void ObjectnessEdgeBoxes::getOrientationImage(Mat &edgeImage, Mat &orientationImage){







			Mat oImage = Mat(edgeImage.rows, edgeImage.cols, edgeImage.depth());
			
			
			

			
			
			
			//compute orientation:
			//Mat triangle = Mat::ones(1,4,CV_64F)/4.0;
			Mat triangle = Mat::ones(1, 4, CV_64F);
			triangle.at<double>(0, 0) = 0.0;
			triangle.at<double>(0, 1) = .667;
			triangle.at<double>(0, 2) = .667;
			triangle.at<double>(0, 3) = 0.0;

			//double triangle[] = { 1.0/4.0, 1.0/4.0, 1.0 / 4.0, 1.0 / 4.0 };
			sepFilter2D(edgeImage, oImage, -1, triangle, triangle, Point(-1, -1), 0, BORDER_DEFAULT);





			Mat Oxx = Mat(edgeImage.rows, edgeImage.cols, edgeImage.depth());
			Mat Oyy = Mat(edgeImage.rows, edgeImage.cols, edgeImage.depth());
			Mat Oxy = Mat(edgeImage.rows, edgeImage.cols, edgeImage.depth());

			gradient_x(oImage, Oxx);
			gradient_x(Oxx, Oxx);





			gradient_y(oImage, Oyy);
			gradient_y(Oyy, Oyy);
			





			
			gradient_x(oImage, Oxy);
			gradient_y(Oxy, Oxy);


			for (int i = 0; i < edgeImage.rows; i++){
				for (int j = 0; j < edgeImage.cols; j++){
					double vOyy = Oyy.at<double>(i, j);
					double vOxx = Oxx.at<double>(i, j);
					double vOxy = Oxy.at<double>(i, j);
					int sgn = ((-vOxy >0) - (-vOxy < 0))*(abs(vOxy)>.001);

					//oImage.at<double>(i, j) = sgn;
					oImage.at<double>(i, j) = fmod(atan((vOyy * sgn / (vOxx + 0.00001))), 3.14);
				}
			}




			orientationImage = oImage;




		}

		void ObjectnessEdgeBoxes::getBoxScores(Mat &edgeImage, Mat &orientationImage, std::vector<Vec4i> &box_list, std::vector<double> &score_list){

			int width = edgeImage.cols;
			int height = edgeImage.rows;

			//threshold edge image:
			for (int i = 0; i < edgeImage.rows; i++){
				for (int j = 0; j < edgeImage.cols; j++){
					edgeImage.at<double>(i, j) = (edgeImage.at<double>(i, j) > 0.1)*edgeImage.at<double>(i, j);
				}
			}


			initializeDataStructures(edgeImage, orientationImage);



			//create window list:
			std::vector<Vec4i> window_list;
			std::vector<float> aspect_list;
			std::vector<float> width_list;
			//std::vector<double> score_list;
			float thresh = .02;
			float start_t = 0.33f;
			float end_t = 3.0f;
			float start_width = 40.0f;
			float end_width = edgeImage.cols;
			float num_t = 10;
			float num_width = 10;
			float iou = 0.65f;
			get_window_list(window_list, score_list, aspect_list, width_list, iou, thresh, start_t, end_t, num_t, start_width, end_width, num_width);

			printf("Number of initial windows: %d\n", window_list.size());
			for (int i = 0; i < window_list.size(); i++){
				printf("Window %d: [%d, %d, %d, %d] \n", i, window_list.at(i)[0], window_list.at(i)[1], window_list.at(i)[2], window_list.at(i)[3]);
			}


			//perform gradient descent:
			for (int i = 0; i < window_list.size(); i++){
				Vec4i box = local_optimum_box(window_list[i], aspect_list[i], width_list[i]);
				printf("Local optimum for window %d / %d ; %d, %d, %d, %d \n", i, window_list.size(), box[0], box[1], box[2], box[3]);
				window_list[i] = box;
			}


			//perform non maximal suppression:
			window_list = non_maximal_suppression(window_list, score_list);

			//score the boxes:
			score_list.clear();
			box_list.clear();
			_score_list.clear();
			_box_list.clear();
			for (int i = 0; i < window_list.size(); i++){
				double score = scoreBoxParams(window_list.at(i));
				_score_list.push_back((float)score);
				_box_list.push_back(window_list.at(i));
				box_list.push_back(window_list.at(i));
				score_list.push_back(score);

			}


			return;

		}


		bool ObjectnessEdgeBoxes::computeSaliencyMap( Mat &edgeImage, Mat &orientationImage, Mat &resultImage){

			//generate window list:

			Vec4i box1 = Vec4i(0, 0, 10, 10);
			Vec4i box2 = Vec4i(5, 0, 15, 10);
			float IOU = calculateIOU(box1, box2);

			std::vector<Vec4i> boxList;
			std::vector<double> scoreList;
			getBoxScores(edgeImage, orientationImage, boxList, scoreList);

			//evaluate all windows:
			for (int bb = 0; bb < boxList.size(); bb++){
				Vec4i box = boxList[bb];
				double score = scoreList[bb];

				//assign all pixels in window to score value:
				printf("Filling in box %d of %d: %d %d %d %d \n", bb, boxList.size(), box[0], box[1], box[2], box[3]);
				for (int rb = box[0]; rb < box[2]; rb++){
					for (int cb = box[1]; cb < box[3]; cb++){
						resultImage.at<double>(Point(cb, rb)) = max(score, resultImage.at<double>(Point(cb, rb)));
						//resultImage.at<double>(Point(cb, rb)) = rb;
						//printf("Value is now: %f \n", (float)resultImage.at<double>(Point(cb, rb)));
					}
				}
			}


			return true;
		}

	}
}