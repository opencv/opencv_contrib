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

		std::vector<float> ObjectnessEdgeBoxes::getobjectnessValues()
		{
			return objectnessValues;
		}

		void ObjectnessEdgeBoxes::setTrainingPath(std::string trainingPath)
		{
			_trainingPath = trainingPath;
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
				edge_weight_list.push_back(1.0f);
			}


			if (edge_idx_list.size() > 0){
				for (int i = 0; i < edge_idx_list.size(); i++){
					edge_weight_list.at(edge_idx_list.at(i)) = 0;
				}
			}
			else{
				for (int i = 0; i < N; i++){
					edge_weight_list.at(i) = 0.0f;
				}
			}


			std::vector<int> todo_list;
			int todo_counter;
			for (int i = 0; i < edge_idx_list.size(); i++){
				todo_counter = 0;
				todo_list.push_back(i); //source equals straddling node

				for (int tt = todo_counter; tt < todo_list.size(); tt++){
					double source_affinity = 1 - edge_weight_list.at(i);
					for (int nn = 0; nn < N; nn++){ //for all possible nodes
						if (nn != tt){ 
							//find affinity between source and current node:
							double source_to_node_affinity = affinity_matrix.at<double>(Point(nn,i));

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
								edge_weight_list.at(nn) = 0;
							}
						}
					}
					todo_counter++;
				}

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
			int c1 = box[2];

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


			std::vector<std::vector<double>> all_orientations;
			std::vector<std::vector<double>> all_magnitudes;
			std::vector<std::vector<Vec2i>> all_positions;


			int cumulative_error = 0;
			int current_col, current_row, next_col, next_row = 0;
			double current_orientation, current_magnitude, new_orientation, new_magnitude;



			Mat isProcessed = Mat(Size(edgeImage.cols, edgeImage.rows), CV_8U, Scalar(0));
			group_idx_img = Mat(Size(edgeImage.cols, edgeImage.rows), CV_32S, Scalar(0));
			int segment_count = 0;
			for (int rr = 0; rr < edgeImage.rows; rr++){
				for (int cc = 0; cc < edgeImage.cols; cc++){
					if (isProcessed.at<bool>(Point(cc, rr))==false){
						double value = edgeImage.at<double>(Point(cc, rr));
						if (edgeImage.at<double>(Point(cc, rr)) > 0.1){
							
							

							current_col = cc;
							current_row = rr;

							segment_count++;
							cumulative_error = 0;
							current_orientation = edgeImage.at<double>(Point(cc, rr));
							current_magnitude = orientationImage.at<double>(Point(cc, rr));

							std::vector<double> this_group_magnitude_list, this_group_orientation_list;
							std::vector<Vec2i> this_group_position_list;

							this_group_magnitude_list.push_back(current_magnitude);
							this_group_orientation_list.push_back(current_orientation);
							this_group_position_list.push_back(Vec2i(cc,rr));


							while (cumulative_error < 3.14 / 2.0f && getNextComponent(edgeImage, rr, cc, isProcessed, next_row, next_col)){





								//find a connected component, add it to current edge group
								//if (getNextComponent(edgeImage, rr, cc, isProcessed, next_row, next_col)){

									new_orientation = orientationImage.at<double>(Point(next_col, next_row));
									new_magnitude = edgeImage.at<double>(Point(next_col, next_row));
									cumulative_error = cumulative_error + abs(current_orientation - new_orientation);

									current_orientation = new_orientation;
									current_magnitude = new_magnitude;
									current_col = next_col;
									current_row = next_row;

									isProcessed.at<bool>(Point(current_col, current_row)) = true;
									group_idx_img.at<int>(Point(current_col, current_row)) = segment_count;
									this_group_magnitude_list.push_back(current_magnitude);
									this_group_orientation_list.push_back(current_orientation);
									this_group_position_list.push_back(Vec2i(current_row,current_col));


								//}

							}//while
							all_magnitudes.push_back(this_group_magnitude_list);
							all_orientations.push_back(this_group_orientation_list);
							all_positions.push_back(this_group_position_list);

						} // edgeMag > 0.1
						else{
							//pixel has magnitude < 0.1
							isProcessed.at<bool>(Point(cc, rr)) = true;
						}
						
					} //is processed					
				}
			}


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
				std::vector<Vec2i> position_list = all_positions.at(i);
				group_position.push_back(position_list.at(0)); //just grab the first position
			}

			return true;


		}




		bool ObjectnessEdgeBoxes::computeSaliencyMap( Mat &edgeImage, Mat &orientationImage, Mat &resultImage){

			//generate window list:
			std::vector<Vec4i> window_list;
			int width = edgeImage.cols;
			int height = edgeImage.rows;
			resultImage = Mat(Size(width, height), CV_64FC1, Scalar(0.0f));


			//initialize data structures:
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


			/*
			for (int i = 0; i < column_intersection_list.size(); i++){
				std::vector<int> intersection_list = column_intersection_list.at(i);
				printf("%d : ", i);
				for (int j = 0; j < intersection_list.size(); j++){
					printf("%d,", intersection_list.at(j));
				}
				printf("\n");
			}*/


			int N = 6;
			for (int rm = 0; rm < N; rm++){
				for (int cm = 0; cm < N; cm++){

					Vec4i box(rm*(int)height / N, cm*(int)width / N, min((rm + 1)*(int)height / N, height-1), min((cm + 1)*(int)width / N, width-1));
					window_list.push_back(box);
						
				}
			}


			//evaluate all windows:
			for (int bb = 0; bb < window_list.size(); bb++){
				Vec4i box = window_list[bb];
				double score = scoreBox(box,
					row_intersection_list,
					row_intersection_img,
					column_intersection_list,
					column_intersection_img,
					affinity_matrix,
					group_position,
					group_sum_magnitude);

				//assign all pixels in window to score value:
				for (int rb = box[0]; rb < box[2]; rb++){
					for (int cb = box[1]; cb < box[3]; cb++){
						resultImage.at<double>(Point(cb, rb)) = score;
					}
				}
			}



			//resultImage = edgeImage;
			//resultImage = affinity_matrix;
			//resultImage = column_intersection_img;
			//resultImage.convertTo(resultImage, CV_64FC1);
			return true;
		}

	}
}