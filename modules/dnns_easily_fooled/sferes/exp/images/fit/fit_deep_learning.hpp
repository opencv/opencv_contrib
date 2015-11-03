//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#ifndef FIT_DEEP_LEARNING_HPP
#define FIT_DEEP_LEARNING_HPP

#include <sferes/fit/fitness.hpp>

// Caffe -------------------------------------------------
#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <stdio.h>

#include <caffe/caffe.hpp>
#include <caffe/vision_layers.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// Caffe -------------------------------------------------

using namespace caffe;

#define FIT_DEEP_LEARNING(Name) SFERES_FITNESS(Name, sferes::fit::Fitness)

namespace sferes
{
  namespace fit
  {
    SFERES_FITNESS(FitDeepLearning, sferes::fit::Fitness)
    {
    	protected:

    	/**
    	 * Crop an image based on the coordinates and the size of the crop.
    	 */
    	static cv::Mat crop(const cv::Mat& image,
    			const size_t x, const size_t y, const size_t width, const size_t height, const size_t offset, const bool flip = false)
    	{
    		// Setup a rectangle to define your region of interest
    		// int x, int y, int width, int height
    		cv::Rect myROI(x, y, width, height);	// top-left

    		// Crop the full image to that image contained by the rectangle myROI
    		// Note that this doesn't copy the data
    		cv::Mat croppedImage = image(myROI);

    		// Create a background image of size 256x256
    		cv::Mat background (Params::image::size, Params::image::size, CV_8UC3, cv::Scalar(255, 255, 255));

    		// Because we are using crop size of 227x227 which is odd, when the image size is even 256x256
    		// This adjustment helps aligning the crop.
    		int left = offset/2;
    		if (flip)
    		{
    			left++;
    		}

    		// Because Caffe requires 256x256 images, we paste the crop back to a dummy background.
    		croppedImage.copyTo(background(cv::Rect(left, offset/2, width, height)));

    		return background;
    	}

    	/**
    	 * Create ten crops (4 corners, 1 center, and x2 for mirrors).
    	 * Following Alex 2012 paper.
    	 * The 10 crops are added back to the list.
    	 */
    	static void _createTenCrops(const cv::Mat& image, vector<cv::Mat>& list)
    	{
    		// Offset
    		const int crop_size = Params::image::crop_size;
				const int offset = Params::image::size - crop_size;

				// 1. Top-left
				{
					cv::Mat cropped = crop(image, 0, 0, crop_size, crop_size, offset);

					// Add a crop to list
					list.push_back(cropped);

					cv::Mat flipped;
					cv::flip(crop(image, 0, 0, crop_size, crop_size, offset, true), flipped, 1);

					// Add a flipped crop to list
					list.push_back(flipped);
				}

				// 2. Top-Right
				{
					cv::Mat cropped = crop(image, offset, 0, crop_size, crop_size, offset);

					// Add a crop to list
					list.push_back(cropped);

					cv::Mat flipped;
					cv::flip(crop(image, offset, 0, crop_size, crop_size, offset, true), flipped, 1);

					// Add a flipped crop to list
					list.push_back(flipped);
				}

				// 3. Bottom-left
				{
					cv::Mat cropped = crop(image, 0, offset, crop_size, crop_size, offset);

					// Add a crop to list
					list.push_back(cropped);

					cv::Mat flipped;
					cv::flip(crop(image, 0, offset, crop_size, crop_size, offset, true), flipped, 1);

					// Add a flipped crop to list
					list.push_back(flipped);
				}

				// 4. Bottom-right
				{
					cv::Mat cropped = crop(image, offset, offset, crop_size, crop_size, offset);

					// Add a crop to list
					list.push_back(cropped);

					cv::Mat flipped;
					cv::flip(crop(image, offset, offset, crop_size, crop_size, offset, true), flipped, 1);

					// Add a flipped crop to list
					list.push_back(flipped);
				}

				// 5. Center and its mirror
				{
					cv::Mat cropped = crop(image, offset/2, offset/2, crop_size, crop_size, offset);

					// Add a crop to list
					list.push_back(cropped);

					cv::Mat flipped;
					cv::flip(crop(image, offset/2, offset/2, crop_size, crop_size, offset, true), flipped, 1);

					// Add a flipped crop to list
					list.push_back(flipped);
				}
    	}

    	private:
    	/**
    	 * Evaluate the given image to see its probability in the given category.
    	 */
    	float _getProbability(const cv::Mat& image, const int category)
    	{
    		this->initCaffeNet();	//Initialize caffe

				// Initialize test network
				shared_ptr<Net<float> > caffe_test_net = shared_ptr<Net<float> >( new Net<float>(Params::image::model_definition));

				// Get the trained model
				caffe_test_net->CopyTrainedLayersFrom(Params::image::pretrained_model);

				// Run ForwardPrefilled
				float loss;

				// Add images and labels manually to the ImageDataLayer
				vector<int> labels(10, 0);
				vector<cv::Mat> images;

				// Add images to the list
				if (Params::image::use_crops)
				{
					// Ten crops have been stored in the vector
					_createTenCrops(image, images);
				}
				else
				{
					images.push_back(image);
				}

				// Classify images
				const shared_ptr<ImageDataLayer<float> > image_data_layer =
						boost::static_pointer_cast<ImageDataLayer<float> >(
								caffe_test_net->layer_by_name("data"));

				image_data_layer->AddImagesAndLabels(images, labels);

				const vector<Blob<float>*>& result = caffe_test_net->ForwardPrefilled(&loss);

				// Get the highest layer of Softmax
				const float* softmax = result[1]->cpu_data();

				// If use 10 crops, we have to average the predictions of 10 crops
				if (Params::image::use_crops)
				{
					vector<double> values;

					// Average the predictions of evaluating 10 crops
					for(int i = 0; i < Params::image::num_categories; ++i)
					{
						boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean> > avg;

						for(int j = 0; j < 10 * Params::image::num_categories; j += Params::image::num_categories)
						{
							avg(softmax[i + j]);
						}

						double mean = boost::accumulators::mean(avg);

						values.push_back(mean);
					}

	    		return values[category];
				}
				// If use only 1 crop
				else
				{
					return softmax[category];
				}

    	}

     public:

      // Indiv will have the type defined in the main (phen_t)
      template<typename Indiv>
        void eval(const Indiv& ind)
      {
      	// Convert image to BGR before evaluating
				cv::Mat output;

				// Convert HLS into BGR because imwrite uses BGR color space
				cv::cvtColor(ind.image(), output, CV_HLS2BGR);

      	// Evolve images to be categorized as a soccer ball
      	this->_value = _getProbability(output, Params::image::category_id);
      }

      // Indiv will have the type defined in the main (phen_t)
			void setFitness(float value)
			{
				this->_value = value;
			}

      void initCaffeNet()
			{
				// Set test phase
				Caffe::set_phase(Caffe::TEST);

				if (Params::image::use_gpu)
				{
					// Set GPU mode
					Caffe::set_mode(Caffe::GPU);
				}
				else
				{
					// Set CPU mode
					Caffe::set_mode(Caffe::CPU);
				}
			}
    };
  }
}

#endif
