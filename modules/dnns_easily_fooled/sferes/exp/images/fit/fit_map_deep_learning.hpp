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

#ifndef FIT_MAP_DEEP_LEARNING_HPP
#define FIT_MAP_DEEP_LEARNING_HPP

#include "fit_deep_learning.hpp"
#include <modules/map_elite/fit_map.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>

// Headers specifics to the computations we need
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>


#define FIT_MAP_DEEP_LEARNING(Name) SFERES_FITNESS(Name, sferes::fit::FitDeepLearning)

namespace sferes
{
  namespace fit
  {
    SFERES_FITNESS(FitMapDeepLearning, sferes::fit::FitDeepLearning)
    {
    	/*
    	private:
    	struct ArgMax
    	{
    		unsigned int category;
    		float probability;
    	};

    	ArgMax getMaxProbability(const cv::Mat& image)
			{
    		this->initCaffeNet();	//Initialize caffe

				// Initialize test network
				shared_ptr<Net<float> > caffe_test_net = shared_ptr<Net<float> >( new Net<float>(Params::image::model_definition));

				// Get the trained model
				caffe_test_net->CopyTrainedLayersFrom(Params::image::pretrained_model);

				// Run ForwardPrefilled
				float loss;		//	const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled(&loss);

				// Add images and labels manually to the ImageDataLayer
				vector<cv::Mat> images(1, image);
				vector<int> labels(1, 0);
				const shared_ptr<ImageDataLayer<float> > image_data_layer =
						boost::static_pointer_cast<ImageDataLayer<float> >(
								caffe_test_net->layer_by_name("data"));

				image_data_layer->AddImagesAndLabels(images, labels);

				vector<Blob<float>* > dummy_bottom_vec;
				const vector<Blob<float>*>& result = caffe_test_net->Forward(dummy_bottom_vec, &loss);

				// Get the highest layer of Softmax
				const float* argmax = result[1]->cpu_data();

				ArgMax m;
				m.category = (int) argmax[0];		// Category
				m.probability = (float) argmax[1];	// Probability

				return m;
			}
    	*/

    private:
    	void _setProbabilityList(const cv::Mat& image)
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
					this->_createTenCrops(image, images);
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

				vector<double> values;

				boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::max> > max;

				// Clear the probability in case it is called twice
				_prob.clear();

				// If use 10 crops, we have to average the predictions of 10 crops
				if (Params::image::use_crops)
				{
					// Average the predictions of evaluating 10 crops
					for(int i = 0; i < Params::image::num_categories; ++i)
					{
						boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean> > avg;

						for(int j = 0; j < 10 * Params::image::num_categories; j += Params::image::num_categories)
						{
							avg(softmax[i + j]);
						}

						double mean = boost::accumulators::mean(avg);

						// Push 1000 probabilities in the list
						_prob.push_back(mean);

						max(mean);	// Add this mean to a list for computing the max later
					}
				}
				else
				{
					for(int i = 0; i < Params::image::num_categories; ++i)
					{
						float v = softmax[i];

						// Push 1000 probabilities in the list
						_prob.push_back(v);

						max(v);	// Add this mean to a list for computing the max later
					}
				}

				float max_prob = boost::accumulators::max(max);

				// Set the fitness
				this->_value = max_prob;
			}

      public:
    	FitMapDeepLearning() : _prob(Params::image::num_categories) { }
			const std::vector<float>& desc() const { return _prob; }

			// Indiv will have the type defined in the main (phen_t)
			template<typename Indiv>
			void eval(const Indiv& ind)
			{
				if (Params::image::color)
				{
					// Convert image to BGR before evaluating
					cv::Mat output;

					// Convert HLS into BGR because imwrite uses BGR color space
					cv::cvtColor(ind.image(), output, CV_HLS2BGR);

					// Create an empty list to store get 1000 probabilities
					_setProbabilityList(output);
				}
				else	// Grayscale
				{
					// Create an empty list to store get 1000 probabilities
					_setProbabilityList(ind.image());
				}
			}

			float value(int category) const
			{
				assert(category < _prob.size());
				return _prob[category];
			}

			float value() const
			{
				return this->_value;
			}

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				sferes::fit::Fitness<Params,  typename stc::FindExact<FitMapDeepLearning<Params, Exact>, Exact>::ret>::serialize(ar, version);
				ar & BOOST_SERIALIZATION_NVP(_prob);
			}

      protected:
			  std::vector<float> _prob; // List of probabilities
    };
  }
}

#endif
