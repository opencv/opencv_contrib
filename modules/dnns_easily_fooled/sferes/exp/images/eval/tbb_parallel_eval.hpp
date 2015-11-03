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

#ifndef EVAL_TBB_PARALLEL_EVAL_HPP_
#define EVAL_TBB_PARALLEL_EVAL_HPP_

#include <sferes/parallel.hpp>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tbb_parallel_develop.hpp"

#include <stdexcept>

namespace sferes {

  namespace eval {

  	/**
		 * Develop phenotypes in parallel using TBB.
		 */
		template<typename Phen>
		struct parallel_tbb_eval {
			typedef std::vector<boost::shared_ptr<Phen> > pop_t;
			pop_t _pop;
			std::string _model_definition;
			std::string _pretrained_model;

			~parallel_tbb_eval() { }
			parallel_tbb_eval(pop_t& pop, const std::string model_definition, const std::string pretrained_model) :
				_pop(pop),
				_model_definition(model_definition),
				_pretrained_model(pretrained_model)
			{
			}

			parallel_tbb_eval(const parallel_tbb_eval& ev) :
				_pop(ev._pop),
				_model_definition(_model_definition),
				_pretrained_model(_pretrained_model)
			{
			}

			void operator() (const parallel::range_t& r) const
			{
				size_t begin = r.begin();
				size_t end = r.end();

				LOG(INFO) << "Begin: " << begin << " --> " << end << "\n";

				dbg::trace trace("eval_cuda", DBG_HERE);
				assert(_pop.size());
				assert(begin < _pop.size());
				assert(end <= _pop.size());

				// Algorithm works as follow:
				// Send the individuals to Caffe first
				// Get back a list of results
				// Assign the results to individuals

				// Construct a list of images to be in the batch
				std::vector<cv::Mat> images(0);

				for (size_t i = begin; i < end; ++i)
				{
					cv::Mat output;
					_pop[i]->imageBGR(output);

					images.push_back( output );	// Add to a list of images
				}

				// Initialize Caffe net
				shared_ptr<Net<float> > caffe_test_net =
				boost::shared_ptr<Net<float> >(new Net<float>(_model_definition));

				// Get the trained model
				caffe_test_net->CopyTrainedLayersFrom(_pretrained_model);

				// Run ForwardPrefilled
				float loss;		//	const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled(&loss);

				// Number of eval iterations
				const size_t num_images = end - begin;

				// Add images and labels manually to the ImageDataLayer
				//	vector<cv::Mat> images(num_images, image);
				vector<int> labels(num_images, 0);
				const shared_ptr<ImageDataLayer<float> > image_data_layer =
					boost::static_pointer_cast<ImageDataLayer<float> >(
							caffe_test_net->layer_by_name("data"));

				image_data_layer->AddImagesAndLabels(images, labels);

				// Classify this batch of 512 images
				const vector<Blob<float>*>& result = caffe_test_net->ForwardPrefilled(&loss);

				// Get the highest layer of Softmax
				const float* argmaxs = result[1]->cpu_data();

				// Get back a list of results
				LOG(INFO) << "Number of results: " << result[1]->num() << "\n";

				// Assign the results to individuals
				for(int i = 0; i < num_images * 2; i += 2)
				{
					LOG(INFO)<< " Image: "<< i/2 + 1 << " class:" << argmaxs[i] << " : " << argmaxs[i+1] << "\n";

					int pop_index = begin + i/2;	// Index of individual in the batch

					// Set the fitness of this individual
					_pop[pop_index]->fit().setFitness((float) argmaxs[i+1]);

					// For Map-Elite, set the cell description
					_pop[pop_index]->fit().set_desc(0, argmaxs[i]);
				}
			}
		};

  }
}

#endif
