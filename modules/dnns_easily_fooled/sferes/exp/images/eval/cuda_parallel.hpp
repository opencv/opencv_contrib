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

#ifndef EVAL_CUDA_PARALLEL_HPP_
#define EVAL_CUDA_PARALLEL_HPP_

#include <sferes/parallel.hpp>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdexcept>

namespace sferes {

	namespace caffe
	{
	 /**
		 * Using a shared_ptr to hold a pointer to a statically allocated object.
		 * http://www.boost.org/doc/libs/1_55_0/libs/smart_ptr/sp_techniques.html#static
		 */
		struct null_deleter
		{
				void operator()(void const *) const
				{
				}
		};

		class CaffeFactory
		{
		private:
			static bool initialized;
			static Net<float>* _net_1;
			static Net<float>* _net_2;
			static int _status;

		public:
			static shared_ptr<Net<float> > getCaffe(const std::string model_definition, const std::string pretrained_model)
			{
				if (!initialized)
				{
					// Initialize Caffe net 1
					_net_1 = new Net<float>(model_definition);

					// Get the trained model
					_net_1->CopyTrainedLayersFrom(pretrained_model);

					// Initialize Caffe net 2
					_net_2 = new Net<float>(model_definition);

					// Get the trained model
					_net_2->CopyTrainedLayersFrom(pretrained_model);

					initialized = true;
				}

				if (_status == 1)
				{
					_status = 2;
					shared_ptr<Net<float> > c(_net_1, null_deleter());
					return c;
				}
				else
				{
					_status = 1;
					shared_ptr<Net<float> > c(_net_2, null_deleter());
					return c;
				}
			}

			CaffeFactory()
			{
				initialized = false;
				_status = 1;
			}

		};
	}

  namespace eval {
  	/**
  	 * Develop phenotypes in parallel using TBB.
  	 */
    template<typename Phen>
    struct _parallel_develop {
      typedef std::vector<boost::shared_ptr<Phen> > pop_t;
      pop_t _pop;

      ~_parallel_develop() { }
      _parallel_develop(pop_t& pop) : _pop(pop) {}
      _parallel_develop(const _parallel_develop& ev) : _pop(ev._pop) {}

      void operator() (const parallel::range_t& r) const {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          assert(i < _pop.size());
          _pop[i]->develop();
        }
      }
    };


    SFERES_CLASS(CudaParallel)
    {
    private:

      /**
  		 * Develop phenotypes in parallel using TBB.
  		 */
  		template<typename Phen>
  		struct _parallel_cuda_eval {
  			typedef std::vector<boost::shared_ptr<Phen> > pop_t;
  			pop_t _pop;

  			~_parallel_cuda_eval() { }
  			_parallel_cuda_eval(pop_t& pop) : _pop(pop) {}
  			_parallel_cuda_eval(const _parallel_cuda_eval& ev) : _pop(ev._pop) {}

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
					shared_ptr<Net<float> > caffe_test_net = sferes::caffe::CaffeFactory::getCaffe(
							Params::image::model_definition,
							Params::image::pretrained_model
							);

//					shared_ptr<Net<float> > caffe_test_net =
//					boost::shared_ptr<Net<float> >(new Net<float>(Params::image::model_definition));
//
//					// Get the trained model
//					caffe_test_net->CopyTrainedLayersFrom(Params::image::pretrained_model);

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

    public:
      template<typename Phen>
      void eval(std::vector<boost::shared_ptr<Phen> >& pop, size_t begin, size_t end)
      {
        dbg::trace trace("eval", DBG_HERE);

        assert(pop.size());
				assert(begin < pop.size());
				assert(end <= pop.size());

				// Develop phenotypes in parallel using TBB.
				// The barrier is implicitly set here after the for-loop in TBB.
				//parallel::init();
				// We have only 2 GPUs per node
				//tbb::task_scheduler_init init1(4);
				parallel::p_for(parallel::range_t(begin, end),
						_parallel_develop<Phen>(pop));

        // Number of eval iterations
        const size_t count = end - begin;

        LOG(INFO) << "Size: " << count << " vs " << Params::image::batch << "\n";

        // Load balancing
				// We have only 2 GPUs per node
				//tbb::task_scheduler_init init2(2);

				parallel::p_for(
						parallel::range_t(begin, end, Params::image::batch),
						_parallel_cuda_eval<Phen>(pop));
      }

    };

  }
}

bool sferes::caffe::CaffeFactory::initialized;
int sferes::caffe::CaffeFactory::_status;
Net<float>* sferes::caffe::CaffeFactory::_net_1;
Net<float>* sferes::caffe::CaffeFactory::_net_2;

#endif
