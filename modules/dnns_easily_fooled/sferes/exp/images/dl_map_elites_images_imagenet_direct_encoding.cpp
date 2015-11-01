#include "dl_images.hpp"

#include <iostream>
#include <sferes/phen/parameters.hpp>
#include "gen/evo_float_image.hpp"
#include <sferes/eval/eval.hpp>

#include "stat/best_fit_map_image.hpp"
#include "stat/stat_map_image.hpp"

#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>


// Evolutionary algorithms --------------------------------
#include "fit/fit_map_deep_learning.hpp"
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include <modules/nn2/gen_hyper_nn.hpp>
#include "phen/phen_image_direct.hpp"

#include <glog/logging.h>
// Caffe -------------------------------------------------

#include <modules/map_elite/map_elite.hpp>
#include "eval/mpi_parallel.hpp" // MPI
#include "continue_run/continue_run.hpp" // MPI

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float_image;

struct Params
{
	struct cont
	{
	    static const int getPopIndex = 0;
  };

	struct log
	{
		SFERES_CONST bool best_image = false;
	};

	struct ea
	{
		SFERES_CONST size_t res_x = 1; // 256;
		SFERES_CONST size_t res_y = 1000; // 256;
	};

  struct evo_float_image
  {
    // we choose the polynomial mutation type
    SFERES_CONST mutation_t mutation_type = polynomial;
    // we choose the polynomial cross-over type
    SFERES_CONST cross_over_t cross_over_type = sbx;
    // the mutation rate of the real-valued vector
    static float mutation_rate;
    // the cross rate of the real-valued vector
    SFERES_CONST float cross_rate = 0.5f;
    // a parameter of the polynomial mutation
    SFERES_CONST float eta_m = 15.0f;
    // a parameter of the polynomial cross-over
    SFERES_CONST float eta_c = 10.0f;
  };

  struct pop
  {
  	//number of initial random points
  	SFERES_CONST size_t init_size = 200; // 1000
    // size of the population
    SFERES_CONST unsigned size = 200; //200;
    // number of generations
    SFERES_CONST unsigned nb_gen = 5010; //10,000;
    // how often should the result file be written (here, each 5
    // generation)
    static int dump_period;// 5;
    // how many individuals should be created during the random
    // generation process?
    SFERES_CONST int initial_aleat = 1;
    // used by RankSimple to select the pressure
    SFERES_CONST float coeff = 1.1f;
    // the number of individuals that are kept from on generation to
    // another (elitism)
    SFERES_CONST float keep_rate = 0.6f;
  };

  struct parameters
  {
    // maximum value of parameters
    SFERES_CONST float min = -10.0f;
    // minimum value
    SFERES_CONST float max = 10.0f;
  };

  // Specific settings for MNIST database of grayscale
	struct image : ParamsCaffe::image
	{
		static const std::string model_definition;
		static const std::string pretrained_model;
	};

};

// Initialize the parameter files for Caffe network.
#ifdef LOCAL_RUN

const std::string Params::image::model_definition = "/home/anh/src/model/imagenet_deploy_image_memory_data.prototxt";
const std::string Params::image::pretrained_model = "/home/anh/src/model/caffe_reference_imagenet_model";

#else

const std::string Params::image::model_definition = "/project/EvolvingAI/anguyen8/model/imagenet_deploy_image_memory_data.prototxt";
const std::string Params::image::pretrained_model = "/project/EvolvingAI/anguyen8/model/caffe_reference_imagenet_model";

#endif

int Params::pop::dump_period = 1000;
float Params::evo_float_image::mutation_rate = 0.1f;


int main(int argc, char **argv)
{
	// Disable GLOG output from experiment and also Caffe
	// Comment out for debugging
	google::InitGoogleLogging("");
	google::SetStderrLogging(3);

  // Our fitness is the class FitTest (see above), that we will call
  // fit_t. Params is the set of parameters (struct Params) defined in
  // this file.
	typedef sferes::fit::FitMapDeepLearning<Params> fit_t;
	// We define the genotype. Here we choose EvoFloat (real
  // numbers). We evolve 10 real numbers, with the params defined in
  // Params (cf the beginning of this file)
	typedef gen::EvoFloatImage<Params::image::size * Params::image::size * 3, Params> gen_t;
  // This genotype should be simply transformed into a vector of
  // parameters (phen::Parameters). The genotype could also have been
  // transformed into a shape, a neural network... The phenotype need
  // to know which fitness to use; we pass fit_t.
  typedef phen::ImageDirect<gen_t, fit_t, Params> phen_t;
  // The evaluator is in charge of distributing the evaluation of the
  // population. It can be simple eval::Eval (nothing special),
  // parallel (for multicore machines, eval::Parallel) or distributed
  // (for clusters, eval::Mpi).
//  typedef eval::Eval<Params> eval_t;
  typedef eval::MpiParallel<Params> eval_t;	// TBB

  // Statistics gather data about the evolutionary process (mean
  // fitness, Pareto front, ...). Since they can also stores the best
  // individuals, they are the container of our results. We can add as
  // many statistics as required thanks to the boost::fusion::vector.
//  typedef boost::fusion::vector<stat::BestFit<phen_t, Params>, stat::MeanFit<Params> >  stat_t;
  typedef boost::fusion::vector<stat::MapImage<phen_t, Params>, stat::BestFitMapImage<phen_t, Params> >  stat_t;
  // Modifiers are functors that are run once all individuals have
  // been evalutated. Their typical use is to add some evolutionary
  // pressures towards diversity (e.g. fitness sharing). Here we don't
  // use this feature. As a consequence we use a "dummy" modifier that
  // does nothing.
  typedef modif::Dummy<> modifier_t;
  // We can finally put everything together. RankSimple is the
  // evolutianary algorithm. It is parametrized by the phenotype, the
  // evaluator, the statistics list, the modifier and the general params.
//  typedef ea::RankSimple<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
  typedef ea::MapElite<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
  // We now have a special class for our experiment: ea_t. The next
  // line instantiate an object of this class
  ea_t ea;
  // we can now process the command line options an run the
  // evolutionary algorithm (if a --load argument is passed, the file
  // is loaded; otherwise, the algorithm is launched).

  if (argc > 1) // if a number is provided on the command line
	{
		int randomSeed = atoi(argv[1]);
		printf("randomSeed:%i\n", randomSeed);
		srand(randomSeed);  //set it as the random seed

		boost::program_options::options_description add_opts =
								boost::program_options::options_description();

		shared_ptr<boost::program_options::option_description> opt (new boost::program_options::option_description(
				"continue,t", boost::program_options::value<std::string>(),
																"continue from the loaded file starting from the generation provided"
				));

		add_opts.add(opt);

		options::run_ea(argc, argv, ea, add_opts, false);
	}
	else
	{
		run_ea(argc, argv, ea);
	}

  return 0;
}
