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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dnn

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <sferes/fit/fitness.hpp>
#include <sferes/phen/parameters.hpp>

#include "gen_dnn.hpp"
#include "gen_hyper_nn.hpp"
#include "phen_hyper_nn.hpp"

#include <boost/algorithm/string/join.hpp>
#include <vector>

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

struct Params1
{
  struct dnn
  {
    SFERES_CONST size_t nb_inputs = 3;
    SFERES_CONST size_t nb_outputs = 1;

    SFERES_CONST float m_rate_add_conn = 1.0f;
    SFERES_CONST float m_rate_del_conn = 0.3f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron = 1.0f;
    SFERES_CONST float m_rate_del_neuron = 0.2f;

    SFERES_CONST init_t init = ff;
  };
  struct evo_float
  {
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.5f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 15.0f;
  };
  struct parameters
  {
    SFERES_CONST float min = -2.0f;
    SFERES_CONST float max = 2.0f;
  };

  struct cppn
  {
     // params of the CPPN
    struct sampled
    {
      SFERES_ARRAY(float, values, 0, 1, 2);
      SFERES_CONST float mutation_rate = 0.1f;
      SFERES_CONST float cross_rate = 0.25f;
      SFERES_CONST bool ordered = false;
    };
    struct evo_float
    {
      SFERES_CONST float mutation_rate = 0.1f;
      SFERES_CONST float cross_rate = 0.1f;
      SFERES_CONST mutation_t mutation_type = polynomial;
      SFERES_CONST cross_over_t cross_over_type = sbx;
      SFERES_CONST float eta_m = 15.0f;
      SFERES_CONST float eta_c = 15.0f;
    };
  };
};


struct Params2
{
  struct dnn
  {
    SFERES_CONST size_t nb_inputs = 4;
    SFERES_CONST size_t nb_outputs = 1;

    SFERES_CONST float m_rate_add_conn = 1.0f;
    SFERES_CONST float m_rate_del_conn = 0.3f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron = 1.0f;
    SFERES_CONST float m_rate_del_neuron = 0.2f;

    SFERES_CONST float weight_sigma = 0.5f;
    SFERES_CONST float vect_sigma = 0.5f;
    SFERES_CONST float m_rate_weight = 1.0f;
    SFERES_CONST float m_rate_fparams = 1.0f;
    SFERES_CONST init_t init = ff;
  };
  struct evo_float
  {
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.1f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 15.0f;
  };
  struct parameters
  {
    SFERES_CONST float min = -2.0f;
    SFERES_CONST float max = 2.0f;
  };

  struct cppn
  {
     // params of the CPPN
    struct sampled
    {
      SFERES_ARRAY(float, values, 0, 1, 2);
      SFERES_CONST float mutation_rate = 0.1f;
      SFERES_CONST float cross_rate = 0.25f;
      SFERES_CONST bool ordered = false;
    };
    struct evo_float
    {
      SFERES_CONST float mutation_rate = 0.1f;
      SFERES_CONST float cross_rate = 0.1f;
      SFERES_CONST mutation_t mutation_type = polynomial;
      SFERES_CONST cross_over_t cross_over_type = sbx;
      SFERES_CONST float eta_m = 15.0f;
      SFERES_CONST float eta_c = 15.0f;
    };
  };
  struct hyper_nn
  {
    SFERES_ARRAY(float, substrate,
                 0.2f, 0.2f, // in 1
                 0.2f, 0.8f, // in 2
                 0.5f, 0.5f, // out 1
                 0.8f, 0.8f, // hidden 1
                 0.8f, 0.2f, // hidden 2
                 0.2f, 0.5f, // hidden 3
                 0.5f, 0.2f  // hidden 4
                 );
    SFERES_ARRAY(float, weights, -1, 0, 1);
    SFERES_ARRAY(float, bias, -1, 0, 1);
    SFERES_CONST size_t nb_inputs = 2;
    SFERES_CONST size_t nb_outputs = 1;
    SFERES_CONST size_t nb_hidden = 4;
    SFERES_CONST size_t nb_pfparams = 0;
    SFERES_CONST size_t nb_afparams = 1;
    SFERES_CONST float conn_threshold = 0.2f;
    SFERES_CONST float max_y = 10.0f;
    typedef nn::Neuron<nn::PfWSum<>,
                       nn::AfTanh<nn::params::Vectorf<1> > > neuron_t;
    typedef nn::Connection<> connection_t;
  };
};


BOOST_AUTO_TEST_CASE(gen_cppn)
{
	int nb_images = 0;

	for (; nb_images < 10; ++nb_images)
		{

//			time_t seconds = time (nb_images);
			srand (nb_images);

			std::string ts = boost::lexical_cast<std::string> (nb_images);

			typedef phen::Parameters<gen::EvoFloat<1, Params1>, fit::FitDummy<>,
					Params1> weight_t;
			typedef gen::HyperNn<weight_t, Params1> cppn_t;

			cppn_t gen1, gen2, gen3, gen4;

			gen1.random ();
//			for (size_t i = 0; i < 20; ++i)
//				gen1.mutate ();
			gen1.init ();
			BOOST_CHECK(gen1.get_depth () >= 1);
//			std::ofstream ofs2 ("./nn.dot");
//			gen1.write (ofs2);

			//  generate a picture
			char*pic = new char[256 * 256];
			std::vector<float> in (3);
			in[0] = 1;
			for (size_t i = 0; i < 256; ++i)
				for (size_t j = 0; j < 256; ++j)
					{
						in[1] = i / 128.0f - 1.0;
						in[2] = j / 128.0f - 1.0;
						for (size_t k = 0; k < gen1.get_depth (); ++k)
							gen1.step (in);
						pic[256 * i + j] = (int) (gen1.get_outf (0) * 256 + 255);
					}

			std::vector < std::string > list;
			list.push_back ("/home/anh/workspace/sferes/tmp/");
			list.push_back ("image_");
			list.push_back (ts);
			list.push_back (".pgm");
			std::string joined = boost::algorithm::join (list, "");

			std::ofstream ofs (joined.c_str ());
			ofs << "P5" << std::endl;
			ofs << "256 256" << std::endl;
			ofs << "255" << std::endl;
			ofs.write (pic, 256 * 256);

		}
}

/*
BOOST_AUTO_TEST_CASE(phen_hyper_nn)
{
  srand(time(0));
  typedef fit::FitDummy<> fit_t;
  typedef phen::Parameters<gen::EvoFloat<1, Params2>, fit::FitDummy<>, Params2> weight_t;
  typedef gen::HyperNn<weight_t, Params2> gen_t;
  typedef phen::HyperNn<gen_t, fit_t, Params2> phen_t;

  phen_t indiv;
  indiv.random();
  for (size_t i = 0; i < 5; ++i)
    indiv.mutate();
  std::ofstream ofs("/tmp/nn_substrate.svg");
  indiv.develop();
  indiv.show(ofs);
  // BOOST_CHECK_EQUAL(indiv.nn().get_nb_neurons(), 7);

}
*/
