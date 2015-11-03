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

#include <iostream>
#include <cmath>
#include <algorithm>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/assign/list_of.hpp>

#include <sferes/fit/fitness.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/phen/parameters.hpp>

#include "gen_dnn.hpp"
#include "phen_dnn.hpp"

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

template<typename T1, typename T2>
void check_list_equal(const T1& v1, const T2& v2)
{
  BOOST_CHECK_EQUAL(v1.size(), v2.size());
  typename T1::const_iterator it1 = v1.begin();
  typename T1::const_iterator it2 = v2.begin();
  for (; it1 != v1.end(); ++it1, ++it2)
    BOOST_CHECK(fabs(*it1 - *it2) < 1e-3);
}
 
template<typename NN>
void check_nn_equal(NN& nn1, NN& nn2)
{
  nn1.init();
  nn2.init();

  BOOST_CHECK_EQUAL(nn1.get_nb_inputs(), nn2.get_nb_inputs());
  BOOST_CHECK_EQUAL(nn1.get_nb_outputs(), nn2.get_nb_outputs());
  BOOST_CHECK_EQUAL(nn1.get_nb_neurons(), nn2.get_nb_neurons());
  BOOST_CHECK_EQUAL(nn1.get_nb_connections(), nn2.get_nb_connections());
//   nn1.write("/tmp/tmp1.dot");
//   nn2.write("/tmp/tmp2.dot");
//   std::ifstream ifs1("/tmp/tmp1.dot"), ifs2("/tmp/tmp2.dot");
//   while(!ifs1.eof() && !ifs2.eof())
//     {
//       //if (ifs1.get() != ifs2.get()) exit(1);
//       BOOST_CHECK_EQUAL((char)ifs1.get(), (char)ifs2.get());
//     }

  std::pair<typename NN::vertex_it_t, typename NN::vertex_it_t> vp1 = 
    boost::vertices(nn1.get_graph());
  std::pair<typename NN::vertex_it_t, typename NN::vertex_it_t> vp2 = 
    boost::vertices(nn2.get_graph());
  while (vp1.first != vp1.second)
    {
      BOOST_CHECK_EQUAL(nn1.get_graph()[*vp1.first].get_in_degree(),
			nn2.get_graph()[*vp2.first].get_in_degree());
      check_list_equal(nn1.get_graph()[*vp1.first].get_afparams(),
		       nn2.get_graph()[*vp1.first].get_afparams());
      check_list_equal(nn1.get_graph()[*vp1.first].get_pfparams(),
		       nn2.get_graph()[*vp1.first].get_pfparams());
      ++vp1.first;
      ++vp2.first;
    }

}

 
struct Params
{
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
    // maximum value of parameters
    SFERES_CONST float min = -5.0f;
    // minimum value
    SFERES_CONST float max = 5.0f;
  };
  struct dnn
  {
    SFERES_CONST size_t nb_inputs	= 4;
    SFERES_CONST size_t nb_outputs	= 1;
    SFERES_CONST size_t min_nb_neurons	= 4;
    SFERES_CONST size_t max_nb_neurons	= 5;
    SFERES_CONST size_t min_nb_conns	= 100;
    SFERES_CONST size_t max_nb_conns	= 101;
    SFERES_CONST float  max_weight	= 2.0f;
    SFERES_CONST float  max_bias	= 2.0f;

    SFERES_CONST float m_rate_add_conn	= 1.0f;
    SFERES_CONST float m_rate_del_conn	= 1.0f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron  = 1.0f;
    SFERES_CONST float m_rate_del_neuron  = 1.0f;

    SFERES_CONST int io_param_evolving = true;
    SFERES_CONST init_t init = random_topology;
  };
};

BOOST_AUTO_TEST_CASE(direct_gen)
{
  using namespace nn;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  typedef PfWSum<weight_t> pf_t;
  typedef AfTanh<bias_t> af_t; 

  sferes::gen::Dnn<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen1, gen2, gen3, gen4;
      
  gen1.random();
  gen2.random();

  gen1.cross(gen2, gen3, gen4);
  gen3.mutate();
  gen4.mutate();
  gen2.mutate();
}



BOOST_AUTO_TEST_CASE(direct_nn_serialize)
{
  srand(0);

  using namespace nn;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  typedef PfWSum<weight_t> pf_t;
  typedef AfTanh<bias_t> af_t; 
  typedef sferes::gen::Dnn<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t;
  typedef phen::Dnn<gen_t, fit::FitDummy<>, Params> phen_t;


  typedef boost::archive::binary_oarchive oa_t;
  typedef boost::archive::binary_iarchive ia_t;

  for (size_t i = 0; i < 10; ++i) 
    {
      phen_t indiv[3];
      indiv[0].random();
      indiv[0].mutate();
      indiv[0].mutate();
      indiv[0].mutate();
      indiv[0].nn().init();
      {
	std::ofstream ofs("/tmp/serialize_nn1.bin", std::ios::binary);
	oa_t oa(ofs); 
	oa & indiv[0];
      }
      {
	std::ifstream ifs("/tmp/serialize_nn1.bin", std::ios::binary);
	ia_t ia(ifs);
	ia & indiv[1];
      }
      indiv[2].nn() = indiv[0].nn();
      using namespace boost::assign;
      std::vector<float> in = list_of(0.5f)(1.0f)(-0.25f)(1.101f);
      for (size_t j = 0; j < 3; ++j)
	indiv[j].nn().init();
      for (size_t i = 0; i < 10; ++i)
	for (size_t j = 0; j < 3; ++j)
	  indiv[j].nn().step(in);
      
      for (size_t j = 1; j < 3; ++j)
	BOOST_CHECK_CLOSE(indiv[0].nn().get_outf(0), indiv[j].nn().get_outf(0), 1e-5);
    }
}

