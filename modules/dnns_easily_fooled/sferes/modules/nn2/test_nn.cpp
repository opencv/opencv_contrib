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
#define BOOST_TEST_MODULE nn


#include <boost/test/unit_test.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "nn.hpp"

BOOST_AUTO_TEST_CASE(nn_basic)
{
  using namespace nn;
  
  NN<Neuron<PfWSum<>, AfTanh<float> >, Connection<> > nn1, nn2, nn3;
 
  nn1.set_nb_inputs(1);
  nn1.set_nb_outputs(2);
  nn1.full_connect(nn1.get_inputs(), nn1.get_outputs(), 1.0f);
  nn1.init();
  std::vector<float> in(1); 
  in[0] = 1.0f;
  for (size_t i = 0; i < 200; ++i)
    nn1.step(in);

  float out = nn1.get_outf(0);
  BOOST_CHECK_CLOSE((double)out, tanh(5.0 * 1.0f), 1e-5);
  std::ofstream ofs("/tmp/test.dot");
  nn1.write(ofs);
  
  // check memory usage
  nn2 = nn1;
  for (size_t i = 0; i < 2000; ++i)
    {
      nn3 = nn2;
      nn2 = nn1;
      nn1 = nn3;
    }
  BOOST_CHECK_EQUAL(nn3.get_nb_connections(), nn1.get_nb_connections());
  BOOST_CHECK_EQUAL(nn3.get_nb_neurons(), nn1.get_nb_neurons());
  BOOST_CHECK_EQUAL(nn3.get_nb_inputs(), nn1.get_nb_inputs());
  BOOST_CHECK_EQUAL(nn3.get_nb_outputs(), nn1.get_nb_outputs());
  
}


BOOST_AUTO_TEST_CASE(nn_remove_small_weights)
{
  using namespace nn;
  NN<Neuron<PfWSum<>, AfTanh<float> >, Connection<> > nn;

  nn.set_nb_inputs(3);
  nn.set_nb_outputs(3);
  nn.add_connection_w(nn.get_input(0), nn.get_output(0), 0.5);
  nn.add_connection_w(nn.get_input(1), nn.get_output(0), 0.25);
  nn.add_connection_w(nn.get_input(2), nn.get_output(2), -0.25);
  nn.add_connection_w(nn.get_input(0), nn.get_output(2), 0.05);
  nn.add_connection_w(nn.get_input(2), nn.get_output(1), -0.05);
  nn.init();
  BOOST_CHECK_EQUAL(nn.get_nb_connections(), 5);
  int k = nn.remove_low_weights(0.1);
  std::cout << k << std::endl;
  BOOST_CHECK_EQUAL(nn.get_nb_connections(), 3);
}


BOOST_AUTO_TEST_CASE(nn_speed)
{
  using namespace nn;

  typedef NN<Neuron<PfWSum<>, AfTanh<float> >, Connection<> > nn_t;
  nn_t nn;
  
  nn.set_nb_inputs(40000);
  nn.set_nb_outputs(4);
  typedef std::vector<nn_t::vertex_desc_t> layer_t;
  //std::vector<layer_t> layers;
  //  layers.push_back(nn.get_inputs());
  // for (size_t i = 0; i < 10; ++i)
  //   {
  //     layer_t layer;
  //     for (size_t j = 0; j < 10; ++j)
  // 	layer.push_back(nn.add_neuron("n"));
  //     layers.push_back(layer);
  //   }
  // layers.push_back(nn.get_outputs());
  // for (size_t i = 0; i < layers.size() - 1; ++i)
  //   nn.full_connect(layers[i], layers[i + 1], 1.0);
  nn.full_connect(nn.get_inputs(), nn.get_outputs(), 0.25);
  
  nn.init();
  boost::timer timer;
  std::vector<float> in(40000); 
  std::fill(in.begin(), in.end(), 0.10f);
  for (size_t i = 0; i < 100; ++i)
    nn.step(in);
  std::cout<<"timer (1000 iterations):" << timer.elapsed() << std::endl;
  std::ofstream ofs("/tmp/test.dot");
  nn.write(ofs);

}
