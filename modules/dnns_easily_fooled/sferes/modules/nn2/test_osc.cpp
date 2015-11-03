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
#define BOOST_TEST_MODULE nn_osc


#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cmath>
#include <algorithm>
#include "nn.hpp"

BOOST_AUTO_TEST_CASE(nn_osc)
{
  using namespace nn;
  typedef std::pair<float, float> weight_t;
  typedef PfIjspeert<params::Vectorf<3> > pf_t;
  typedef AfDirectT<weight_t> af_t;
  typedef Neuron<pf_t, af_t, weight_t> neuron_t;
  typedef Connection<weight_t, weight_t> connection_t;
  typedef NN<neuron_t, connection_t> nn_t;
  typedef nn_t::vertex_desc_t vertex_desc_t;
  nn_t nn;
  std::vector<vertex_desc_t> vs;

  float omega = 0.6;
  float x = 0;
  float r = 0.59;
  float phi_ij = 0.81;
  for (size_t i = 0; i < 5; ++i)
    {
      vertex_desc_t v = nn.add_neuron(boost::lexical_cast<std::string>(i));
      nn.get_neuron_by_vertex(v).get_pf().set_omega(omega);
      nn.get_neuron_by_vertex(v).get_pf().set_x(x);
      nn.get_neuron_by_vertex(v).get_pf().set_r(r);
      vs.push_back(v);
    }

  for (size_t i = 0; i < 4; ++i)
    {
      nn.add_connection(vs[i], vs[i + 1], std::make_pair(5, phi_ij));
      nn.add_connection(vs[i + 1], vs[i], std::make_pair(5, -phi_ij));
    }


  nn.init();
  for (size_t s = 0; s < 1000; ++s)
    {
      std::vector<weight_t> in;
      nn.step(in);
      for (size_t i = 0; i < vs.size(); ++i)
	std::cout<< i << " "
		 << nn.get_neuron_by_vertex(vs[i]).get_pf().get_theta_i()
		 << std::endl;
    }
  // you should have beautiful oscillations

}
