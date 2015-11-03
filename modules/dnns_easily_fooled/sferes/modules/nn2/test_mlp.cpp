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
#define BOOST_TEST_MODULE nn_mlp


#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cmath>
#include <algorithm>
#include "mlp.hpp"

BOOST_AUTO_TEST_CASE(nn_elman)
{
  using namespace nn;
  Mlp<Neuron<PfWSum<>, AfSigmoidNoBias<> >, Connection<> > nn(2, 3, 2);
  nn.init();
  BOOST_CHECK_EQUAL(nn.get_nb_neurons(), 3 + 3 + 2);
  BOOST_CHECK_EQUAL(nn.get_nb_connections(), 9 + 6 + 2);
  std::vector<float> in(2); 
  in[0] = 1.0f;
  in[1] = 0.2f;
  for (unsigned i = 0; i < 1000; ++i)
    nn.step(in);
  float out1 = nn.get_outf(0);
  float out2 = nn.get_outf(1);
  
}
