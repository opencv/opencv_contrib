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


#include "nn.hpp"

int main()
{
  using namespace nn;
  typedef NN<Neuron<PfWSum<>, AfTanh<float> >, Connection<> > nn_t;

  nn_t nn;

  BOOST_STATIC_CONSTEXPR size_t nb_io = 5;
  BOOST_STATIC_CONSTEXPR size_t nb_h = 100;

  nn.set_nb_inputs(nb_io);
  nn.set_nb_outputs(nb_io);

  std::vector<nn_t::vertex_desc_t> neurons;
  for (size_t i = 0; i < nb_h; ++i)
    neurons.push_back(nn.add_neuron("n"));

  for (size_t i = 0; i < nn.get_nb_inputs(); ++i)
    for (size_t j = 0; j < neurons.size(); ++j)
      nn.add_connection(nn.get_input(i), neurons[j], 1.0f);

  for (size_t i = 0; i < nn.get_nb_outputs(); ++i)
    for (size_t j = 0; j < neurons.size(); ++j)
      nn.add_connection(neurons[j], nn.get_output(i), 0.20f);

  std::vector<float> in(nn.get_nb_inputs()); 
  nn.init();
  std::fill(in.begin(), in.end(), 1.0f);
  size_t nb_steps = 50000;
  for (size_t i = 0; i < nb_steps; ++i)
    nn.step(in);
  return 0;
}
