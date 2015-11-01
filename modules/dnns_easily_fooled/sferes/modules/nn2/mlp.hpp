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




#ifndef _NN_MLP_HPP_
#define _NN_MLP_HPP_

#include "nn.hpp"
#include "connection.hpp"
#include "neuron.hpp"

namespace nn
{
  // a basic multi-layer perceptron (feed-forward neural network)
  // only one hidden layer in this version
  // there's one autmatically added input for the bias
  template<typename N, typename C>
  class Mlp : public NN<N, C>
  {
  public:
    typedef nn::NN<N, C> nn_t;
    typedef typename nn_t::io_t io_t;
    typedef typename nn_t::vertex_desc_t vertex_desc_t;
    typedef typename nn_t::edge_desc_t edge_desc_t;
    typedef typename nn_t::adj_it_t adj_it_t;    
    typedef typename nn_t::graph_t graph_t;
    typedef N neuron_t;
    typedef C conn_t;

    Mlp(size_t nb_inputs, 
	size_t nb_hidden, 
	size_t nb_outputs)
    {
      // neurons
      this->set_nb_inputs(nb_inputs + 1); 
      this->set_nb_outputs(nb_outputs);
      for (size_t i = 0; i < nb_hidden; ++i)
	_hidden_neurons.
	  push_back(this->add_neuron(std::string("h") + boost::lexical_cast<std::string>(i)));
      // connections
      this->full_connect(this->_inputs, this->_hidden_neurons, 
			 trait<typename N::weight_t>::zero());
      this->full_connect(this->_hidden_neurons, this->_outputs, 
			 trait<typename N::weight_t>::zero());
      // bias outputs too
      for (size_t i = 0; i < nb_outputs; ++i)
	this->add_connection(this->get_input(nb_inputs), this->get_output(i), 
			     trait<typename N::weight_t>::zero());
    }
    unsigned get_nb_inputs() const { return this->_inputs.size() - 1; }
    void step(const std::vector<io_t>& in)
    {
      assert(in.size() == this->get_nb_inputs());
      std::vector<io_t> inf = in;
      inf.push_back(1.0f);
      nn_t::_step(inf);
    }
  protected:
    std::vector<vertex_desc_t> _hidden_neurons;
  };

  // a basic MLP with float weights
  typedef Mlp<Neuron<PfWSum<>, AfSigmoidNoBias<> >, Connection<> > mlp_t;

}

#endif
