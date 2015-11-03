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




#ifndef GEN_HYPER_NN_HPP_
#define GEN_HYPER_NN_HPP_

#include "neuron.hpp"
#include "pf.hpp"

#include "gen_dnn_ff.hpp"
#include "af_cppn.hpp"

namespace sferes
{
  namespace gen
  {
    template<typename W, typename Params>
    class HyperNn : public DnnFF<nn::Neuron<nn::PfWSum<W>,
                                            nn::AfCppn<nn::cppn::AfParams<typename Params::cppn> > >,
                                 nn::Connection<W>,
                                 Params>
    {
      public:
        typedef DnnFF<nn::Neuron<nn::PfWSum<W>,
                                 nn::AfCppn<nn::cppn::AfParams<typename Params::cppn> > >,
                      nn::Connection<W>,
                      Params> nn_t;
        typedef typename nn_t::neuron_t neuron_t;
        typedef typename nn_t::conn_t conn_t;
        void init()
        {
          BGL_FORALL_EDGES_T(e, this->get_graph(),
                             typename nn_t::graph_t)
          this->get_graph()[e].get_weight().develop();

           // develop the parameters
          BGL_FORALL_VERTICES_T(v, this->get_graph(),
                                typename nn_t::graph_t)
          {
            this->get_graph()[v].get_afparams().develop();
            this->get_graph()[v].get_pfparams().develop();
          }
          nn_t::init();
        }
    };
  }
}

#endif
