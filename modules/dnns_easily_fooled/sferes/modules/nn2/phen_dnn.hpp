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




#ifndef PHEN_DNN_HPP
#define PHEN_DNN_HPP

#include <map>
#include <sferes/phen/indiv.hpp>
#include "gen_dnn.hpp"

namespace sferes
{
  namespace phen
  {
    SFERES_INDIV(Dnn, Indiv)
      {
      public:
	void develop()
	{
	  // develop the parameters
	  BGL_FORALL_VERTICES_T(v, this->gen().get_graph(), 
				typename nn_t::graph_t)
	    {
	      this->gen().get_graph()[v].get_afparams().develop();
	      this->gen().get_graph()[v].get_pfparams().develop();
	    }
	  BGL_FORALL_EDGES_T(e, this->gen().get_graph(), 
			     typename nn_t::graph_t)
	    {
	      this->gen().get_graph()[e].get_weight().develop();
	    }
	  // init everything
	  this->_gen.init();	  
	}
	void show(std::ostream& os) {  this->gen().write(os);  }
	typedef typename Gen::nn_t nn_t;
	nn_t& nn() { return this->gen(); }
	const nn_t& nn() const { return this->gen(); }
      protected:
      };
  }
}


#endif
