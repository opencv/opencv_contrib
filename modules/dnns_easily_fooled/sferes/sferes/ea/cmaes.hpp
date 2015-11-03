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

#ifndef CMAES_HPP_
#define CMAES_HPP_

#include <algorithm>
#include <boost/foreach.hpp>
#include <sferes/stc.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/parallel.hpp>
#include "cmaes_interface.h"

namespace sferes {

  namespace ea {

    SFERES_EA(Cmaes, Ea) {
    public:
      Cmaes() {
        _ar_funvals = cmaes_init(&_evo, dim, NULL, NULL, 0, 0, NULL);
        _lambda = cmaes_Get(&_evo, "lambda"); // default lambda (pop size)
      }
      ~Cmaes() {
        cmaes_exit(&_evo);
      }
      void random_pop() {
        // we don't really need the random here
        this->_pop.resize(_lambda);
        BOOST_FOREACH(boost::shared_ptr<Phen>&indiv, this->_pop) {
          indiv = boost::shared_ptr<Phen>(new Phen());
        }
      }
      void epoch() {
        //
        _cmaes_pop = cmaes_SamplePopulation(&_evo);
        // copy pop
        for (size_t i = 0; i < this->_pop.size(); ++i)
          for (size_t j = 0; j < this->_pop[i]->size(); ++j) {
            this->_pop[i]->gen().data(j, _cmaes_pop[i][j]);
            this->_pop[i]->develop();
          }
        // eval
        this->_eval.eval(this->_pop, 0, this->_pop.size());
        this->apply_modifier();
        for (size_t i = 0; i < this->_pop.size(); ++i) {
          //warning: CMAES minimizes the fitness...
          _ar_funvals[i] = - this->_pop[i]->fit().value();
        }
        //
        cmaes_UpdateDistribution(&_evo, _ar_funvals);
      }
    protected:
      SFERES_CONST size_t dim = Phen::gen_t::gen_size;
      cmaes_t _evo;
      double *_ar_funvals;
      double * const * _cmaes_pop;
      int _lambda;
    };
  }
}
#endif