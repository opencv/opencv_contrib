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




#ifndef RANK_SIMPLE_HPP_
#define RANK_SIMPLE_HPP_

#include <algorithm>
#include <boost/foreach.hpp>
#include <sferes/stc.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/fit/fitness.hpp>

namespace sferes {
  namespace ea {
    SFERES_EA(RankSimple, Ea) {
    public:
      SFERES_CONST unsigned nb_keep = (unsigned)(Params::pop::keep_rate * Params::pop::size);

      void random_pop() {
        this->_pop.resize(Params::pop::size * Params::pop::initial_aleat);
        BOOST_FOREACH(boost::shared_ptr<Phen>& indiv, this->_pop) {
          indiv = boost::shared_ptr<Phen>(new Phen());
          indiv->random();
        }
        this->_eval.eval(this->_pop, 0, this->_pop.size());
        this->apply_modifier();
        std::partial_sort(this->_pop.begin(), this->_pop.begin() + Params::pop::size,
                          this->_pop.end(), fit::compare());
        this->_pop.resize(Params::pop::size);
      }
      void epoch() {
        assert(this->_pop.size());
        for (unsigned i = nb_keep; i < 	this->_pop.size(); i += 2) {
          unsigned r1 = _random_rank();
          unsigned r2 = _random_rank();
          boost::shared_ptr<Phen> i1, i2;
          this->_pop[r1]->cross(this->_pop[r2], i1, i2);
          i1->mutate();
          i2->mutate();
          this->_pop[i] = i1;
          this->_pop[i + 1] = i2;
        }
#ifndef EA_EVAL_ALL
        this->_eval.eval(this->_pop, nb_keep, Params::pop::size);
#else
        this->_eval.eval(this->_pop, 0, Params::pop::size);
#endif
        this->apply_modifier();
        std::partial_sort(this->_pop.begin(), this->_pop.begin() + nb_keep,
                          this->_pop.end(), fit::compare());
        dbg::out(dbg::info, "ea")<<"best fitness: " << this->_pop[0]->fit().value() << std::endl;
      }
    protected:
      unsigned _random_rank() {
        static float kappa = pow(Params::pop::coeff, nb_keep + 1.0f) - 1.0f;
        static float facteur = nb_keep / ::log(kappa + 1);
        return (unsigned) (this->_pop.size() - facteur * log(misc::rand<float>(1) * kappa + 1));
      }
    };
  }
}
#endif
