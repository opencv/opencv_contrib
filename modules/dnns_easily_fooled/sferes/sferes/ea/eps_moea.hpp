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




#ifndef EPSMOEA_HPP_
#define EPSMOEA_HPP_

#include <algorithm>
#include <limits>

#include <boost/foreach.hpp>

#include <sferes/stc.hpp>
#include <sferes/parallel.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/ea/common.hpp>
namespace sferes {
  namespace ea {
    // param : eps (array)
    // param : min_fit (array)
    // param : grain
    SFERES_EA(EpsMOEA, Ea) {
    public:
      void random_pop() {
        parallel::init();
        this->_pop.resize(Params::pop::size);
        parallel::p_for(parallel::range_t(0, this->_pop.size()),
                        random<Phen>(this->_pop));
        this->_eval.eval(this->_pop, 0, this->_pop.size());

        // create archive
        add_to_archive(this->_pop.front());
        for (typename pop_t :: const_iterator it = this->_pop.begin();
             it != this->_pop.end(); ++it)
          archive_acceptance(*it);
        sync_archive();
      }

      void epoch() {
        std::vector<indiv_t> indivs;

        for (size_t i = 0; i < Params::pop::grain; ++i) {
          indiv_t i1 = pop_selection();
          indiv_t i2 = archive_selection();
          indiv_t c1, c2;
          i1->cross(i2, c1, c2);
          indivs.push_back(c1);
          indivs.push_back(c2);
        }
        parallel::p_for(parallel::range_t(0, indivs.size()),
                        mutate<Phen>(indivs));

        this->_eval.eval(indivs, 0, indivs.size());

        BOOST_FOREACH(indiv_t i, indivs)
        if (pop_acceptance(i))
          archive_acceptance(i);
        sync_archive();
      }
      const std::vector<boost::shared_ptr<Phen> >& pareto_front() const {
        return _pareto_front;
      }
    protected:
      typedef boost::shared_ptr<Phen> indiv_t;
      typedef std::vector<indiv_t> pop_t;
      typedef std::pair<indiv_t, std::vector<float> > elite_t;
      typedef std::list<elite_t> archive_t;
      typedef std::vector<std::vector<float> > id_t;

      // elite
      archive_t pop_e;

      // identification vectors
      id_t id_b;

      pop_t _pareto_front;

      // keep pareto_front & elite synchronized (for stat reporting)
      void sync_archive() {
        _pareto_front.clear();
        BOOST_FOREACH(elite_t& i, pop_e)
        _pareto_front.push_back(i.first);
      }

      /// return a random + tournament individual in P
      indiv_t pop_selection() {

        indiv_t i1 = this->_pop[misc::rand(this->_pop.size())];
        indiv_t i2 = this->_pop[misc::rand(this->_pop.size())];

        int flag = check_dominance(i1, i2);
        switch (flag) {
        case 1: // a dom b
          return i1;
        case -1:
          return i2;
        case 0:
          if (misc::flip_coin())
            return i1;
          else
            return i2;
        }
        assert(0);
        return indiv_t();
      }

      ///  return a random individual in E
      indiv_t archive_selection() {
        return misc::rand_in_list(pop_e)->first;

      }

      /// try to insert the offspring in population
      /// return true if accepted
      bool pop_acceptance(indiv_t ind) {
        dbg::out(dbg::info, "epsmoea")<<"pop_acceptance :"<<indiv_str(ind)<<std::endl;
        int flag = 0;
        std::vector<int> array;
        int i = 0;

        for (typename pop_t :: const_iterator it = this->_pop.begin();
             it != this->_pop.end(); ++it) {
          flag = check_dominance(ind, *it);
          switch (flag) {
          case 1:
            array.push_back(i);
            break;
          case -1:
            dbg::out(dbg::info, "epsmoea")<<"pop_acceptance -> rejected"<<std::endl;
            return false;
          case 0:
            break;
          default:
            assert(0);
          }
          ++i;
        }

        int k;
        if (array.size())
          k = array[misc::rand(array.size())];
        else
          k = misc::rand(this->_pop.size());
        dbg::out(dbg::info, "epsmoea")<<"pop_acceptance, removing :"
                                      <<indiv_str(this->_pop[k])
                                      <<"  array.size()="<<array.size()<<std::endl;
        this->_pop[k] = ind;
        dbg::out(dbg::info, "epsmoea")<<"pop_acceptance -> accepted (k="<<k<<")"<<std::endl;
        return true;
      }

      ///  try to insert the offspring in pop_e
      /// return true if accepted
      bool archive_acceptance(indiv_t indiv) {
        dbg::out(dbg::info, "epsmoea")<<"archive_acceptance :"<<indiv_str(indiv)<<std::endl;
        elite_t ind = make_identification_vector(indiv);
        typename archive_t :: iterator it = pop_e.begin();
        bool same_box = false;
        do {
          assert(it != pop_e.end());
          int flag = check_box_dominance(ind, *it);
          switch (flag) {
          case 1:
            // if ind eps-dominates *it, we delete *it
            it = pop_e.erase(it);
            break;
          case 2:
            // *it eps-dominates ind, we stop the procedure
            dbg::out(dbg::info, "epsmoea")<<"archive_acceptance -> rejected"<<std::endl;
            return false;
          case 3:
            // both are non-dominated and are in different boxes, we
            // continue
            ++it;
            break;
          case 4:
            // both are non-dominated and are in same hyper-box
            same_box = true;
            break;
          default:
            assert(0);
          }
        } while(!same_box && it != pop_e.end());

        //=> the offspring (indiv) is eps-non-dominated
        // if it isn't in any filled box, we add it to the archive
        if (!same_box) {
          add_to_archive(indiv);
          return true;
        }
        assert(it != pop_e.end());
        // else, they are in the same box and we do a dominance check
        int flag = check_dominance(ind.first, it->first);
        float d1, d2;
        switch (flag) {
        case 1:
          pop_e.erase(it);
          add_to_archive(indiv);
          return true;
        case -1:
          return false;
        case 0:
          //both are non-dominated, we select the closest to the B
          //vector
          //  /!\ -> loss of a archived individual !
          d1 = dist_to_id(ind);
          d2 = dist_to_id(*it);
          if (d1 <= d2) {
            pop_e.erase(it);
            add_to_archive(indiv);
            return true;
          } else
            return false;
        default:
          assert(0);
        }
        assert(0);
        return false;
      }

      /// check dominance using the identification vector
      /// returns the following:
      ///	* 1 if a dominates b
      ///	* 2 if b dominates a
      ///	* 3 if a and b are non-dominated and a!=b (identification arrays unequal)
      ///	* 4 if a and b are non-dominated and a=b
      int check_box_dominance(const elite_t &a, const elite_t &b) const {
        int flag1 = 0, flag2 = 0;

        for (unsigned i = 0; i < Params::pop::eps_size(); ++i)
          if (a.second[i] > b.second[i])
            flag1 = 1;
          else if (b.second[i] > a.second[i])
            flag2 = 1;

        // a dominates b
        if (flag1 && !flag2)
          return 1;
        // b dominates a
        if (!flag1 && flag2)
          return 2;
        // a and b are non-dominated and a!=b (identification arrays unequal)
        if (flag1 && flag2)
          return 3;
        // a and b are non-dominated and a=b
        assert(!flag1 && !flag2);
        return 4;
      }

      /// standard dominance
      /// * 1 if a dominates b
      /// * -1 if b dominates a
      /// * 0 if both a and b are non-dominated
      int check_dominance(const indiv_t a, const indiv_t b) const {
        assert(a->fit().objs().size() == b->fit().objs().size());
        assert(a->fit().objs().size());
        size_t nb_objs = a->fit().objs().size();
        int flag1 = 0, flag2 = 0;
        for (size_t i = 0; i < nb_objs; ++i)
          if (a->fit().obj(i) > b->fit().obj(i))
            flag1 = 1;
          else if (a->fit().obj(i) < b->fit().obj(i))
            flag2 = 1;

        if (flag1 && !flag2)
          return 1;

        if (!flag1 && flag2)
          return -1;

        return 0;
      }

      /// compute an identification vector and return it
      elite_t make_identification_vector(indiv_t indiv) const {
        elite_t e;
        e.first = indiv;
        e.second.resize(Params::pop::eps_size());
        dbg::out(dbg::info, "epsmoea")<<"eps_size="<<Params::pop::eps_size()
                                      <<" fitsize:"<<indiv->fit().objs().size()
                                      <<std::endl;
        assert(e.second.size() == indiv->fit().objs().size());
        for (size_t i = 0; i < Params::pop::eps_size(); ++i)
          e.second[i] =
            ceil((indiv->fit().obj(i) - Params::pop::min_fit(i)) / Params::pop::eps(i));
        return e;
      }

      /// compute a squared euclidean distance between a's fitness and
      /// a's identification vector
      float dist_to_id(const elite_t& a) const {
        float res = 0;
        assert(a.first->fit().objs().size() == a.second.size());
        for (unsigned i = 0; i < a.first->fit().objs().size(); ++i)
          res += powf(a.first->fit().obj(i) - a.second[i], 2.0);
        return res;
      }

      /// make the identification vector and add to the archive / elite
      /// list
      void add_to_archive(indiv_t indiv) {
        dbg::out(dbg::info, "epsmoea")<<"add_to_archive :"<<indiv_str(indiv)<<std::endl;
        pop_e.push_back(make_identification_vector(indiv));
      }

      /// debug function
      std::string indiv_str(indiv_t indiv) {
        std::string s = "";
        for (size_t i = 0; i < indiv->fit().objs().size(); ++i)
          s += boost::lexical_cast<std::string>(indiv->fit().obj(i)) + " ";
        return s;
      }

    };
  }
}
#endif


