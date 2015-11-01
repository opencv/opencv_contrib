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




#ifndef DOM_SORT_HPP
#define DOM_SORT_HPP

#include <sferes/eval/parallel.hpp>
//#warning NEW algorithm for NSGA-2 (2 objectives)-> define SFERES_FAST_DOMSORT !
namespace sferes {
  namespace ea {
    namespace _dom_sort {
      template<typename Phen>
      struct count_dom {
        const std::vector<Phen>& pop;
        std::vector<size_t>& n;
        std::vector<std::vector<size_t> >& s;
        std::vector<size_t>& r;

        ~count_dom() { }
        count_dom(const std::vector<Phen>& pop_,
                  std::vector<size_t>& n_,
                  std::vector<std::vector<size_t> >& s_,
                  std::vector<size_t>& r_) :
          pop(pop_), n(n_), s(s_), r(r_) {
        }
        count_dom(const count_dom& ev) :
          pop(ev.pop), n(ev.n), s(ev.s), r(ev.r) {}
        void operator() (const parallel::range_t& range) const {
          assert(n.size() == pop.size());
          assert(s.size() == pop.size());
          for (size_t p = range.begin(); p != range.end(); ++p) {
            assert(s[p].empty());
            n[p] = 0;
            for (size_t q = 0; q < pop.size(); ++q) {
              int flag = fit::dominate_flag(pop[p], pop[q]);
              if (flag > 0)
                s[p].push_back(q);
              else if (flag < 0)
                ++n[p];
            }
            if (n[p] == 0)
              r[p] = 0;
          }
        }
      };


      // cf deb's paper on NSGA-2 :
      /// @article{deb2002nsga,
      // title={{NSGA-II}},
      //   author={Deb, K. and Pratap, A. and Agarwal, S. and Meyarivan, T. and Fast, A. and Algorithm, E.M.G.},
      //  journal={IEEE transactions on evolutionary computation},
      //   volume={6},
      //  number={2},
      //  year={2002}
      // }
      // this algorithm is in O(n^2)
      template<typename Indiv>
      inline void sort_deb(const std::vector<Indiv>& pop,
                           std::vector<std::vector<Indiv> >& fronts,
                           std::vector<size_t>& ranks) {
        assert(!pop.empty());
        std::vector<std::vector<size_t> > s(pop.size());
        std::vector<std::vector<size_t> > f(1);
        std::vector<size_t> n(pop.size());
        ranks.resize(pop.size());

        std::fill(ranks.begin(), ranks.end(), pop.size());

#ifndef NDEBUG
        BOOST_FOREACH(const Indiv& ind, pop)
        for (size_t i = 0; i < ind->fit().objs().size(); ++i) {
          assert(!std::isnan(ind->fit().objs()[i]));
        }
#endif

        parallel::p_for(parallel::range_t(0, pop.size()),
                        _dom_sort::count_dom<Indiv>(pop, n, s, ranks));

        for (size_t i = 0; i < pop.size(); ++i)
          if (ranks[i] == 0)
            f[0].push_back(i);

#ifndef NDEBUG
        BOOST_FOREACH(size_t k, n) {
          assert(k < pop.size());
        }

#endif
        assert(!f[0].empty());
        // second step : make layers
        size_t i = 0;
        while (!f[i].empty()) {
          f.push_back(std::vector<size_t>());
          for (size_t pp = 0; pp < f[i].size(); ++pp) {
            size_t p = f[i][pp];
            for (size_t k = 0; k < s[p].size(); ++k) {
              size_t q = s[p][k];
              assert(q != p);
              assert(n[q] != 0);
              --n[q];
              if (n[q] == 0) {
                ranks[q] = i + 1;
                f.back().push_back(q);
              }
            }
          }
          ++i;
          assert(i < f.size());
        }

#ifndef NDEBUG
        size_t size = 0;
        BOOST_FOREACH(std::vector<size_t>& v, f)
        size += v.size();
        assert(size == pop.size());
#endif

        // copy indivs to the res
        fronts.clear();
        fronts.resize(f.size());
        for (unsigned i = 0; i < f.size(); ++i)
          for (unsigned j = 0; j < f[i].size(); ++j)
            fronts[i].push_back(pop[f[i][j]]);

        assert(fronts.back().size() == 0);
        fronts.pop_back();
        assert(!fronts.empty());
        assert(!fronts[0].empty());
      }

      template<typename T>
      inline std::vector<T> new_vector(const T& t) {
        std::vector<T> v;
        v.push_back(t);
        return v;
      }

      struct _comp_fronts {
        // this functor is ONLY for dom_sort_fast2
        template<typename T>
        bool operator()(const T& f2, const T& f1) {
          assert(f1.size() == 1);
          assert(f1[0]->fit().objs().size() == 2);
          // we only need to compare f1 to the value of the last element of f2
          if (f1[0]->fit().obj(1) < f2.back()->fit().obj(1))
            return true;
          else
            return false;
        }
      };

      // see M. T. Jensen, 2003
      template<typename Indiv>
      inline void sort_2objs(const std::vector<Indiv>& pop,
                             std::vector<std::vector<Indiv> > & f,
                             std::vector<size_t>& ranks) {
        std::vector<Indiv> p = pop;
        parallel::sort(p.begin(), p.end(), fit::compare_objs_lex());
        f.push_back(new_vector(p[0]));
        size_t e = 0;
        for (size_t i = 1; i < p.size(); ++i) {
          if (p[i]->fit().obj(1) > f[e].back()->fit().obj(1)) { // !dominate(si, f_e)
            typename std::vector<std::vector<Indiv> >::iterator b =
              std::lower_bound(f.begin(), f.end(), new_vector(p[i]),
                               _comp_fronts());
            assert(b != f.end());
            b->push_back(p[i]);
          } else {
            ++e;
            f.push_back(new_vector(p[i]));
          }
        }
        // assign ranks to follow the interface
        for (size_t i = 0; i < f.size(); ++i)
          for (size_t j = 0; j < f[i].size(); ++j)
            f[i][j]->set_rank(i);
      }
    }

    template<typename Indiv>
    inline void dom_sort(const std::vector<Indiv>& pop,
                         std::vector<std::vector<Indiv> >& fronts,
                         std::vector<size_t>& ranks) {
#ifdef SFERES_FAST_DOMSORT
      if (pop[0]->fit().objs().size() == 2)
        _dom_sort::sort_2objs(pop, fronts, ranks);
      else
#endif
        _dom_sort::sort_deb(pop, fronts, ranks);
    }
  }
}

#endif
