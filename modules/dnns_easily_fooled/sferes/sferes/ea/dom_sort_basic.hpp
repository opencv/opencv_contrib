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




#ifndef DOM_SORT_BASIC_HPP
#define DOM_SORT_BASIC_HPP


namespace sferes {
  namespace ea {
    namespace _dom_sort_basic {
      struct non_dominated_f {
        template<typename Indiv>
        inline bool operator() (const Indiv& ind, const std::vector<Indiv>& pop) const {

          BOOST_FOREACH(Indiv i, pop) {
            assert(i);
            assert(ind);
            if (fit::dominate(i, ind))
              return false;
          }
          return true;
        }
      };
    }
    template<typename Indiv, typename ND>
    inline void dom_sort_basic(const std::vector<Indiv>& pop,
                               std::vector<std::vector<Indiv> >& fronts,
                               const ND& nd,
                               std::vector<size_t>& ranks) {
      std::vector<size_t> p(pop.size());
      for (size_t i = 0; i < p.size(); ++i)
        p[i] = i;
      ranks.resize(pop.size());
      int rank = 0;
      while (!p.empty()) {
        std::vector<size_t> non_dominated;
        std::vector<Indiv> non_dominated_ind;
        std::vector<Indiv> tmp_pop;
        for (size_t i = 0; i < p.size(); ++i)
          tmp_pop.push_back(pop[p[i]]);
        for (size_t i = 0; i < p.size(); ++i)
          if (nd(pop[p[i]], tmp_pop)) {
            non_dominated.push_back(p[i]);
            ranks[p[i]] = rank;
            non_dominated_ind.push_back(pop[p[i]]);
          }
        assert(non_dominated.size());
        std::vector<size_t> np;
        std::set_difference(p.begin(), p.end(),
                            non_dominated.begin(), non_dominated.end(),
                            std::back_insert_iterator<std::vector<size_t> >(np));
        assert(np.size() < p.size());
        p.swap(np);
        fronts.push_back(non_dominated_ind);
        ++rank;
      }
    }

    template<typename Indiv>
    inline void dom_sort_basic(const std::vector<Indiv>& pop,
                               std::vector<std::vector<Indiv> >& fronts,
                               std::vector<size_t>& ranks) {
      dom_sort_basic(pop, fronts, _dom_sort_basic::non_dominated_f(), ranks);
    }
  }
}

#endif
