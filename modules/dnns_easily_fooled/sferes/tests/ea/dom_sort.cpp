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

#ifndef NO_PARALLEL
#define NO_PARALLEL
#endif

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dom_sort

#include <boost/test/unit_test.hpp>
#include <boost/timer.hpp>
#include <cmath>
#include <iostream>
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/misc/rand.hpp>


#include <sferes/ea/dom_sort.hpp>
#include <sferes/ea/dom_sort_basic.hpp>

using namespace sferes;
using namespace sferes::gen::evo_float;

struct Params {
  struct evo_float {
    SFERES_CONST float cross_rate = 0.5f;
    SFERES_CONST float mutation_rate = 1.0f / 30.0f;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 10.0f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
  };
  struct parameters {
    SFERES_CONST float min = 0.0f;
    SFERES_CONST float max = 1.0f;
  };
};

SFERES_FITNESS(FitRand, sferes::fit::Fitness) {
public:
  template<typename Indiv>
  void eval(Indiv& ind) {
    this->_objs.resize(3);
    this->_objs[0] = sferes::misc::rand<float>(10);
    this->_objs[1] = sferes::misc::rand<float>(10);
    this->_objs[2] = sferes::misc::rand<float>(10);
  }
};


BOOST_AUTO_TEST_CASE(test_domsort) {
  srand(time(0));
  typedef gen::EvoFloat<30, Params> gen_t;
  typedef phen::Parameters<gen_t, FitRand<Params>, Params> phen_t;
  typedef boost::shared_ptr<phen_t> pphen_t;
  typedef std::vector<pphen_t> pop_t;

  pop_t pop;
  for (size_t i = 0; i < 2000; ++i) {
    boost::shared_ptr<phen_t> ind(new phen_t());
    ind->random();
    ind->fit().eval(*ind);
    pop.push_back(ind);
  }
  // basic
  std::vector<pop_t> fronts_basic;
  std::vector<size_t> ranks;
  boost::timer tbasic;
  ea::dom_sort_basic(pop, fronts_basic, ranks);
  std::cout << "dom sort basic (2000 indivs):"
            << tbasic.elapsed() << " s" << std::endl;
  // standard
  std::vector<pop_t> fronts;
  boost::timer tstd;
  ea::dom_sort(pop, fronts, ranks);
  std::cout << "dom sort deb (2000 indivs):"
            << tstd.elapsed() << " s" << std::endl;
  BOOST_CHECK_EQUAL(fronts.size(), fronts_basic.size());
  for (size_t i = 0; i < fronts.size(); ++i) {
    BOOST_CHECK_EQUAL(fronts[i].size(), fronts_basic[i].size());
    std::sort(fronts[i].begin(), fronts[i].end());
    std::sort(fronts_basic[i].begin(), fronts_basic[i].end());
    for (size_t j = 0; j < fronts[i].size(); ++j) {
      BOOST_CHECK_EQUAL(fronts[i][j]->fit().obj(0),
                        fronts_basic[i][j]->fit().obj(0));
      BOOST_CHECK_EQUAL(fronts[i][j]->fit().obj(1),
                        fronts_basic[i][j]->fit().obj(1));
    }
  }
}
