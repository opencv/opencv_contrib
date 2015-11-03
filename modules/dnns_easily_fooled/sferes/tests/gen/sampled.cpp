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




#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sampled

#include <boost/test/unit_test.hpp>
#include <sferes/gen/sampled.hpp>
#include <tests/check_serialize.hpp>

struct Params1 {
  struct sampled {
    SFERES_ARRAY(float, values, 0, 1, 2, 3, 4);
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.25f;
    SFERES_CONST bool ordered = false;
  };
};

struct Params2 {
  struct sampled {
    SFERES_ARRAY(int, values, 0, 1, 2, 3, 4);
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.25f;
    SFERES_CONST bool ordered = false;
  };
};

template<typename P>
void test() {
  typedef sferes::gen::Sampled<10, P> gen_t;
  gen_t g[4];
  for (size_t k = 0; k < 4; ++k) {
    g[k].random();
    g[k].mutate();
    g[k].mutate();
  }
  g[0].cross(g[1], g[2], g[3]);
  for (size_t k = 0; k < 4; ++k)
    for (size_t i = 0; i < g[k].size(); ++i)
      BOOST_CHECK(g[k].data(i) == 0
                  || g[k].data(i) == 1
                  || g[k].data(i) == 2
                  || g[k].data(i) == 3
                  || g[k].data(i) == 4);

}


struct check_sampled_eq {
  template<typename T>
  void operator()(const T& gen1, const T& gen2) const {
    BOOST_CHECK_EQUAL(gen1.size(), gen2.size());
    for (size_t i = 0; i < gen1.size(); ++i)
      BOOST_CHECK_EQUAL(gen1.data(i), gen2.data(i));
  }
};

BOOST_AUTO_TEST_CASE(polynomial_sbx_serialize) {
  sferes::gen::Sampled<100, Params1> gen1, gen2;
  gen1.random();
  sferes::tests::check_serialize(gen1, gen2, check_sampled_eq());
}

BOOST_AUTO_TEST_CASE(sampled_gen_ordered) {
  test<Params1>();
}

BOOST_AUTO_TEST_CASE(sampled_gen_unordered) {
  test<Params2>();
}

