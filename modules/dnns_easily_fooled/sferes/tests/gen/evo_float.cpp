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
#define BOOST_TEST_MODULE evo_float

#include <boost/test/unit_test.hpp>
#include <sferes/gen/evo_float.hpp>
#include <tests/check_serialize.hpp>

using namespace sferes::gen;
using namespace sferes::gen::evo_float;


template<int Size, typename P>
void test() {
  for (unsigned i = 0; i < 30; ++i) {
    EvoFloat<Size, P> gen1;
    EvoFloat<Size, P> gen2;
    EvoFloat<Size, P> gen3;
    EvoFloat<Size, P> gen4;
    gen1.random();
    gen1.mutate();
    gen1.cross(gen2, gen3, gen4);
  }
}

struct Params1 {
  struct evo_float {
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.1f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 15.0f;
  };
};


BOOST_AUTO_TEST_CASE(polynomial_sbx) {
  test<10, Params1>();
}

struct check_evofloat_eq {
  template<typename T>
  void operator()(const T& gen1, const T& gen2) const {
    BOOST_CHECK_EQUAL(gen1.size(), gen2.size());
    for (size_t i = 0; i < gen1.size(); ++i)
      BOOST_CHECK(fabs(gen1.data(i) - gen2.data(i) < 0.001));
  }
};

BOOST_AUTO_TEST_CASE(polynomial_sbx_serialize) {
  EvoFloat<10, Params1> gen1, gen2;
  gen1.random();
  sferes::tests::check_serialize(gen1, gen2, check_evofloat_eq());
}

struct Params2 {
  struct evo_float {
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.1f;
    SFERES_CONST mutation_t mutation_type = gaussian;
    SFERES_CONST cross_over_t cross_over_type = recombination;
    SFERES_CONST float sigma = 0.3f;
    SFERES_CONST float eta_c = 15.0f;
  };
};


BOOST_AUTO_TEST_CASE(gaussian_recomb) {
  test<10, Params2>();
}

struct Params3 {
  struct evo_float {
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.1f;
    SFERES_CONST mutation_t mutation_type = uniform;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float max = 0.3f;
    SFERES_CONST float eta_c = 15.0f;
  };
};

BOOST_AUTO_TEST_CASE(uniform_sbx) {
  test<10, Params3>();
}

