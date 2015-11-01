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
#define BOOST_TEST_MODULE bit_string
#include <boost/test/unit_test.hpp>
#include <sferes/gen/bit_string.hpp>
#include <tests/check_serialize.hpp>
using namespace sferes::gen;


template<int Size, typename P>
void test() {
  for (unsigned i = 0; i < 30; ++i) {
    BitString<Size, P> gen1;
    BitString<Size, P> gen2;
    BitString<Size, P> gen3;
    BitString<Size, P> gen4;
    BOOST_CHECK(gen1.data(0) < 1.0f);
    BOOST_CHECK(gen1.data(0) >= 0.0f);
    BOOST_CHECK_CLOSE(gen1.data(0), gen1.int_data(0), 0.0001);
    BOOST_CHECK_CLOSE(gen2.data(0), gen2.int_data(0), 0.0001);
    gen1.random();
    gen2.random();
    gen1.mutate();
    gen1.cross(gen2, gen3, gen4);


  }
}

struct Params1 {
  struct bit_string {
    SFERES_CONST size_t nb_bits = 8;
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float mutation_rate_bit = 0.1f;
  };
};


struct Params2 {
  struct bit_string {
    SFERES_CONST size_t nb_bits = 50;
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float mutation_rate_bit = 0.1f;
  };
};



BOOST_AUTO_TEST_CASE(bitstring) {
  test<10, Params1>();

}

struct check_bitstring_eq {
  template<typename T>
  void operator()(const T& gen1, const T& gen2) const {
    BOOST_CHECK_EQUAL(gen1.size(), gen2.size());
    for (size_t i = 0; i < gen1.size(); ++i)
      BOOST_CHECK(fabs(gen1.data(i) - gen2.data(i) < 0.001));
  }
};

BOOST_AUTO_TEST_CASE(bitstring_serialize) {
  BitString<10, Params1> gen1, gen2;
  gen1.random();
  sferes::tests::check_serialize(gen1, gen2, check_bitstring_eq());
}




BOOST_AUTO_TEST_CASE(bitstring_serialize_long) {
  BitString<10, Params2> gen1, gen2;
  gen1.random();
  sferes::tests::check_serialize(gen1, gen2, check_bitstring_eq());
}


