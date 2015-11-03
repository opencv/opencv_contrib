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




#ifndef TEST_SERIALIZE_HPP
#define TEST_SERIALIZE_HPP

#include <fstream>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/tmpdir.hpp>

// a generic serialize test to be used in test suites
namespace sferes {
  namespace tests {

    // pass this if you don't want to write the equality test
    struct check_nothing {
      template<typename T>
      void operator()(const T& x1, const T& x2) const {
      }
    };


    template<typename T, typename CheckEqual, typename Ia, typename Oa>
    void check_serialize(const T& src, T& dest, const CheckEqual &check_equal) {
      BOOST_CHECK(true);
      std::string filename = boost::archive::tmpdir();
      filename += "/serialize_g.xml";

      BOOST_CHECK(true);
      {
        std::ofstream ofs(filename.c_str());
        std::cout<<filename.c_str()<<std::endl;
        Oa oa(ofs);
        BOOST_CHECK(true);
        oa << boost::serialization::make_nvp("gen", src);
        BOOST_CHECK(true);
      }
      {
        BOOST_CHECK(true);
        std::ifstream ifs(filename.c_str());
        BOOST_CHECK(true);
        Ia ia(ifs);
        BOOST_CHECK(true);
        ia >> boost::serialization::make_nvp("gen", dest);
        BOOST_CHECK(true);
      }
      check_equal(src, dest);
    }
    template<typename T, typename CheckEqual>
    void check_serialize(const T& src, T& dest, const CheckEqual &check_equal) {
      typedef boost::archive::xml_oarchive oa_xml_t;
      typedef boost::archive::xml_iarchive ia_xml_t;
      typedef boost::archive::text_oarchive oa_text_t;
      typedef boost::archive::text_iarchive ia_text_t;
      typedef boost::archive::binary_oarchive oa_bin_t;
      typedef boost::archive::binary_iarchive ia_bin_t;

      std::cout<<"XML archive"<<std::endl;
      check_serialize<T, CheckEqual, ia_xml_t, oa_xml_t>(src, dest, check_equal);
      std::cout<<"test archive"<<std::endl;
      check_serialize<T, CheckEqual, ia_text_t, oa_text_t>(src, dest, check_equal);
      std::cout<<"binary archive" <<std::endl;
      check_serialize<T, CheckEqual, ia_bin_t, oa_bin_t>(src, dest, check_equal);

    }
  }
}
#endif
