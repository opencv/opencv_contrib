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




#ifndef BITSTRING_HPP_
#define BITSTRING_HPP_

#include <vector>
#include <limits>
#include <bitset>
#include <boost/foreach.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stc.hpp>
#include <sferes/misc.hpp>
#include <sferes/dbg/dbg.hpp>
#include <iostream>

namespace boost {
  namespace serialization {
    template<class Archive, size_t Nb>
    void save(Archive& ar, const std::bitset<Nb>& bs, const unsigned int version) {
      std::string s = bs.to_string();
      ar << BOOST_SERIALIZATION_NVP(s);
    }

    template<class Archive, size_t Nb>
    void load(Archive& ar, std::bitset<Nb>& bs, const unsigned int version) {
      std::string s;
      ar >> BOOST_SERIALIZATION_NVP(s);
      assert(s.size() == bs.size());
      bs = 0;
      for (size_t i = 0; i < Nb; ++i)
        bs[Nb - i - 1] = (s[i] == '1');
    }
    template<class Archive, size_t Nb>
    void serialize(Archive& ar, std::bitset<Nb>& bs, const unsigned int version) {
      boost::serialization::split_free(ar, bs, version);
    }
  }
}

namespace sferes {
  namespace gen {
    namespace _bitstring {
      template<long K, long N>
      struct _pow {
        SFERES_CONST double result = K * _pow<K, N - 1>::result;
      };
      template<long K>
      struct _pow<K, 1> {
        SFERES_CONST double result = K;
      };

    }
    /// in range [0;1]
    template<int Size, typename Params, typename Exact = stc::Itself>
    class BitString : public stc::Any<Exact> {
     public:
      typedef Params params_t;
      typedef BitString<Size, Params, Exact> this_t;
      typedef std::bitset<Params::bit_string::nb_bits> bs_t;
      SFERES_CONST double bs_max = _bitstring::_pow<2, Params::bit_string::nb_bits>::result - 1;

      BitString() : _data(Size) {
      }

      //@{
      void mutate() {
        BOOST_FOREACH(bs_t & b, _data)
        if (misc::rand<float>() < Params::bit_string::mutation_rate)
          for (size_t i = 0; i < b.size(); ++i)
            if (misc::rand<float>() < Params::bit_string::mutation_rate_bit)
              b[i].flip();
      }
      // 1-point cross-over
      void cross(const BitString& o, BitString& c1, BitString& c2) {
        assert(Size == _data.size());
        assert(c1._data.size() == _data.size());
        assert(c2._data.size() == _data.size());

        for (size_t i = 0; i < c1._data.size(); ++i) {
          size_t k = misc::rand(c1._data.size());
          for (size_t j = 0; j < c1._data[i].size(); ++j)
            if (j < k) {
              c1._data[i][j] = _data[i][j];
              c2._data[i][j] = o._data[i][j];
            } else {
              c1._data[i][j] = o._data[i][j];
              c2._data[i][j] = _data[i][j];
            }
        }
      }
      void random() {
        BOOST_FOREACH(bs_t & v, _data)
        for (size_t i = 0; i < v.size(); ++i)
          v[i] = (int) misc::flip_coin();
      }
      //@}

      //@{
      float data(size_t i) const {
        assert(bs_max != 0);
        assert(i < _data.size());
        return _to_double(_data[i]) / bs_max;
      }
      unsigned long int_data(size_t i) const {
        assert(i < _data.size());
        return _data[i].to_ulong();
      }

      bs_t bs_data(size_t i) const {
        assert(i < _data.size());
        return _data[i];
      }

      size_t size() const {
        return Size;
      }
      //@}

      template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar& BOOST_SERIALIZATION_NVP(_data);
      }
     protected:
      template<size_t N>
      double _to_double(const std::bitset<N>& d) const {
        double x = 0;
        size_t k = 1;
        for (size_t i = 0; i < N; ++i) {
          x += d[i] * k;
          k *= 2;
        }
        return x;
      }
      std::vector<bs_t> _data;
    };

  } // gen
} // sferes


#endif
