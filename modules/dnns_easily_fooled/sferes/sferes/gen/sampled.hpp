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


#ifndef GEN_SAMPLED_HPP_
#define GEN_SAMPLED_HPP_

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


namespace sferes {
  namespace gen {
    template<int Size, typename Params, typename Exact = stc::Itself>
    class Sampled : public stc::Any<Exact> {
     public:
      typedef Params params_t;
      typedef Sampled<Size, Params, Exact> this_t;
      typedef typename Params::sampled::values_t values_t;
      Sampled() : _data(Size) {
      }

      //@{
      void mutate() {
        if (Params::sampled::ordered) {
          for (size_t i = 0; i < _data.size(); ++i)
            if (misc::rand<float>() < Params::sampled::mutation_rate) {
              if (misc::flip_coin())
                _data[i] = std::max(0, (int)_data[i] - 1);
              else
                _data[i] = std::min((int)Params::sampled::values_size() - 1,
                                    (int)_data[i] + 1);
            }
        } else {
          BOOST_FOREACH(size_t & v, _data)
          if (misc::rand<float>() < Params::sampled::mutation_rate)
            v = misc::rand<size_t>(0, Params::sampled::values_size());
          _check_invariant();
        }
        _check_invariant();
      }

      // 1-point cross-over
      void cross(const Sampled& o, Sampled& c1, Sampled& c2) {
        assert(c1._data.size());
        assert(c1._data.size() == c2._data.size());
        if (misc::rand<float>() < Params::sampled::cross_rate) {
          size_t k = misc::rand(c1._data.size());
          for (size_t j = 0; j < c1._data.size(); ++j)
            if (j < k) {
              c1._data[j] = _data[j];
              c2._data[j] = o._data[j];
            } else {
              c1._data[j] = o._data[j];
              c2._data[j] = _data[j];
            }
        } else {
          c1 = *this;
          c2 = o;
        }
        c1._check_invariant();
        c2._check_invariant();
      }
      void random() {
        BOOST_FOREACH(size_t & v, _data)
        v = misc::rand<size_t>(0, Params::sampled::values_size());
        _check_invariant();
      }
      //@}

      //@{
      values_t data(size_t i) const {
        assert(i < _data.size());
        assert(i >= 0);
        _check_invariant();
        return Params::sampled::values(_data[i]);
      }
      size_t data_index(size_t i) const {
        assert(i < _data.size());
        assert(i >= 0);
        _check_invariant();
        return _data[i];
      }
      void set_data(size_t pos, size_t k) {
        assert(pos < _data.size());
        _data[pos] = k;
        _check_invariant();
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
      void _check_invariant() const {
#ifndef NDEBUG
        for (size_t i = 0; i < _data.size(); ++i) {
          assert(_data[i] >= 0);
          assert(_data[i] < Params::sampled::values_size());
        }
#endif
      }
      std::vector<size_t> _data;
    };

  } // gen
} // sferes


#endif
