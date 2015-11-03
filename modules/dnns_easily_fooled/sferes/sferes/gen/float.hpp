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

#ifndef FLOAT_HPP_
#define FLOAT_HPP_

#include <vector>
#include <limits>
#include <boost/foreach.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stc.hpp>
#include <sferes/misc.hpp>
#include <sferes/dbg/dbg.hpp>
#include <iostream>
#include <cmath>


namespace sferes {
  namespace gen {
    // A basic class that represent an array of float, typically in range [0;1]
    // it is used by CMAES and EvoFloat derives from this class
    template<int Size, typename Params, typename Exact = stc::Itself>
    class Float : public stc::Any<Exact> {
     public:
      typedef Params params_t;
      typedef Float<Size, Params, Exact> this_t;
      SFERES_CONST size_t gen_size = Size;

      Float() : _data(Size) {
        std::fill(_data.begin(), _data.end(), 0.5f);
      }

      //@{
      void mutate() {
        assert(0);//should not be used (use evo_float)
      }
      void cross(const Float& o, Float& c1, Float& c2) {
        assert(0); // should not be used (use evo_float)
      }
      void random() {
        assert(0); // should not be used (use evo_float)
      }
      //@}

      //@{
      float data(size_t i) const {
        assert(this->_data.size());
        assert(i < this->_data.size());
        assert(!std::isinf(this->_data[i]));
        assert(!std::isnan(this->_data[i]));
        return this->_data[i];
      }
      void  data(size_t i, float v) {
        assert(this->_data.size());
        assert(i < this->_data.size());
        assert(!std::isinf(v));
        assert(!std::isnan(v));
        this->_data[i] = v;
      }
      size_t size() const {
        return Size;
      }
      //@}
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(_data);
      }
     protected:
      std::vector<float> _data;
    };
  } // gen
} // sferes


#endif
