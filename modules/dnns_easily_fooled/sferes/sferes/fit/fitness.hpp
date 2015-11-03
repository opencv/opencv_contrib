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




#ifndef FITNESS_HPP_
#define FITNESS_HPP_

#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <boost/serialization/nvp.hpp>
#include <boost/shared_ptr.hpp>
#include <sferes/stc.hpp>
#include <sferes/dbg/dbg.hpp>


#define SFERES_FITNESS SFERES_CLASS_D

namespace sferes {
  namespace fit {

    namespace mode {
      enum mode_t { eval = 0, view, usr1, usr2, usr3, usr4, usr5 };
    }
    SFERES_CLASS(Fitness) {
    public:
      Fitness() : _value(0), _mode(mode::eval) {}
      float value() const {
        return _value;
      }
      const std::vector<float>& objs() const {
        return _objs;
      }
      void add_obj() {
        _objs.resize(_objs.size() + 1);
      }
      float obj(size_t i) const {
        assert(i < _objs.size());
        assert(!std::isnan(_objs[i]));
        return _objs[i];
      }
      void set_obj(size_t i, float v) {
        assert(i < _objs.size());
        assert(!std::isnan(v));
        _objs[i] = v;
      }
      template<typename Indiv>
      void eval(Indiv& i) {
        stc::exact(this)->eval(i);
      }
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        dbg::trace trace("fit", DBG_HERE);
        ar & BOOST_SERIALIZATION_NVP(_value);
        ar & BOOST_SERIALIZATION_NVP(_objs);
      }
      void set_mode(mode::mode_t m) {
        _mode = m;
      }
      mode::mode_t mode() const {
        return _mode;
      }
    protected:
      float _value;
      std::vector<float> _objs;
      mode::mode_t _mode;
    };

    struct compare {
      template<typename I>
      bool operator()(const boost::shared_ptr<I> i1, const boost::shared_ptr<I> i2) const {
        return i1->fit().value() > i2->fit().value();
      }
    };

    struct compare_obj {
      compare_obj(unsigned i) : _i(i) {}
      template<typename I>
      bool operator()(const boost::shared_ptr<I> i1, const boost::shared_ptr<I> i2) const {
        assert(_i < i1->fit().objs().size());
        assert(_i < i2->fit().objs().size());
        assert(i1->fit().objs().size());
        assert(i2->fit().objs().size());
        return i1->fit().obj(_i) > i2->fit().obj(_i);
      }
      size_t _i;
    };
    // lexical order
    struct compare_objs_lex {
      compare_objs_lex() {}
      template<typename I>
      bool operator()(const boost::shared_ptr<I> i1, const boost::shared_ptr<I> i2) const {
        assert(i1->fit().objs().size() == i2->fit().objs().size());
        assert(i1->fit().objs().size());
        assert(i2->fit().objs().size());
        for (size_t i = 0; i < i1->fit().objs().size(); ++i)
          if (i1->fit().obj(i) > i2->fit().obj(i))
            return true;
          else if (i1->fit().obj(i) < i2->fit().obj(i))
            return false;
        return false;
      }

    };

    // returns :
    //  1 if i1 dominates i2
    // -1 if i2 dominates i1
    // 0 if both a and b are non-dominated
    template<typename I1, typename I2>
    inline int dominate_flag(const boost::shared_ptr<I1> i1, const boost::shared_ptr<I2> i2) {
      assert(i1->fit().objs().size());
      assert(i2->fit().objs().size());
      if (i1->fit().objs().size() != i2->fit().objs().size())
        std::cout<<i1->fit().objs().size()<<" vs "<<i2->fit().objs().size()<<std::endl;
      assert(i1->fit().objs().size() == i2->fit().objs().size());

      size_t nb_objs = i1->fit().objs().size();
      assert(nb_objs);

      bool flag1 = false, flag2 = false;
      for (unsigned i = 0; i < nb_objs; ++i) {
        float fi1 = i1->fit().obj(i);
        float fi2 = i2->fit().obj(i);
        if (fi1 > fi2)
          flag1 = true;
        else if (fi2 > fi1)
          flag2 = true;
      }
      if (flag1 && !flag2)
        return 1;
      else if (!flag1 && flag2)
        return -1;
      else
        return 0;
    }

    // true if i1 dominate i2
    template<typename I1, typename I2>
    inline bool dominate(const boost::shared_ptr<I1> i1, const boost::shared_ptr<I2> i2) {
      return (dominate_flag(i1, i2) == 1);
    }


    struct compare_pareto {
      template<typename I1, typename I2>
      int operator()(const boost::shared_ptr<I1> i1, const boost::shared_ptr<I2> i2) const {
        return dominate_flag(i1, i2);
      }
      template<typename I>
      bool eq(const boost::shared_ptr<I> i1, const boost::shared_ptr<I> i2) {
        bool c1 = operator()(i1, i2);
        bool c2 = operator()(i2, i1);
        return !c1 && !c2;
      }
    };

    SFERES_CLASS_D(FitDummy, Fitness) {
    public:
      template<typename Indiv>
      void eval(Indiv& i) {
      }
    };
  }
}



#endif
