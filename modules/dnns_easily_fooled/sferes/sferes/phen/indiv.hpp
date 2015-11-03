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




#ifndef INDIV_HPP_
#define INDIV_HPP_
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stc.hpp>
#include <sferes/dbg/dbg.hpp>

#define SFERES_INDIV(Class, Parent)					\
  template <typename Gen, typename Fit, typename Params, typename Exact = stc::Itself> \
  class Class : public Parent<Gen, Fit, Params, typename stc::FindExact<Class<Gen, Fit, Params, Exact>, Exact>::ret>


namespace sferes {
  namespace phen {
    template<typename Gen, typename Fit, typename Params, typename Exact = stc::Itself>
    class Indiv {
     public:
      typedef Fit fit_t;
      typedef Gen gen_t;

      Fit& fit() {
        return _fit;
      }
      const Fit& fit() const {
        return _fit;
      }

      Gen& gen()  {
        return _gen;
      }
      const Gen& gen() const {
        return _gen;
      }
      void mutate() {
        dbg::trace trace("phen", DBG_HERE);
        this->_gen.mutate();
      }
      void cross(const boost::shared_ptr<Exact> i2,
                 boost::shared_ptr<Exact>& o1,
                 boost::shared_ptr<Exact>& o2) {
        dbg::trace trace("phen", DBG_HERE);
        if (!o1)
          o1 = boost::shared_ptr<Exact>(new Exact());
        if (!o2)
          o2 = boost::shared_ptr<Exact>(new Exact());
        _gen.cross(i2->gen(), o1->gen(), o2->gen());
      }
      void random() {
        dbg::trace trace("phen", DBG_HERE);
        this->_gen.random();
      }
      void develop() {
        dbg::trace trace("phen", DBG_HERE);
        stc::exact(this)->develop();
      }

      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        dbg::trace trace("phen", DBG_HERE);
        ar & BOOST_SERIALIZATION_NVP(_gen);
        ar & BOOST_SERIALIZATION_NVP(_fit);
      }
      void show(std::ostream& os) {
        os<<"nothing to show in a basic individual"<<std::endl;
      }
     protected:
      Gen _gen;
      Fit _fit;
    };

    SFERES_INDIV(Dummy, Indiv) {
    public:
      void develop() {}
    };

  }
}


#endif
