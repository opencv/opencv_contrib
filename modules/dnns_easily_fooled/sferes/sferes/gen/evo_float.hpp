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




#ifndef EVO_FLOAT_HPP_
#define EVO_FLOAT_HPP_

#include <vector>
#include <limits>
#include <boost/foreach.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stc.hpp>
#include <sferes/misc.hpp>
#include <sferes/dbg/dbg.hpp>
#include <sferes/gen/float.hpp>
#include <iostream>
#include <cmath>
namespace sferes {
  namespace gen {
    namespace evo_float {
      enum mutation_t { polynomial = 0, gaussian, uniform };
      enum cross_over_t { recombination = 0, sbx, no_cross_over };

      template<typename Ev, int T>
      struct Mutation_f {
        void operator()(Ev& ev, size_t i) {
          assert(0);
        }
      };
      template<typename Ev, int T>
      struct CrossOver_f {
        void operator()(const Ev& f1, const Ev& f2, Ev &c1, Ev &c2) {
          assert(0);
        }
      };
    }

    /// in range [0;1]
    template<int Size, typename Params, typename Exact = stc::Itself>
    class EvoFloat :
      public Float<Size, Params,
      typename stc::FindExact<Float<Size, Params, Exact>, Exact>::ret> {
     public:
      typedef Params params_t;
      typedef EvoFloat<Size, Params, Exact> this_t;

      EvoFloat() {}

      //@{
      void mutate() {
        for (size_t i = 0; i < Size; i++)
          if (misc::rand<float>() < Params::evo_float::mutation_rate)
            _mutation_op(*this, i);
        _check_invariant();
      }
      void cross(const EvoFloat& o, EvoFloat& c1, EvoFloat& c2) {
        if (Params::evo_float::cross_over_type != evo_float::no_cross_over &&
            misc::rand<float>() < Params::evo_float::cross_rate)
          _cross_over_op(*this, o, c1, c2);
        else if (misc::flip_coin()) {
          c1 = *this;
          c2 = o;
        } else {
          c1 = o;
          c2 = *this;
        }
        _check_invariant();
      }
      void random() {
        BOOST_FOREACH(float &v, this->_data) v = misc::rand<float>();
        _check_invariant();
      }
      //@}

     protected:
      evo_float::Mutation_f<this_t, Params::evo_float::mutation_type> _mutation_op;
      evo_float::CrossOver_f<this_t, Params::evo_float::cross_over_type> _cross_over_op;
      void _check_invariant() const {
#ifdef DBG_ENABLED
        BOOST_FOREACH(float p, this->_data) {
          assert(!std::isnan(p));
          assert(!std::isinf(p));
          assert(p >= 0 && p <= 1);
        }
#endif
      }
    };

    // partial specialization for operators
    namespace evo_float {
      // polynomial mutation. Cf Deb 2001, p 124 ; param: eta_m
      // perturbation of the order O(1/eta_m)
      template<typename Ev>
      struct Mutation_f<Ev, polynomial> {
        void operator()(Ev& ev, size_t i) {
          SFERES_CONST float eta_m = Ev::params_t::evo_float::eta_m;
          assert(eta_m != -1.0f);
          float ri = misc::rand<float>();
          float delta_i = ri < 0.5 ?
                          pow(2.0 * ri, 1.0 / (eta_m + 1.0)) - 1.0 :
                          1 - pow(2.0 * (1.0 - ri), 1.0 / (eta_m + 1.0));
          assert(!std::isnan(delta_i));
          assert(!std::isinf(delta_i));
          float f = ev.data(i) + delta_i;
          ev.data(i, misc::put_in_range(f, 0.0f, 1.0f));
        }
      };

      // gaussian mutation
      template<typename Ev>
      struct Mutation_f<Ev, gaussian> {
        void operator()(Ev& ev, size_t i) {
          SFERES_CONST float sigma = Ev::params_t::evo_float::sigma;
          float f = ev.data(i)
                    + misc::gaussian_rand<float>(0, sigma * sigma);
          ev.data(i, misc::put_in_range(f, 0.0f, 1.0f));
        }
      };
      // uniform mutation
      template<typename Ev>
      struct Mutation_f<Ev, uniform> {
        void operator()(Ev& ev, size_t i) {
          SFERES_CONST float max = Ev::params_t::evo_float::max;
          float f = ev.data(i)
                    + misc::rand<float>(max) - max / 2.0f;
          ev.data(i, misc::put_in_range(f, 0.0f, 1.0f));
        }
      };

      // recombination
      template<typename Ev>
      struct CrossOver_f<Ev, recombination> {
        void operator()(const Ev& f1, const Ev& f2, Ev &c1, Ev &c2) {
          size_t k = misc::rand<unsigned int>(f1.size());
          for (size_t i = 0; i < k; ++i) {
            c1.data(i, f1.data(i));
            c2.data(i, f2.data(i));
          }
          for (size_t i = k; i < f1.size(); ++i) {
            c1.data(i, f2.data(i));
            c2.data(i, f1.data(i));
          }
        }
      };

      // no cross-over
      template<typename Ev>
      struct CrossOver_f<Ev, no_cross_over> {
        void operator()(const Ev& f1, const Ev& f2, Ev &c1, Ev &c2) {
        }
      };

      // SBX (cf Deb 2001, p 113) Simulated Binary Crossover
      // suggested eta : 15
      /// WARNING : this code is from deb's code (different from the
      // article ...)
      // A large value ef eta gives a higher probablitity for
      // creating a `near-parent' solutions and a small value allows
      // distant solutions to be selected as offspring.
      template<typename Ev>
      struct CrossOver_f<Ev, sbx> {
        void operator()(const Ev& f1, const Ev& f2, Ev &child1, Ev &child2) {
          SFERES_CONST float eta_c = Ev::params_t::evo_float::eta_c;
          assert(eta_c != -1);
          for (unsigned int i = 0; i < f1.size(); i++) {
            float y1 = std::min(f1.data(i), f2.data(i));
            float y2 = std::max(f1.data(i), f2.data(i));
            SFERES_CONST float yl = 0.0;
            SFERES_CONST float yu = 1.0;
            if (fabs(y1 - y2) > std::numeric_limits<float>::epsilon()) {
              float rand = misc::rand<float>();
              float beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1));
              float alpha = 2.0 - pow(beta, -(eta_c + 1.0));
              float betaq = 0;
              if (rand <= (1.0 / alpha))
                betaq = pow((rand * alpha), (1.0 / (eta_c + 1.0)));
              else
                betaq = pow ((1.0 / (2.0 - rand * alpha)) , (1.0 / (eta_c + 1.0)));
              float c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
              beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
              alpha = 2.0 - pow(beta, -(eta_c + 1.0));
              if (rand <= (1.0 / alpha))
                betaq = pow ((rand * alpha), (1.0 / (eta_c + 1.0)));
              else
                betaq = pow ((1.0/(2.0 - rand * alpha)), (1.0 / (eta_c + 1.0)));
              float c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

              c1 = misc::put_in_range(c1, yl, yu);
              c2 = misc::put_in_range(c2, yl, yu);

              assert(!std::isnan(c1));
              assert(!std::isnan(c2));

              if (misc::flip_coin()) {
                child1.data(i, c1);
                child2.data(i, c2);
              } else {
                child1.data(i, c2);
                child2.data(i, c1);
              }
            }
          }
        }
      };

    } //evo_float
  } // gen
} // sferes


#endif
