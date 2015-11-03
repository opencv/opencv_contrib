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




#ifndef AF_CPPN_HPP_
#define AF_CPPN_HPP_

#include <sferes/gen/sampled.hpp>
#include <sferes/gen/evo_float.hpp>

// classic activation functions
namespace nn
{
  namespace cppn
  {
    enum func_e { sine = 0, sigmoid, gaussian, linear, tanh };
    SFERES_CONST size_t nb_functions = 3;
    SFERES_CLASS(AfParams)
    {
      public:
        void set(float t, float p)
        {
          _type.set_data(0, t);
          _param.data(0, p);
        }
        void mutate()
        {
          _type.mutate();
          _param.mutate();
        }
        void random()
        {
          _type.random();
          _param.random();
        }
        void develop() {
        }
        int type() const {
          return _type.data(0);
        }
        float param() const {
          return _param.data(0);
        }
        template<typename A>
        void serialize(A& ar, unsigned int v)
        {
          ar& BOOST_SERIALIZATION_NVP(_type);
          ar& BOOST_SERIALIZATION_NVP(_param);
        }
      protected:
        sferes::gen::Sampled<1, Params> _type;
        sferes::gen::EvoFloat<1, Params> _param;
    };
  }

   // Activation function for Compositional Pattern Producing Networks
  template<typename P>
  struct AfCppn : public Af<P>
  {
    typedef P params_t;
    float operator() (float p) const
    {
      float s = p > 0 ? 1 : -1;
       //std::cout<<"type:"<<this->_params.type()<<" p:"<<p<<" this:"<<this
       //<< " out:"<< p * exp(-powf(p * 10/ this->_params.param(), 2))<<std::endl;
      switch (this->_params.type())
      {
      case cppn::sine:
        return sin(p);
      case cppn::sigmoid:
        return ((1.0 / (1.0 + exp(-p))) - 0.5) * 2.0;
      case cppn::gaussian:
        return exp(-powf(p, 2));
      case cppn::linear:
        return std::min(std::max(p, -3.0f), 3.0f) / 3.0f;
      case cppn::tanh:
        return tanh(p * 5.0f);
      default:
        assert(0);
      }
      return 0;
    }
  };


}

#endif
