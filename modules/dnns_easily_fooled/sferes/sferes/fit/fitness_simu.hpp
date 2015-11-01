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




#ifndef FITNESS_SIMU_HPP
#define FITNESS_SIMU_HPP

#include <sferes/stc.hpp>
#include <sferes/fit/fitness.hpp>
#include <iostream>
#include <boost/foreach.hpp>

#define SFERES_FITNESS_SIMU(Class, Parent)					\
  template <typename Simu, typename Agent, typename Params, typename Exact = stc::Itself> \
  class Class : public Parent<Simu, Agent, Params, typename stc::FindExact<Class<Simu, Agent, Params, Exact>, Exact>::ret>



namespace sferes {
  namespace fit {
    namespace state {
      enum state_t { not_started, running, end_exp, end_eval, fast_fw, eval_done };
    }

    template<typename Simu, typename Agent, typename Params, typename Exact = stc::Itself>
    class FitnessSimu : public stc::Any<Exact> {
     public:
      typedef Simu simu_t;
      FitnessSimu() :
        _step(0),
        _exp_step(0),
        _nb_exps(0),
        _value(0.0f),
        _state(state::not_started),
        _mode(mode::eval) {
      }

      template<typename Phen>
      void eval(Phen& p) {

        dbg::out(dbg::info, "fit")<<"eval mode="<<this->mode()<<" (mode view="<<mode::view<<")"<<std::endl;

        if (this->mode() == mode::view)
          _simu.init_view();
        if (_state == state::eval_done)
          return;
        _state = state::not_started;
        while (_state != state::end_eval)
          _exp(p);
        _state = state::eval_done;
      }
      template<typename Phen>
      int refresh(Phen& p) {
        return stc::exact(this)->refresh(p);
      }
      template<typename Phen>
      void refresh_end_exp(Phen& p) {
        dbg::trace t1("fit", DBG_HERE);
        stc::exact(this)->refresh_end_exp(p);
      }
      template<typename Phen>
      void refresh_end_eval(Phen& p) {
        dbg::trace t1("fit", DBG_HERE);
        stc::exact(this)->refresh_end_eval(p);
      }
      template<typename Phen>
      void scheduler(Phen& p) {
        dbg::trace t1("fit", DBG_HERE);
        stc::exact(this)->scheduler(p);
      }

      template<typename Phen>
      void new_exp(Phen& p) {
        dbg::out(dbg::info, "fit")<<"new_exp, _step="<<_step<<std::endl;
        _state = state::running;
        _exp_step = 0;
      }
      template<typename Phen>
      void end_exp(Phen& p) {
        dbg::out(dbg::info, "fit")<<"end_exp, _step="<<_step<<std::endl;
        assert(_state == state::running || _state == state::fast_fw);
        _state = state::end_exp;
        refresh_end_exp(p);
        _exp_step = 0;
        ++_nb_exps;
      }
      template<typename Phen>
      void end_eval(Phen& p) {
        assert(_state == state::end_exp);
        dbg::out(dbg::info, "fit")<<"end_eval, _step="<<_step<<std::endl;
        _state = state::end_eval;
        refresh_end_eval(p);
        if (!_objs.empty()) {
          BOOST_FOREACH(float v, _objs)
          _value += v;
          _value /= _objs.size();
        }
      }

      size_t step() const {
        return _step;
      }
      size_t exp_step() const {
        return _exp_step;
      }
      // number of finished experiments
      size_t nb_exps() const {
        return _nb_exps;
      }

      const Simu& simu() const {
        return _simu;
      }
      Simu& simu() {
        return _simu;
      }

      const Agent& agent() const {
        return _agent;
      }
      Agent& agent() {
        return _agent;
      }

      float value() const {
        return _value;
      }
      const std::vector<float>& objs() const {
        assert(!_objs.empty());
        return _objs;
      }
      float obj(size_t i) const {
        assert(i < _objs.size());
        return _objs[i];
      }

      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
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
      Simu _simu;
      Agent _agent;
      int _step;
      size_t _exp_step;
      size_t _nb_exps;
      std::vector<float> _objs;
      float _value;
      state::state_t _state;
      mode::mode_t _mode;

      template<typename Phen>
      void _exp(Phen& p) {
        dbg::out(dbg::tracing, "fit")<<"starting _step = "
                                     <<_step<<" state="<<_state<<std::endl;
        _simu.init();
        _agent.init(p);
        //	_agent.refresh_params(p);
        _state = state::not_started;
        while(_state != state::end_exp
              && _state != state::end_eval) {
          scheduler(p);
          _simu.refresh();
          if (mode() == mode::view)
            _simu.refresh_view();
          _agent.refresh(_simu, p);// call refresh param only the
          // first time
          int res = refresh(p);
          ++_step;
          ++_exp_step;
          if (res == -1)
            _goto_next_exp(p);
        }
      }
      template<typename Phen>
      void _goto_next_exp(Phen& p) {
        dbg::trace t1("fit", DBG_HERE);
        dbg::out(dbg::tracing, "fit")<<"exp stopped, _step = "<<_step<<" state="<<_state<<std::endl;
        while(_state != state::end_exp && _state != state::end_eval) {
          _state = state::fast_fw;
          scheduler(p);
          ++_step;
        }
      }


    };

#define AT(K) if (this->_state == ::sferes::fit::state::running	\
		  && this->_step == K)

#define EVERY(K, E) assert(K != 0);				\
    if (this->_state == ::sferes::fit::state::running		\
	&& ((this->_step + K) % E) == 0)

#define SFERES_SCHEDULER()			\
    template<typename Phen>			\
    void scheduler(Phen& p)


#define NEW_EXP(K) { if (this->_step == K) this->new_exp(p); }
#define END_EXP(K) { if (this->_step == K) this->end_exp(p); }
#define END_EVAL(K) { if (this->_step == K) this->end_eval(p); }

    SFERES_FITNESS_SIMU(FitnessSimuDummy, FitnessSimu) {
    public:
      template<typename Phen>
      int refresh(Phen& p) {
        return 0;
      }
      template<typename Phen>
      void refresh_end_exp(Phen& p) { }
      template<typename Phen>
      void refresh_end_eval(Phen& p) {}
      SFERES_SCHEDULER() {
        NEW_EXP(0);
        END_EXP(1);
        END_EVAL(1);
      }
    };

  }
};



#endif
