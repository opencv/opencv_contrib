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




#ifndef EA_HPP_
#define EA_HPP_

#include <iostream>
#include <vector>
#include <fstream>

#include <boost/fusion/container.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <sferes/dbg/dbg.hpp>
#include <sferes/misc.hpp>
#include <sferes/stc.hpp>

namespace sferes {
  namespace ea {

    template<typename E>
    struct RefreshStat_f {
      RefreshStat_f(const E &ea) : _ea(ea) {
      }
      const E& _ea;
      template<typename T>
      void operator() (T & x) const {
        x.refresh(_ea);
      }
    };
    template<typename A>
    struct WriteStat_f {
      WriteStat_f(A & a) : _archive(a) {
      }
      A& _archive;
      template<typename T>
      void operator() (const T &x) const {
        std::string version(VERSION);
        _archive << boost::serialization::make_nvp("version",
                 version);
        _archive << BOOST_SERIALIZATION_NVP(x);
      }
    };

    template<typename A>
    struct ReadStat_f {
      ReadStat_f(A & a) : _archive(a) {
      }
      A& _archive;
      template<typename T>
      void operator() (T & x) const {
        std::string version;
        _archive >> boost::serialization::make_nvp("version", version);
        if (version != std::string(VERSION))
          std::cerr << "WARNING: your are loading a file made with sferes version "
                    << version << " while the current version is:"
                    << VERSION
                    << std::endl;
        _archive >> BOOST_SERIALIZATION_NVP(x);
      }
    };

    struct ShowStat_f {
      ShowStat_f(unsigned n, std::ostream & os, size_t k) :
        _n(n), _i(0), _os(os), _k(k) {
      }
      template<typename T>
      void operator() (T & x) const {
        if (_i == _n)
          x.show(_os, _k);

        ++_i;
      }
      int _n;
      mutable int _i;
      std::ostream& _os;
      size_t _k;
    };


    template<typename E>
    struct ApplyModifier_f {
      ApplyModifier_f(E &ea) : _ea(ea) {
      }
      E& _ea;
      template<typename T>
      void operator() (T & x) const {
        x.apply(_ea);
      }
    };

    template<typename Phen, typename Eval, typename Stat, typename FitModifier,
             typename Params,
             typename Exact = stc::Itself>
    class Ea : public stc::Any<Exact> {
     public:
      typedef Phen phen_t;
      typedef Eval eval_t;
      typedef Stat stat_t;
      typedef Params params_t;
      typedef typename
      boost::mpl::if_<boost::fusion::traits::is_sequence<FitModifier>,
            FitModifier,
            boost::fusion::vector<FitModifier> >::type modifier_t;
      typedef std::vector<boost::shared_ptr<Phen> > pop_t;

      Ea() : _pop(Params::pop::size), _gen(0) {
        _make_res_dir();
      }

      void run() {
        dbg::trace trace("ea", DBG_HERE);
        random_pop();
        for (_gen = 0; _gen < Params::pop::nb_gen; ++_gen) {
          epoch();
          update_stats();
          if (_gen % Params::pop::dump_period == 0)
            _write(_gen);
        }
      }
      void random_pop() {
        dbg::trace trace("ea", DBG_HERE);
        stc::exact(this)->random_pop();
      }
      void epoch() {
        dbg::trace trace("ea", DBG_HERE);
        stc::exact(this)->epoch();
      }
      const pop_t& pop() const {
        return _pop;
      };
      pop_t& pop() {
        return _pop;
      };
      const eval_t& eval() const {
        return _eval;
      }
      eval_t& eval() {
        return _eval;
      }
      const stat_t& stat() const {
        return _stat;
      }
      const modifier_t& fit_modifier() const {
        return _fit_modifier;
      }

      // modifiers
      void apply_modifier() {
        boost::fusion::for_each(_fit_modifier, ApplyModifier_f<Exact>(stc::exact(*this)));
      }

      // stats
      template<int I>
      const typename boost::fusion::result_of::value_at_c<Stat, I>::type& stat() const {
        return boost::fusion::at_c<I>(_stat);
      }
      void load(const std::string& fname) {
        _load(fname);
      }
      void show_stat(unsigned i, std::ostream& os, size_t k = 0) {
        boost::fusion::for_each(_stat, ShowStat_f(i, os, k));
      }
      void update_stats() {
        boost::fusion::for_each(_stat, RefreshStat_f<Exact>(stc::exact(*this)));
      }


      const std::string& res_dir() const {
        return _res_dir;
      }
      size_t gen() const {
        return _gen;
      }
      bool dump_enabled() const {
        return Params::pop::dump_period != -1;
      }
      void write() const {
        _write(gen());
      }
      void write(size_t g) const {
        _write(g);
      }
     protected:
      pop_t _pop;
      eval_t _eval;
      stat_t _stat;
      modifier_t _fit_modifier;
      std::string _res_dir;
      size_t _gen;

      void _make_res_dir() {
        if (Params::pop::dump_period == -1)
          return;

        _res_dir = misc::hostname() + "_" + misc::date() + "_" + misc::getpid();
        boost::filesystem::path my_path(_res_dir);
        boost::filesystem::create_directory(my_path);
      }
      void _write(int gen) const {
        dbg::trace trace("ea", DBG_HERE);
        if (Params::pop::dump_period == -1)
          return;
        std::string fname = _res_dir + std::string("/gen_")
                            + boost::lexical_cast<std::string>(gen);
        std::ofstream ofs(fname.c_str());
        typedef boost::archive::xml_oarchive oa_t;
        oa_t oa(ofs);
        boost::fusion::for_each(_stat, WriteStat_f<oa_t>(oa));
        std::cout << fname << " written" << std::endl;
      }
      void _load(const std::string& fname) {
        dbg::trace trace("ea", DBG_HERE);
        std::cout << "loading " << fname << std::endl;
        std::ifstream ifs(fname.c_str());
        if (ifs.fail()) {
          std::cerr << "Cannot open :" << fname
                    << "(does file exist ?)" << std::endl;
          exit(1);
        }
        typedef boost::archive::xml_iarchive ia_t;
        ia_t ia(ifs);
        boost::fusion::for_each(_stat, ReadStat_f<ia_t>(ia));
      }
    };
  }
}

#define SFERES_EA(Class, Parent)                                                               \
  template<typename Phen, typename Eval, typename Stat, typename FitModifier, typename Params, \
           typename Exact = stc::Itself>                                                       \
  class Class : public Parent < Phen, Eval, Stat, FitModifier, Params,                         \
  typename stc::FindExact<Class<Phen, Eval, Stat, FitModifier, Params, Exact>, Exact>::ret >

#endif
