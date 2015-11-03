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

#ifndef NN_NEURON_HPP
#define NN_NEURON_HPP

#include <boost/graph/properties.hpp>

#include "trait.hpp"

namespace nn
{

  // generic neuron
  // Pot : potential functor (see pf.hpp)
  // Act : activation functor (see af.hpp)
  // IO : type of coupling between "neurons" (float or std::pair<float, float>)
  template<typename Pot, typename Act, typename IO = float>
  class Neuron
  {
  public:
    typedef typename Pot::weight_t weight_t;
    typedef IO io_t;
    typedef Pot pf_t;
    typedef Act af_t;
    static io_t zero() { return trait<IO>::zero(); }
    Neuron() : 
      _current_output(zero()), 
      _next_output(zero()), 
      _fixed(false), 
      _in(-1), 
      _out(-1) 
    {}
    bool get_fixed() const { return _fixed; }
    void set_fixed(bool b = true) { _fixed = b; }
    io_t activate()
    {
      if (!_fixed)
	_next_output = _af(_pf(_inputs));
      return _next_output;
    }
 
    void init() 
    { 
      _pf.init(); 
      _af.init();
      if (get_in_degree() != 0)
	_inputs = trait<io_t>::zero(get_in_degree()); 
      _current_output = zero(); 
      _next_output = zero(); 
    }

    void set_input(unsigned i, const io_t& in) { assert(i < _inputs.size()); _inputs[i] = in; }

    void set_weight(unsigned i, const weight_t& w) { _pf.set_weight(i, w); }
   
    typename af_t::params_t& get_afparams() { return _af.get_params(); }
    typename pf_t::params_t& get_pfparams() { return _pf.get_params(); }
    const typename af_t::params_t& get_afparams() const { return _af.get_params(); }
    const typename pf_t::params_t& get_pfparams() const { return _pf.get_params(); }
    void set_afparams(const typename af_t::params_t& p) { _af.set_params(p); }
    void set_pfparams(const typename pf_t::params_t& p) { _pf.set_params(p); }

    void step() { _current_output = _next_output; }
    void set_in_degree(unsigned k)
    { 
      _pf.set_nb_weights(k); 
      _inputs.resize(k);
      if (k == 0)
	return;
      _inputs = trait<io_t>::zero(k); 
    }
    unsigned get_in_degree() const { return _pf.get_weights().size(); }

    // for input neurons
    void set_current_output(const io_t& v) { _current_output = v; }
    void set_next_output(const io_t& v) { _next_output = v; }
      
    // standard output
    const io_t& get_current_output() const { return _current_output; }

    // next output
    const io_t& get_next_output() const { return _next_output; }

    // i/o
    int get_in() const { return _in; }
    void set_in(int i) { _in = i; }
    int get_out() const { return _out; }
    void set_out(int o) { _out = o; }
    bool is_input() const { return _in != -1; }
    bool is_output() const { return _out != -1; }
    
    const Pot& get_pf() const { return _pf; }
    Pot& get_pf() { return _pf; }

    const Act& get_af() const { return _af; }
    Act& get_af() { return _af; }

    void set_id(const std::string& s) { _id = s; }
    const std::string& get_id() const { return _id; }
    const std::string& get_label() const { return _label; }
    
    // for graph algorithms
    std::string _id;
    std::string _label;
    boost::default_color_type _color;
    int _index;
  protected:
    // activation functor
    Act _af;
    // potential functor
    Pot _pf;
    // outputs
    io_t _current_output;
    io_t _next_output;
    // cache
    typename trait<io_t>::vector_t _inputs;
    // fixed = current_output is constant
    bool _fixed;
    // -1 if not an input of the nn, id of input otherwise
    int _in;
    // -1 if not an output of the nn, id of output otherwise
    int _out;
  };    
} 
#endif
