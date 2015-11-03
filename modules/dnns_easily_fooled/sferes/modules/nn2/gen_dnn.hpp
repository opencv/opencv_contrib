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

#ifndef DNN_HPP_
#define DNN_HPP_

#include <bitset>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <sferes/dbg/dbg.hpp>
#include <sferes/misc.hpp>

#include "nn.hpp"
#include "trait.hpp"

namespace sferes
{
  namespace gen
  {

    template <class Graph>
    typename boost::graph_traits<Graph>::vertex_descriptor
    random_vertex(Graph& g)
    {
      assert(num_vertices(g));      
      using namespace boost;
      if (num_vertices(g) > 1)
        {
          std::size_t n = misc::rand(num_vertices(g));
          typename graph_traits<Graph>::vertex_iterator i = vertices(g).first;
          while (n-- > 0) ++i;
          return *i;
        } 
      else
	return *vertices(g).first;
    }

    template <class Graph>
    typename boost::graph_traits<Graph>::edge_descriptor
    random_edge(Graph& g)
    {
      assert(num_edges(g));
      using namespace boost;
      if (num_edges(g) > 1)
        {
          std::size_t n = misc::rand(num_edges(g));
          typename graph_traits<Graph>::edge_iterator i = edges(g).first;
          while (n-- > 0) ++i;
          return *i;
        } 
      else
	return *edges(g).first;
    }

  namespace dnn
  {
    enum init_t { ff = 0, random_topology };
  }
  template<typename N, typename C, typename Params>
  class Dnn : public nn::NN<N, C>
  {
  public:
    typedef nn::NN<N, C> nn_t;
    typedef N neuron_t;
    typedef C conn_t;
    typedef typename nn_t::io_t io_t;
    typedef typename nn_t::weight_t weight_t;
    typedef typename nn_t::vertex_desc_t vertex_desc_t;
    typedef typename nn_t::edge_desc_t edge_desc_t;
    typedef typename nn_t::adj_it_t adj_it_t;
    typedef typename nn_t::graph_t graph_t;
    void random()
    {
      if (Params::dnn::init == dnn::ff)
	_random_ff(Params::dnn::nb_inputs, Params::dnn::nb_outputs);
      else
	_random(Params::dnn::nb_inputs, Params::dnn::nb_outputs,
		Params::dnn::min_nb_neurons, Params::dnn::max_nb_neurons,
		Params::dnn::min_nb_conns, Params::dnn::max_nb_conns);
    }

    void mutate() 
    {
      _change_conns();

      _change_neurons();

      if (misc::rand<float>() < Params::dnn::m_rate_add_conn)
	_add_conn_nodup();

      if (misc::rand<float>() < Params::dnn::m_rate_del_conn)
	_del_conn();

      if (misc::rand<float>() < Params::dnn::m_rate_add_neuron)
	_add_neuron_on_conn();

      if (misc::rand<float>() < Params::dnn::m_rate_del_neuron)
	_del_neuron();

    }
    void cross(const Dnn& o, Dnn& c1, Dnn& c2) 
    {
#ifdef PHELOGENETIC_TREE
    	c1 = *this;
    	c2 = o;
#else
      if (misc::flip_coin())
			{
				c1 = *this;
				c2 = o;
			}
      else
			{
				c2 = *this;
				c1 = o;
			}
#endif
    }
    // serialize the graph "by hand"...
    template<typename Archive>
    void save(Archive& a, const unsigned v) const
    {
      dbg::trace("nn", DBG_HERE);
      std::vector<int> inputs;
      std::vector<int> outputs;
      std::vector<typename neuron_t::af_t::params_t> afparams;
      std::vector<typename neuron_t::pf_t::params_t> pfparams;
      std::map<vertex_desc_t, int> nmap;
      std::vector<std::pair<int, int> > conns;
      std::vector<weight_t> weights;

      BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
	{
	  if (this->is_input(v))
	    inputs.push_back(afparams.size());
	  if (this->is_output(v))
	    outputs.push_back(afparams.size());
	  nmap[v] = afparams.size();
	  afparams.push_back(this->_g[v].get_afparams());
	  pfparams.push_back(this->_g[v].get_pfparams());
	}
      BGL_FORALL_EDGES_T(e, this->_g, graph_t)
	{
	  conns.push_back(std::make_pair(nmap[source(e, this->_g)],
					 nmap[target(e, this->_g)]));
	  weights.push_back(this->_g[e].get_weight());
	}
      assert(pfparams.size() == afparams.size());
      assert(weights.size() == conns.size());

      a & BOOST_SERIALIZATION_NVP(afparams);	
      a & BOOST_SERIALIZATION_NVP(pfparams);	
      a & BOOST_SERIALIZATION_NVP(weights);
      a & BOOST_SERIALIZATION_NVP(conns);	
      a & BOOST_SERIALIZATION_NVP(inputs);	
      a & BOOST_SERIALIZATION_NVP(outputs);	
    }
    template<typename Archive>
    void load(Archive& a, const unsigned v)
    {
      dbg::trace("nn", DBG_HERE);
      std::vector<int> inputs;
      std::vector<int> outputs;
      std::vector<typename neuron_t::af_t::params_t> afparams;
      std::vector<typename neuron_t::pf_t::params_t> pfparams;
      std::map<size_t, vertex_desc_t> nmap;
      std::vector<std::pair<int, int> > conns;
      std::vector<weight_t> weights;

      a & BOOST_SERIALIZATION_NVP(afparams);
      a & BOOST_SERIALIZATION_NVP(pfparams);
      a & BOOST_SERIALIZATION_NVP(weights);
      a & BOOST_SERIALIZATION_NVP(conns);	
      a & BOOST_SERIALIZATION_NVP(inputs);	
      a & BOOST_SERIALIZATION_NVP(outputs);	

      assert(pfparams.size() == afparams.size());

      assert(weights.size() == conns.size());
      this->set_nb_inputs(inputs.size());
      this->set_nb_outputs(outputs.size());
      for (size_t i = 0; i < this->get_nb_inputs(); ++i)
	nmap[inputs[i]] = this->get_input(i);
      for (size_t i = 0; i < this->get_nb_outputs(); ++i)
	nmap[outputs[i]] = this->get_output(i);

      for (size_t i = 0; i < afparams.size(); ++i)
	if (std::find(inputs.begin(), inputs.end(), i) == inputs.end() 
	    && std::find(outputs.begin(), outputs.end(), i) == outputs.end())
	  nmap[i] = this->add_neuron("n", pfparams[i], afparams[i]);
	else
	  {
	    this->_g[nmap[i]].set_pfparams(pfparams[i]);
	    this->_g[nmap[i]].set_afparams(afparams[i]);
	  }


      //assert(nmap.size() == num_vertices(this->_g));
      for (size_t i = 0; i < conns.size(); ++i)
	this->add_connection(nmap[conns[i].first], nmap[conns[i].second], weights[i]);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER();

  protected:
    void _random_neuron_params()
    {
      BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
	{
	  this->_g[v].get_pfparams().random();
	  this->_g[v].get_afparams().random();
	}
    }
    // we start with a fully connected 0-layer perceptron with
    // random weights
    void _random_ff(size_t nb_inputs, size_t nb_outputs) 
    {
      this->set_nb_inputs(nb_inputs);
      this->set_nb_outputs(nb_outputs);
    
      BOOST_FOREACH(vertex_desc_t& i, this->_inputs)
	BOOST_FOREACH(vertex_desc_t& o, this->_outputs)
	this->add_connection(i, o, _random_weight());

      _random_neuron_params();
    }

    void _random(size_t nb_inputs, size_t nb_outputs,
		 size_t min_nb_neurons, size_t max_nb_neurons,
		 size_t min_nb_conns, size_t max_nb_conns) 
    {
      // io
      this->set_nb_inputs(nb_inputs);
      this->set_nb_outputs(nb_outputs);
      _random_neuron_params();

      // neurons
      size_t nb_neurons = misc::rand(min_nb_neurons, max_nb_neurons);
      for (size_t i = 0; i < nb_neurons; ++i)
	_add_neuron();//also call the random params 

      // conns
      size_t nb_conns = misc::rand(min_nb_conns, max_nb_conns);
      for (size_t i = 0; i < nb_conns; ++i)
	_add_conn_nodup();

      this->simplify();
    }

    vertex_desc_t _random_tgt()
    {
      vertex_desc_t v;
      do
	v = random_vertex(this->_g);
      while (this->is_input(v));
      return v;
    }
    vertex_desc_t _random_src()
    {
      vertex_desc_t v;
      do
	v = random_vertex(this->_g);
      while (this->is_output(v));
      return v;
    }

    vertex_desc_t _add_neuron() 
    { 
      vertex_desc_t v = this->add_neuron("n");
      this->_g[v].get_pfparams().random();
      this->_g[v].get_afparams().random();
      return v;
    }

    vertex_desc_t _add_neuron_on_conn()
    {
      if (!num_edges(this->_g))
	return (vertex_desc_t)0x0;	
      edge_desc_t e = random_edge(this->_g);
      vertex_desc_t src = source(e, this->_g);
      vertex_desc_t tgt = target(e, this->_g);
      typename nn_t::weight_t w = this->_g[e].get_weight();
      vertex_desc_t n = this->add_neuron("n");
      this->_g[n].get_pfparams().random();
      this->_g[n].get_afparams().random();
      // 
      remove_edge(e, this->_g);
      this->add_connection(src, n, w);// todo : find a kind of 1 ??
      this->add_connection(n, tgt, w);
      return n;	
    }

    void _del_neuron()
    {	
      assert(num_vertices(this->_g));

      if (this->get_nb_neurons() <= this->get_nb_inputs() + this->get_nb_outputs())
	return;
      vertex_desc_t v;
      do
	v = random_vertex(this->_g);
      while (this->is_output(v) || this->is_input(v));

      clear_vertex(v, this->_g);	
      remove_vertex(v, this->_g);
    }
    typename nn_t::weight_t _random_weight()
    {
      typename nn_t::weight_t w;
      w.random();
      return w;
    }
    void _add_conn()
    {
      this->add_connection(_random_src(), _random_tgt(), _random_weight());
    }
    // add a random connection by avoiding to duplicate an existent connection
    void _add_conn_nodup()
    {
      vertex_desc_t src, tgt;
      // this is only an upper bound; a connection might of course
      // be possible even after max_tries tries.
      size_t max_tries = num_vertices(this->_g) * num_vertices(this->_g), 
	nb_tries = 0;
      do 
	{
	  src = _random_src();
	  tgt = _random_tgt();
	}
      while (is_adjacent(this->_g, src, tgt) && ++nb_tries < max_tries);
      if (nb_tries < max_tries)
	{
	  typename nn_t::weight_t w;
	  w.random();
	  this->add_connection(src, tgt, w);
	}
    }
    void _del_conn()
    {
      if (!this->get_nb_connections())
	return;
      remove_edge(random_edge(this->_g), this->_g);
    }
    void _change_neurons()
    {
      BGL_FORALL_VERTICES_T(v, this->_g, graph_t) 
	{
	  this->_g[v].get_afparams().mutate();
	  this->_g[v].get_pfparams().mutate();
	}
    }

    // No dup version
    void _change_conns()
    {
      BGL_FORALL_EDGES_T(e, this->_g, graph_t)
	this->_g[e].get_weight().mutate();

      BGL_FORALL_EDGES_T(e, this->_g, graph_t)
	if (misc::rand<float>() < Params::dnn::m_rate_change_conn)
	  {
	    vertex_desc_t src = source(e, this->_g);
	    vertex_desc_t tgt = target(e, this->_g);
	    typename nn_t::weight_t w = this->_g[e].get_weight();
	    remove_edge(e, this->_g);
	    int max_tries = num_vertices(this->_g) * num_vertices(this->_g),
	      nb_tries = 0;
	    if (misc::flip_coin())
	      do
		src = _random_src();
	      while(++nb_tries < max_tries && is_adjacent(this->_g, src, tgt));
	    else
	      do
		tgt = _random_tgt();
	      while(++nb_tries < max_tries && is_adjacent(this->_g, src, tgt));
	    if (nb_tries < max_tries)
	      this->add_connection(src, tgt, w);
	    return;
	  }
    }
  };
}
}

#endif
