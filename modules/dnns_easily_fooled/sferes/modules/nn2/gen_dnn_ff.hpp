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




#ifndef DNN_FF_HPP_
#define DNN_FF_HPP_

#include <modules/nn2/gen_dnn.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/dag_shortest_paths.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/property_map/vector_property_map.hpp>

namespace sferes
{
  namespace gen
  {
    template<typename N, typename C, typename Params>
    class DnnFF : public Dnn<N, C, Params>
    {
    public:
      typedef nn::NN<N, C> nn_t;
      typedef N neuron_t;
      typedef C conn_t;
      typedef typename nn_t::io_t io_t;
      typedef typename nn_t::vertex_desc_t vertex_desc_t;
      typedef typename nn_t::edge_desc_t edge_desc_t;
      typedef typename nn_t::graph_t graph_t;
      DnnFF() {}
      DnnFF& operator=(const DnnFF& o)
      {
	static_cast<nn::NN<N, C>& >(*this)
	  = static_cast<const nn::NN<N, C>& >(o);
	return *this;
      }
      DnnFF(const DnnFF& o)
      { *this = o; }
      void init()
      {
	Dnn<N, C, Params>::init();
	_compute_depth();
      }
      void random()
      {
	assert(Params::dnn::init == dnn::ff);
	this->_random_ff(Params::dnn::nb_inputs, Params::dnn::nb_outputs);
	_make_all_vertices();
      }
      void mutate()
      {
 	_change_conns();
	this->_change_neurons();

 	if (misc::rand<float>() < Params::dnn::m_rate_add_conn)
 	  _add_conn();

 	if (misc::rand<float>() < Params::dnn::m_rate_del_conn)
  	  this->_del_conn();

  	if (misc::rand<float>() < Params::dnn::m_rate_add_neuron)
  	  this->_add_neuron_on_conn();

 	if (misc::rand<float>() < Params::dnn::m_rate_del_neuron)
   	  this->_del_neuron();
      }

      void cross(const DnnFF& o, DnnFF& c1, DnnFF& c2)
      {
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
      }
      size_t get_depth() const { return _depth; }
    protected:
      std::set<vertex_desc_t> _all_vertices;
      size_t _depth;

      void _make_all_vertices()
      {
	_all_vertices.clear();
	BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
	  _all_vertices.insert(v);
      }
      void _change_conns()
      {
	BGL_FORALL_EDGES_T(e, this->_g, graph_t)
	  this->_g[e].get_weight().mutate();
      }


      // add only feed-forward connections
      void _add_conn()
      {
	using namespace boost;
	vertex_desc_t v = this->_random_src();
	std::set<vertex_desc_t> preds;
	nn::bfs_pred_visitor<vertex_desc_t> vis(preds);
	breadth_first_search(make_reverse_graph(this->_g),
			     v, color_map(get(&N::_color, this->_g)).visitor(vis));
	_make_all_vertices();
	std::set<vertex_desc_t> tmp, avail, in;
	// avoid to connect to predecessors
	std::set_difference(_all_vertices.begin(), _all_vertices.end(),
			    preds.begin(), preds.end(),
			    std::insert_iterator<std::set<vertex_desc_t> >(tmp, tmp.begin()));
	// avoid to connect to inputs
	BOOST_FOREACH(vertex_desc_t v, this->_inputs) // inputs need
						      // to be sorted
	    in.insert(v);
	std::set_difference(tmp.begin(), tmp.end(),
			    in.begin(), in.end(),
			    std::insert_iterator<std::set<vertex_desc_t> >(avail, avail.begin()));

	if (avail.empty())
	  return;
	vertex_desc_t tgt = *misc::rand_l(avail);
	typename nn_t::weight_t w;
	w.random();
	this->add_connection(v, tgt, w);
      }

      // useful to make the right number of steps
      void _compute_depth()
      {
	using namespace boost;
	typedef std::map<vertex_desc_t, size_t> int_map_t;
	typedef std::map<vertex_desc_t, vertex_desc_t> vertex_map_t;
	typedef std::map<vertex_desc_t, default_color_type> color_map_t;
	typedef std::map<edge_desc_t, int> edge_map_t;

	typedef associative_property_map<int_map_t> a_map_t;
	typedef associative_property_map<color_map_t> c_map_t;
	typedef associative_property_map<vertex_map_t> v_map_t;
	typedef associative_property_map<edge_map_t> e_map_t;

	color_map_t cm; c_map_t cmap(cm);
	vertex_map_t vm; v_map_t pmap(vm);
	edge_map_t em;
	BGL_FORALL_EDGES_T(e, this->_g, graph_t)
	  em[e] = 1;
	e_map_t wmap(em);
	_depth = 0;
	// we compute the longest path between inputs and outputs
	BOOST_FOREACH(vertex_desc_t s, this->_inputs)
	  {
	    int_map_t im; a_map_t dmap(im);
 	    dag_shortest_paths
 	      (this->_g, s, dmap, wmap, cmap, pmap,
	       dijkstra_visitor<null_visitor>(),
 	       std::greater<int>(),
	       closed_plus<int>(),
	       std::numeric_limits<int>::min(), 0);

 	    BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
	      {
		size_t d = get(dmap, v);
		if (this->_g[v].get_out() != -1 && d <= num_vertices(this->_g))
		  _depth = std::max(_depth, d);
	      }
	  }
	// add one to be sure
	_depth ++;
      }

    };

  }
}

#endif
