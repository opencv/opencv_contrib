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


#ifndef _NN_HPP_
#define _NN_HPP_

#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/foreach.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/mpl/assert.hpp>

#include <cmath>
#include <valarray>

#include "pf.hpp"
#include "af.hpp"
#include "neuron.hpp"
#include "connection.hpp"

namespace nn
{
  // a useful boost functor
  template<typename V>
   class bfs_pred_visitor : public boost::default_bfs_visitor
   {
    public:
      bfs_pred_visitor(std::set<V>& pred) : _pred(pred) {}
      template <typename Vertex, typename Graph >
       void discover_vertex(Vertex u, const Graph & g)
        {
         _pred.insert(u);
        }
    protected:
      std::set<V>& _pred;
   };


  // main class
  // N : neuron type, C : connection type
  template<typename N, typename C>
   class NN
    {
     public:
       // types
       typedef boost::adjacency_list<boost::listS, boost::listS,
               boost::bidirectionalS,
               N, C> graph_t;
       typedef typename boost::graph_traits<graph_t>::vertex_iterator vertex_it_t;
       typedef typename boost::graph_traits<graph_t>::edge_iterator edge_it_t;
       typedef typename boost::graph_traits<graph_t>::out_edge_iterator out_edge_it_t;
       typedef typename boost::graph_traits<graph_t>::in_edge_iterator in_edge_it_t;
       typedef typename boost::graph_traits<graph_t>::edge_descriptor edge_desc_t;
       typedef typename boost::graph_traits<graph_t>::vertex_descriptor vertex_desc_t;
       typedef typename boost::graph_traits<graph_t>::adjacency_iterator adj_it_t;
       typedef typename std::vector<vertex_desc_t> vertex_list_t;
       typedef N neuron_t;
       typedef C conn_t;
       typedef typename N::af_t af_t;
       typedef typename N::pf_t pf_t;
       typedef typename C::weight_t weight_t;
       typedef typename C::io_t io_t;

       // constructor
       NN() : _neuron_counter(0), _init_done(false)
     {}
       NN(const NN& o) { *this = o; }
       NN& operator=(const NN& o)
        {
         if (&o == this)
         return *this;
         _g = o._g;
         _neuron_counter = o._neuron_counter;
         _inputs.clear();
         _outputs.clear();
         _inputs.resize(o.get_nb_inputs());
         _outputs.resize(o.get_nb_outputs());
         _init_io();
         _init_done = false;
         return *this;
        }
       // init
       void init() { _init(); }
       // set id for inputs and outputs
       void name_io() { _name_io(); }
       // load/write
       //void load(const std::string& fname) { _load_graph(fname); }
       //    void write(const std::string& fname) { _write_graph(fname); }
       void write(std::ostream& ofs) { _write_dot(ofs); }
       void dump(std::ostream& ofs) const
        {
         std::pair<vertex_it_t, vertex_it_t> vp;
         for (vp = boost::vertices(_g); vp.first != vp.second; ++vp.first)
         ofs<<_g[*vp.first]._id<<" "<<_g[*vp.first].get_next_output()<<" ";
         ofs<<std::endl;
        }

       // get the graph for graph algorithms
       const graph_t& get_graph() const { return _g; }
       graph_t& get_graph() { return _g; }

       // construction
       vertex_desc_t add_neuron(const std::string& label)
        {
         vertex_desc_t v = add_vertex(_g);
         _g[v]._id = boost::lexical_cast<std::string>(_neuron_counter++);
         _g[v]._label = label;
         return v;
        }

       vertex_desc_t add_neuron(const std::string& label,
             const typename pf_t::params_t& pf_params,
             const typename af_t::params_t& af_params)
        {
         vertex_desc_t v = add_neuron(label);
         _g[v].set_pfparams(pf_params);
         _g[v].set_afparams(af_params);
         return v;
        }

       bool add_connection(const vertex_desc_t& u,
             const vertex_desc_t& v,
             weight_t weight)
        {
         std::pair<edge_desc_t, bool> e = add_edge(u, v, _g);
         if (e.second)
         _g[e.first].set_weight(weight);
         return e.second;
        }
       // special version when you need to increase weight
       bool add_connection_w(const vertex_desc_t& u,
             const vertex_desc_t& v,
             weight_t weight)
        {
         std::pair<edge_desc_t, bool> e = add_edge(u, v, _g);
         if (e.second)
         _g[e.first].set_weight(weight);
         else
         _g[e.first].set_weight(_g[e.first].get_weight() + weight);
         return e.second;
        }

       void set_all_pfparams(const std::vector<typename pf_t::params_t>& pfs)
        {
         assert(num_vertices(_g) == pfs.size());
         size_t k = 0;
         BGL_FORALL_VERTICES_T(v, _g, graph_t)
          _g[v].set_pfparamst(pfs[k++]);
        }

       void set_all_afparams(const std::vector<typename af_t::params_t>& afs)
        {
         assert(num_vertices(_g) == afs.size());
         size_t k = 0;
         BGL_FORALL_VERTICES_T(v, _g, graph_t)
          _g[v].set_afparamst(afs[k++]);
        }

       void set_all_weights(const std::vector<weight_t>& ws)
        {
#ifndef NDEBUG
         if (num_edges(_g) != ws.size())
         std::cout << "param errors: "
          << num_edges(_g)
          << " whereas "
          << ws.size()
          << " provided" <<std::endl;
#endif
         assert(num_edges(_g) == ws.size());
         size_t k = 0;
         BGL_FORALL_EDGES_T(e, _g, graph_t)
          _g[e].set_weight(ws[k++]);
        }

       void set_nb_inputs(unsigned i)
        {
         _inputs.resize(i);
         size_t k = 0;
         BOOST_FOREACH(vertex_desc_t& v, _inputs)
          {
           v = add_vertex(_g);
           this->_g[v].set_in(k++);
          }
        }
       void set_nb_outputs(unsigned i)
        {
         _outputs.resize(i);
         size_t k = 0;
         BOOST_FOREACH(vertex_desc_t& v, _outputs)
          {
           v = add_vertex(_g);
           this->_g[v].set_out(k++);
          }
        }
       vertex_desc_t get_input(int i) const
        {
         assert((size_t)i < _inputs.size());
         assert(this->_g[_inputs[i]].get_in() != -1);
         return _inputs[i];
        }
       const std::vector<vertex_desc_t>& get_inputs() const { return _inputs; }
       const std::vector<vertex_desc_t>& get_outputs() const { return _outputs; }
       // warning : O(n)
       vertex_desc_t get_neuron(size_t i) const
        {
         i = std::min(num_vertices(_g) - 1, i);
         size_t k = 0;
         BGL_FORALL_VERTICES_T(v, _g, graph_t)
          if (k++ == i)
          return v;
         assert(0);
         return (vertex_desc_t)(0x0);;
        }
       vertex_list_t get_neuron_list()
        {
         vertex_list_t neuron_list;
         std::pair<vertex_it_t, vertex_it_t> vp;
         for (vp = boost::vertices(_g); vp.first != vp.second; ++vp.first)
          {
           neuron_list.push_back(vertex_desc_t(*vp.first));
          }
         return neuron_list;
        }
       neuron_t& get_neuron_by_vertex(vertex_desc_t v)
        {
         return this->_g[v];
        }
       io_t get_neuron_output(size_t i) const
        {
         return _g[get_neuron(i)].get_current_output();
        }

       std::string get_neuron_id(size_t i) const
        {
         return _g[get_neuron(i)]._id;
        }
       vertex_desc_t get_output(int i) const
        {
         assert((size_t) i < _outputs.size());
         assert(this->_g[_outputs[i]].get_out() != -1);
         return _outputs[i];
        }
       const N& get_output_neuron(int i) const
        {
         return _g[_outputs[i]];
        }
       bool is_output(const vertex_desc_t& v) const
        {
         return std::find(_outputs.begin(), _outputs.end(), v) != _outputs.end();
        }
       bool is_input(const vertex_desc_t& v) const
        {
         return std::find(_inputs.begin(), _inputs.end(), v) != _inputs.end();
        }

       // step
       void step(const std::vector<io_t>& inputs) { _step(inputs); }

       // accessors
       const std::vector<io_t>& get_outf() const { return _outf; }
       io_t get_outf(unsigned i) const { return _outf[i]; }
       const std::vector<io_t>& outf() const { return get_outf(); }
       io_t outf(unsigned i) const { return get_outf(i); }
       unsigned get_nb_inputs() const { return _inputs.size(); }
       unsigned get_nb_outputs() const { return _outputs.size(); }
       unsigned get_nb_connections() const { return num_edges(_g); }
       unsigned get_nb_neurons() const { return num_vertices(_g); }

       // subnns
       void remove_subnn(const std::set<vertex_desc_t>& subnn)
        {
         BOOST_FOREACH(vertex_desc_t v, subnn)
          if (!is_input(v) && !is_output(v))
           {
            clear_vertex(v, _g);
            remove_vertex(v, _g);
           }
         _init_io();
        }
       template<typename NN>
        void add_subnn(const NN& nn,
              const std::vector<size_t>& inputs,
              const std::vector<size_t>& outputs)
         {
          assert(inputs.size() == nn.get_nb_inputs());
          assert(outputs.size() == nn.get_nb_outputs());
          std::map<typename NN::vertex_desc_t, vertex_desc_t> rmap;
          const typename NN::graph_t& g_src = nn.get_graph();
          BGL_FORALL_VERTICES_T(v, g_src, typename NN::graph_t)
           if (g_src[v].get_in() == -1 && g_src[v].get_out() == -1)
            {
             vertex_desc_t nv = add_vertex(_g);
             _g[nv] = g_src[v];
             _g[nv]._id = boost::lexical_cast<std::string>(_neuron_counter++);
             rmap[v] = nv;
            }

          std::vector<vertex_desc_t> vnodes;
          // hoping that the order did not change too much
          BGL_FORALL_VERTICES_T(v, _g, graph_t)
           vnodes.push_back(v);

          BGL_FORALL_EDGES_T(e, g_src, typename NN::graph_t)
           {
            std::pair<edge_desc_t, bool> ne;
            int in  = g_src[source(e, g_src)].get_in();
            int out = g_src[target(e, g_src)].get_out();
            assert(in == -1 || in < inputs.size());
            assert(out == -1 || out < outputs.size());
            if (in != -1 && out != -1)
             {
              int n_in = std::min(vnodes.size() - 1, inputs[in]);
              int n_out = std::min(vnodes.size() - 1, outputs[out]);
              ne = add_edge(vnodes[n_in], vnodes[n_out], _g);
             }
            else if (in != -1)
             {
              int n_in = std::min(vnodes.size() - 1, inputs[in]);
              ne = add_edge(vnodes[n_in], rmap[target(e, g_src)], _g);
             }
            else if (out != -1)
             {
              int n_out = std::min(vnodes.size() - 1, outputs[out]);
              ne = add_edge(rmap[source(e, g_src)], vnodes[n_out], _g);
             }
            else
             {
              assert(rmap.find(source(e, g_src)) != rmap.end());
              assert(rmap.find(target(e, g_src)) != rmap.end());
              ne = add_edge(rmap[source(e, g_src)], rmap[target(e, g_src)], _g);
             }
            _g[ne.first] = g_src[e];
           }

          _init_io();
         }

      // remove the connection with a weigth that is smaller (in absolute value) to the threshold 
      // !!! WARNING
      // this method will destroy your neural network...
      int remove_low_weights(float threshold)
      {
	int nb_removed = 0;
	std::vector<edge_desc_t> to_remove;
	BGL_FORALL_EDGES_T(e, this->_g, graph_t)
          {
	    if (fabs(_g[e].get_weight()) < threshold)
	      to_remove.push_back(e);
          }
	for (size_t i = 0; i < to_remove.size(); ++i)
	  remove_edge(to_remove[i], this->_g);
	return to_remove.size();
      }

       // remove neurons that are not connected to both one input and
       // one output (this is NOT callled automatically in NN
       //
       // WARNING: if simplify_in is true, this can change the behavior
       // of neurons since neurons not connected to inputs but connected
       // to outputs can output a constant value
       //
       // principle : keep the neurons that are successors of inputs
       // and predecessors of outputs
       void simplify(bool simplify_in = false)
        {
         // we need sets and not lists withouh io
         std::set<vertex_desc_t> all_neurons;
         BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
          if (!is_input(v) && !is_output(v))
          all_neurons.insert(v);
         std::set<vertex_desc_t> out_preds, in_succs;

         // out
         BOOST_FOREACH(vertex_desc_t v, this->_outputs)
          {
           std::set<vertex_desc_t> preds;
           nn::bfs_pred_visitor<vertex_desc_t> vis(preds);
           breadth_first_search(boost::make_reverse_graph(_g),
                 v, color_map(get(&N::_color, _g)).visitor(vis));
           out_preds.insert(preds.begin(), preds.end());
          }
         // in
         if (simplify_in)
         BOOST_FOREACH(vertex_desc_t v, this->_inputs)
          {
           std::set<vertex_desc_t> succs;
           nn::bfs_pred_visitor<vertex_desc_t> vis(succs);
           breadth_first_search(_g,
                 v, color_map(get(&N::_color, _g)).visitor(vis));
           in_succs.insert(succs.begin(), succs.end());
          }
         else
         in_succs = all_neurons;
         // make the intersection of in_succ and out_preds
         std::set<vertex_desc_t> valid_neurons;
         std::set_intersection(in_succs.begin(), in_succs.end(),
               out_preds.begin(), out_preds.end(),
               std::insert_iterator<std::set<vertex_desc_t> >(valid_neurons,
                 valid_neurons.begin()));
         // get the list of neurons that are NOT in valid_neurons
         std::set<vertex_desc_t> to_remove;
         std::set_difference(all_neurons.begin(), all_neurons.end(),
               valid_neurons.begin(), valid_neurons.end(),
               std::insert_iterator<std::set<vertex_desc_t> >(to_remove,
                 to_remove.begin()));
         // remove these neurons
         BOOST_FOREACH(vertex_desc_t v, to_remove)
          {
           clear_vertex(v, _g);
           remove_vertex(v, _g);
          }
        }
       // fully connect two vectors of neurons
       void full_connect(const std::vector<vertex_desc_t> v1,
             const std::vector<vertex_desc_t> v2,
             const weight_t& w)
        {
         BOOST_FOREACH(vertex_desc_t x, v1)
          BOOST_FOREACH(vertex_desc_t y, v2)
          this->add_connection(x, y, w);
        }

       // 1 to 1 connection
       void connect(const std::vector<vertex_desc_t> v1,
             const std::vector<vertex_desc_t> v2,
             const weight_t& w)
        {
         assert(v1.size() == v2.size());
         for (size_t i = 0; i < v1.size(); ++i)
         this->add_connection(v1[i], v2[i], w);
        }
     protected:
       // attributes
       graph_t _g;
       vertex_list_t _inputs;
       vertex_list_t _outputs;
       std::vector<io_t> _outf;
       int _neuron_counter;
       bool _init_done;

       // methods
       void _write_dot(std::ostream& ofs)
        {
         ofs << "digraph G {" << std::endl;
         BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
          {
           ofs << this->_g[v].get_id();
           ofs << " [label=\""<<this->_g[v].get_id()<<"\"";
           // ofs << " af"<< this->_g[v].get_afparams();
           // ofs << "| pf"<< this->_g[v].get_pfparams() <<"\"";
           if (is_input(v) || is_output(v))
	     ofs<<" shape=doublecircle";
	   
           ofs <<"]"<< std::endl;
          }
         BGL_FORALL_EDGES_T(e, this->_g, graph_t)
          {
           ofs << this->_g[source(e, this->_g)].get_id()
            << " -> " << this->_g[target(e, this->_g)].get_id()
	       << "[label=\"" << _g[e].get_weight() << "\"]" << std::endl;
          }
         ofs << "}" << std::endl;
        }

       void _activate(vertex_desc_t n)
        {
         using namespace boost;
         if (_g[n].get_fixed()) return;

         in_edge_it_t in, in_end;
         unsigned i = 0;
         for (tie(in, in_end) = in_edges(n, _g); in != in_end; ++in, ++i)
         _g[n].set_input(i, _g[source(*in, _g)].get_current_output());

         _g[n].activate();
        }

       void _set_in(const std::vector<io_t>& inf)
        {
         assert(inf.size() == _inputs.size());
         if (inf.size()>0) {
           unsigned i = 0;
           for (typename vertex_list_t::const_iterator it = _inputs.begin();
                 it != _inputs.end(); ++it, ++i)
            {
             _g[*it].set_current_output(inf[i]);
             _g[*it].set_next_output(inf[i]);
            }
         }
        }
       void _set_out()
        {
         unsigned i = 0;
         for (typename vertex_list_t::const_iterator it = _outputs.begin();
               it != _outputs.end(); ++it, ++i)
         _outf[i] = _g[*it].get_current_output();
        }
       void _step(const std::vector<io_t>& inf)
        {
         assert(_init_done);
         // in
         _set_in(inf);

         // activate
         std::pair<vertex_it_t, vertex_it_t> vp;
         for (vp = boost::vertices(_g); vp.first != vp.second; ++vp.first)
         _activate(*vp.first);

         // step
         for (vp = boost::vertices(_g); vp.first != vp.second; ++vp.first)
         _g[*vp.first].step();

         // out
         _set_out();
        }

    void _name_io()
    {
      int i=0;
      BOOST_FOREACH(vertex_desc_t v, _inputs)
	{
	  _g[v]._id = std::string("i") + boost::lexical_cast<std::string>(i);
	  ++i;
	}
      i = 0;
      BOOST_FOREACH(vertex_desc_t v, _outputs) {
	_g[v]._id = std::string("o") + boost::lexical_cast<std::string>(i);
	++i;
      }
    }
    void _init()
    {
      //      BOOST_MPL_ASSERT((boost::mpl::is_same<N::weight_t, C::weight_t>));
      // BOOST_MPL_ASSERT((boost::mpl::is_same<Pot::weight_t, C::weight_t>));
      _outf.clear();
      in_edge_it_t in, in_end;
      std::pair<vertex_it_t, vertex_it_t> vp;
      int k = 0;
      for (vp = boost::vertices(_g); vp.first != vp.second; ++vp.first)
	{
	  vertex_desc_t n = *vp.first;
	  _g[n].set_in_degree(in_degree(n, _g));
	  _g[n].set_id(boost::lexical_cast<std::string>(k++));
	  unsigned i = 0;
	  for (tie(in, in_end) = in_edges(n, _g); in != in_end; ++in, ++i)
	    _g[n].set_weight(i, _g[*in].get_weight());
	}
      _outf.resize(_outputs.size());
      BOOST_FOREACH(vertex_desc_t v, _inputs)
	{
	  _g[v].set_fixed();
	  _g[v].set_current_output(N::zero());
	}
      // init to 0
      for (vp = boost::vertices(_g); vp.first != vp.second; ++vp.first)
	_g[*vp.first].init();
      _init_io();
      _name_io();
      _init_done = true;
    }
    void _init_io()
    {
      BGL_FORALL_VERTICES_T(v, _g, graph_t)
	{
	  if (_g[v].get_in() != -1)
	    {
	      assert(_g[v].get_in() < (int)_inputs.size());
	      _inputs[_g[v].get_in()] = v;
	    }
	  if (_g[v].get_out() != -1)
	    {
	      assert(_g[v].get_out() < (int)_outputs.size());
	      _outputs[_g[v].get_out()] = v;
	    }
	}
    }
  };
}

#endif



