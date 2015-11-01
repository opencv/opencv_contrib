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




#ifndef PHEN_HYPER_NN_HPP
#define PHEN_HYPER_NN_HPP

#include <map>
#include <sferes/phen/indiv.hpp>
#include <modules/nn2/nn.hpp>

#include <modules/nn2/params.hpp>
#include "gen_hyper_nn.hpp"


namespace sferes
{
  namespace phen
  {
    namespace hnn
    {
      class Pos
      {
        public:
          Pos() {
          }
          Pos(float x, float y, float z) : _x(x), _y(y), _z(z) {
          }
          float dist(const Pos& p) const
          {
            float x = _x - p._x;
            float y = _y - p._y;
            float z = _z - p._z;
            return sqrt(x * x + y * y + z * z);
          }
          float x() const {
            return _x;
          }
          float y() const {
            return _y;
          }
          float z() const {
            return _z;
          }

          template<class Archive>
          void serialize(Archive& ar, const unsigned int version)
          {
            ar& BOOST_SERIALIZATION_NVP(_x);
            ar& BOOST_SERIALIZATION_NVP(_y);
            ar& BOOST_SERIALIZATION_NVP(_z);
          }
          bool operator == (const Pos &p)
          { return _x == p._x && _y == p._y && _z == p._z; }
        protected:
          float _x, _y, _z;
      };
    }

     // hyperneat-inspired phenotype, based on a cppn
    SFERES_INDIV(HyperNn, Indiv)
    {
      public:
        typedef Gen gen_t;
        typedef typename Params::hyper_nn::neuron_t neuron_t;
        typedef typename Params::hyper_nn::connection_t connection_t;
        typedef typename nn::NN<neuron_t, connection_t> nn_t;
        typedef typename nn_t::vertex_desc_t v_d_t;
        typedef typename gen_t::nn_t gen_nn_t;
        SFERES_CONST size_t nb_pfparams = Params::hyper_nn::nb_pfparams;
        SFERES_CONST size_t nb_afparams = Params::hyper_nn::nb_afparams;
        SFERES_CONST size_t nb_cppn_inputs = 2 + 2;
        SFERES_CONST size_t nb_cppn_outputs = 2;

        void develop()
        {
          this->_nn = nn_t();
          this->gen().init();
           // develop the parameters
          BGL_FORALL_VERTICES_T(v, this->gen().get_graph(),
                                typename gen_t::nn_t::graph_t)
          {
            this->gen().get_graph()[v].get_afparams().develop();
            this->gen().get_graph()[v].get_pfparams().develop();
          }
          BGL_FORALL_EDGES_T(e, this->gen().get_graph(),
                             typename gen_t::nn_t::graph_t)
          this->gen().get_graph()[e].get_weight().develop();
          assert(nb_cppn_inputs == this->gen().get_nb_inputs());
          assert(nb_cppn_outputs == this->gen().get_nb_outputs());

          _all_neurons.clear();

          size_t d = this->gen().get_depth();
           // create the nn
          _nn.set_nb_inputs(Params::hyper_nn::nb_inputs);
          _nn.set_nb_outputs(Params::hyper_nn::nb_outputs);
          SFERES_CONST size_t skip =
            Params::hyper_nn::nb_inputs
            + Params::hyper_nn::nb_outputs;
          SFERES_CONST size_t skip_total =
            Params::hyper_nn::nb_inputs
            + Params::hyper_nn::nb_outputs
            + Params::hyper_nn::nb_hidden;

          BGL_FORALL_VERTICES_T(v, _nn.get_graph(), typename nn_t::graph_t)
           /**/ _all_neurons.push_back(v);

           // hidden neurons
          for (size_t i = 0; i < Params::hyper_nn::nb_hidden; ++i)
          {
            v_d_t v = _nn.add_neuron(boost::lexical_cast<std::string>(i));
            _all_neurons.push_back(v);
          }

          assert(_all_neurons.size() ==
                 Params::hyper_nn::substrate_size() / 2
                 - Params::hyper_nn::nb_pfparams
                 - Params::hyper_nn::nb_afparams);

           // build the coordinate map
          for (size_t i = 0; i < _all_neurons.size() * 2; i += 2)
            this->_coords_map[_all_neurons[i / 2]] =
              hnn::Pos(Params::hyper_nn::substrate(i),
                       Params::hyper_nn::substrate(i + 1), 0);

           // afparams and pfparams
          for (size_t i = 0; i < _all_neurons.size(); ++i)
          {
            typename neuron_t::pf_t::params_t pfparams;
             // we put pfparams & afparams in [0:1]
             //for (size_t k = 0; k < nb_pfparams; ++k)
             //{
             // pfparams[k] = cppn_value(skip_total + k, i, false, 1) / 2.0f + 0.5f;
             // std::cout << " " << pfparams[k] << " ";
             //}
            typename neuron_t::af_t::params_t
              afparams;
            for (size_t k = 0; k < nb_afparams; ++k)
            {
              float b = cppn_value(skip_total + k + nb_pfparams, i,
                                   k + 1, false) / 2.0f + 0.5f;
              size_t bi = b * Params::hyper_nn::bias_size();
              bi = std::min(bi, Params::hyper_nn::bias_size() - 1);
              afparams[k] = Params::hyper_nn::bias(bi);
            }
            _nn.get_graph()[_all_neurons[i]].
            set_pfparams(pfparams);
            _nn.get_graph()[_all_neurons[i]].
            set_afparams(afparams);
          }
           // create connections
          for (size_t i = 0; i < skip_total * 2; i += 2)
            for (size_t j = 0; j < skip_total * 2; j += 2)
              if (!_nn.is_input(_all_neurons[j / 2]))
              {
                float w = cppn_value(i, j, 0, true);
                float ws = w >= 0 ? 1 : -1;
                ws *=
                  (fabs(w) -
                   Params::hyper_nn::conn_threshold) / (1 - Params::hyper_nn::conn_threshold);
                 // TODO generalize this
                 // ws is in [-1, 1] TODO : no guarantee that ws is in [-1;1]
                size_t wi = (int) ((ws / 2.0 + 0.5) * Params::hyper_nn::weights_size());
                wi = std::min(wi, Params::hyper_nn::weights_size() - 1);
                float wf = Params::hyper_nn::weights(wi);
                typename connection_t::weight_t weight = typename connection_t::weight_t(wf);
                if (fabs(w) >
                    Params::hyper_nn::conn_threshold)
                  _nn.add_connection(_all_neurons[i / 2],
                                     _all_neurons[j / 2],
                                     weight);
              }
          this->_nn.init();
        }

        float cppn_value(size_t i, size_t j,
                         size_t n, bool ff = false)
        {
          assert(i < Params::hyper_nn::substrate_size());
          assert(j < Params::hyper_nn::substrate_size());
          assert(i + 1 < Params::hyper_nn::substrate_size());
          assert(j + 1 < Params::hyper_nn::substrate_size());
          assert(n < nb_cppn_outputs);
          std::vector<float> in(nb_cppn_inputs);
          this->gen().init();
          in[0] = Params::hyper_nn::substrate(i);
          in[1] = Params::hyper_nn::substrate(i + 1);
          in[2] = Params::hyper_nn::substrate(j);
          in[3] = Params::hyper_nn::substrate(j + 1);
          if (in[1] == in[3])
            return 0;
          if (ff && (in[1] > in[3] || fabs(in[1] - in[3]) > Params::hyper_nn::max_y))
            return 0;
          for (size_t k = 0; k < this->gen().get_depth(); ++k)
            this->gen().step(in);
          return this->gen().get_outf(n);
        }

        void write_svg(std::ostream& ofs)
        {
           //_nn.write(os);
           //std::ofstream ofs("/tmp/nn.svg");
          ofs << "<svg width=\"200px\" height=\"200px\" viewbox=\"-500 -500 500 500\">";
          for (size_t i = 0; i < _all_neurons.size() * 2; i += 2)
          {
            float x = Params::hyper_nn::substrate(i);
            float y = Params::hyper_nn::substrate(i + 1);
            ofs << "<circle cx=\"" << x * 80 + 100
                << "\" cy=\"" << y * 80 + 100
                << "\" r=\"2\" fill=\"black\" "
                << "opacity=\""
                << 1
             // << std::max(0.0f, _nn.get_graph()[_all_neurons[i / 2]].get_pfparams()[0])
                << "\" "
                << " />" << std::endl;
          }
          typedef typename nn_t::graph_t graph_t;
          typedef typename nn_t::vertex_desc_t v_d_t;
          const graph_t& g = this->nn().get_graph();

          BGL_FORALL_EDGES_T(e, g, graph_t)
          {
            v_d_t src = boost::source(e, g);
            v_d_t tgt = boost::target(e, g);
            float x1 = _coords_map[src].x() * 80 + 100;
            float y1 = _coords_map[src].y() * 80 + 100;
            float x2 = _coords_map[tgt].x() * 80 + 100;
            float y2 = _coords_map[tgt].y() * 80 + 100;
            double weight = g[e].get_weight();
            ofs << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" "
                << "x2=\"" << x2 << "\" y2=\"" << y2 << "\""
                << " style=\"stroke:rgb("
                << (weight > 0 ? "0,255,0" : "255,0,0")
                << ");stroke-width:" << fabs(weight)
                << "\"/>"
                << std::endl;
          }

          ofs << "</svg>";
        }
        nn_t& nn() {
          return _nn;
        }
        const nn_t& nn() const {
          return _nn;
        }
        const std::vector<typename nn_t::vertex_desc_t>&
        all_neurons() const {
          return _all_neurons;
        }
        float compute_length(float min_length)
        {
          float length = 0;
          BGL_FORALL_EDGES_T(e, _nn.get_graph(), typename nn_t::graph_t)
          {
            typename nn_t::vertex_desc_t src = boost::source(e, _nn.get_graph());
            typename nn_t::vertex_desc_t tgt = boost::target(e, _nn.get_graph());
            double weight = _nn.get_graph()[e].get_weight();
            float l = _coords_map[src].dist(_coords_map[tgt]);
            length += l > min_length ? l : 0;
          }
          return length;
        }

      protected:
        nn_t _nn;
        std::vector<typename nn_t::vertex_desc_t> _all_neurons;
        std::map<typename nn_t::vertex_desc_t, hnn::Pos> _coords_map;
    };
  }
}


#endif
