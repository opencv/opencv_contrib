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




#ifndef EVAL_MPI_HPP_
#define EVAL_MPI_HPP_

#include <sferes/parallel.hpp>
#include <boost/mpi.hpp>

//#ifndef BOOST_MPI_HAS_NOARG_INITIALIZATION
//#error MPI need arguments (we require a full MPI2 implementation)
//#endif

#define MPI_INFO dbg::out(dbg::info, "mpi")<<"["<<_world->rank()<<"] "
namespace sferes {

  namespace eval {
    SFERES_CLASS(Mpi) {
    public:
      Mpi() {
        static char* argv[] = {(char*)"sferes2", 0x0};
        char** argv2 = (char**) malloc(sizeof(char*) * 2);
        int argc = 1;
        argv2[0] = argv[0];
        argv2[1] = argv[1];
        using namespace boost;
        dbg::out(dbg::info, "mpi")<<"Initializing MPI..."<<std::endl;
        _env = shared_ptr<mpi::environment>(new mpi::environment(argc, argv2, true));
        dbg::out(dbg::info, "mpi")<<"MPI initialized"<<std::endl;
        _world = shared_ptr<mpi::communicator>(new mpi::communicator());
        MPI_INFO << "communicator initialized"<<std::endl;
      }
      template<typename Phen>
      void eval(std::vector<boost::shared_ptr<Phen> >& pop,
                size_t begin, size_t end) {
        dbg::trace("mpi", DBG_HERE);
        if (_world->rank() == 0)
          _master_loop(pop, begin, end);
        else
          _slave_loop<Phen>();
      }
      ~Mpi() {
        MPI_INFO << "Finalizing MPI..."<<std::endl;
        std::string s("bye");
        if (_world->rank() == 0)
          for (size_t i = 1; i < _world->size(); ++i)
            _world->send(i, _env->max_tag(), s);
        _finalize();
      }
    protected:
      void _finalize() {
        _world = boost::shared_ptr<boost::mpi::communicator>();
        dbg::out(dbg::info, "mpi")<<"MPI world destroyed"<<std::endl;
        _env = boost::shared_ptr<boost::mpi::environment>();
        dbg::out(dbg::info, "mpi")<<"environment destroyed"<<std::endl;
      }
      template<typename Phen>
      void _master_loop(std::vector<boost::shared_ptr<Phen> >& pop,
                        size_t begin, size_t end) {
        dbg::trace("mpi", DBG_HERE);
        size_t current = begin;
        std::vector<bool> evaluated(pop.size());
        std::fill(evaluated.begin(), evaluated.end(), false);
        // first round
        for (size_t i = 1; i < _world->size() && current < end; ++i) {
          MPI_INFO << "[master] [send-init...] ->" <<i<<" [indiv="<<current<<"]"<<std::endl;
          _world->send(i, current, pop[current]->gen());
          MPI_INFO << "[master] [send-init ok] ->" <<i<<" [indiv="<<current<<"]"<<std::endl;
          ++current;
        }
        // send a new indiv each time we received a fitness
        while (current < end) {
          boost::mpi::status s = _recv(evaluated, pop);
          MPI_INFO << "[master] [send...] ->" <<s.source()<<" [indiv="<<current<<"]"<<std::endl;
          _world->send(s.source(), current, pop[current]->gen());
          MPI_INFO << "[master] [send ok] ->" <<s.source()<<" [indiv="<<current<<"]"<<std::endl;
          ++current;
        }
        //join
        bool done = true;
        do {
          dbg::out(dbg::info, "mpi")<<"joining..."<<std::endl;
          done = true;
          for (size_t i = begin; i < end; ++i)
            if (!evaluated[i]) {
              _recv(evaluated, pop);
              done = false;
            }
        } while (!done);
      }
      template<typename Phen>
      boost::mpi::status _recv(std::vector<bool>& evaluated,
                               std::vector<boost::shared_ptr<Phen> >& pop) {
        dbg::trace("mpi", DBG_HERE);
        using namespace boost::mpi;
        status s = _world->probe();
        MPI_INFO << "[rcv...]" << getpid() << " tag=" << s.tag() << std::endl;
        _world->recv(s.source(), s.tag(), pop[s.tag()]->fit());
        MPI_INFO << "[rcv ok]" << " tag=" << s.tag() << std::endl;
        evaluated[s.tag()] = true;
        return s;
      }
      template<typename Phen>
      void _slave_loop() {
        dbg::trace("mpi", DBG_HERE);
        while(true) {
          Phen p;
          boost::mpi::status s = _world->probe();
          if (s.tag() == _env->max_tag()) {
            MPI_INFO << "[slave] Quit requested" << std::endl;
            MPI_Finalize();
            exit(0);
          } else {
            MPI_INFO <<"[slave] [rcv...] [" << getpid()<< "]" << std::endl;
            _world->recv(0, s.tag(), p.gen());
            MPI_INFO <<"[slave] [rcv ok] " << " tag="<<s.tag()<<std::endl;
            p.develop();
            p.fit().eval(p);
            MPI_INFO <<"[slave] [send...]"<<" tag=" << s.tag()<<std::endl;
            _world->send(0, s.tag(), p.fit());
            MPI_INFO <<"[slave] [send ok]"<<" tag=" << s.tag()<<std::endl;

          }
        }
      }
      boost::shared_ptr<boost::mpi::environment> _env;
      boost::shared_ptr<boost::mpi::communicator> _world;
    };

  }
}

#endif
