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




#ifndef BATCH_EVAL_MPI_TBB_PARALLEL_HPP_
#define BATCH_EVAL_MPI_TBB_PARALLEL_HPP_

#include <sferes/parallel.hpp>
#include <boost/mpi.hpp>
#include "tbb_parallel_eval.hpp"
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

//#ifndef BOOST_MPI_HAS_NOARG_INITIALIZATION
//#error MPI need arguments (we require a full MPI2 implementation)
//#endif

#define MPI_INFO dbg::out(dbg::info, "mpi")<<"["<<_world->rank()<<"] "
namespace sferes {

  namespace eval {
    SFERES_CLASS(BatchMpiParallel) {
    public:
    	BatchMpiParallel()
    	{
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

        // Disable dumping out results for slave processes.
        if (_world->rank() > 0)
        {
        	Params::pop::dump_period = -1;
        }
      }

      template<typename Phen>
      void eval(std::vector<boost::shared_ptr<Phen> >& pop,
                size_t begin, size_t end) {
        dbg::trace("mpi", DBG_HERE);

        // Develop phenotypes in parallel
        // Each MPI process develops one phenotype
        if (_world->rank() == 0)
          _master_develop(pop, begin, end);
        else
          _slave_develop<Phen>();

        // Make sure the processes have finished developing phenotypes

        // Evaluate phenotypes in parallel but in batches of 256.
        // Caffe GPU supports max of 512.
        // There is no limit for CPU but we try to find out what batch size works best.
        if (_world->rank() == 0)
        {
        	_master_eval(pop, begin, end);
        }
      }

      ~BatchMpiParallel()
      {
        MPI_INFO << "Finalizing MPI..."<<std::endl;
        std::string s("bye");
        if (_world->rank() == 0)
          for (size_t i = 1; i < _world->size(); ++i)
            _world->send(i, _env->max_tag(), s);
        _finalize();
      }

    protected:
      void _finalize()
      {
        _world = boost::shared_ptr<boost::mpi::communicator>();
        dbg::out(dbg::info, "mpi")<<"MPI world destroyed"<<std::endl;
        _env = boost::shared_ptr<boost::mpi::environment>();
        dbg::out(dbg::info, "mpi")<<"environment destroyed"<<std::endl;
      }

      template<typename Phen>
      void _master_develop(std::vector<boost::shared_ptr<Phen> >& pop, size_t begin, size_t end)
      {
        dbg::trace("mpi", DBG_HERE);
        size_t current = begin;
        std::vector<bool> developed(pop.size());
        std::fill(developed.begin(), developed.end(), false);
        // first round
        for (size_t i = 1; i < _world->size() && current < end; ++i) {
          MPI_INFO << "[master] [send-init...] ->" <<i<<" [indiv="<<current<<"]"<<std::endl;
          _world->send(i, current, pop[current]->gen());
          MPI_INFO << "[master] [send-init ok] ->" <<i<<" [indiv="<<current<<"]"<<std::endl;
          ++current;
        }
        // send a new indiv each time we received a fitness
        while (current < end) {
          boost::mpi::status s = _recv(developed, pop);
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
            if (!developed[i]) {
              _recv(developed, pop);
              done = false;
            }
        } while (!done);
      }

      template<typename Phen>
      boost::mpi::status _recv(std::vector<bool>& developed,
                               std::vector<boost::shared_ptr<Phen> >& pop)
      {
        dbg::trace("mpi", DBG_HERE);
        using namespace boost::mpi;
        status s = _world->probe();
        MPI_INFO << "[rcv...]" << getpid() << " tag=" << s.tag() << std::endl;
        //_world->recv(s.source(), s.tag(), pop[s.tag()]->fit());

        // Receive the whole developed phenotype from slave processes
        Phen p;
        _world->recv(s.source(), s.tag(), p);

        // Assign the developed phenotype back to the current population for further evaluation
        pop[s.tag()]->image() = p.image();

        MPI_INFO << "[rcv ok]" << " tag=" << s.tag() << std::endl;
        developed[s.tag()] = true;
        return s;
      }

      template<typename Phen>
      void _slave_develop()
      {
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

            MPI_INFO <<"[slave] [send...]"<<" tag=" << s.tag()<<std::endl;
            //_world->send(0, s.tag(), p.fit());	// Send only the fitness back to master process

            // Send the whole phenotype back to master process
            _world->send(0, s.tag(), p);
            MPI_INFO <<"[slave] [send ok]"<<" tag=" << s.tag()<<std::endl;

          }
        }
      }

      // ----------------------------------------------------------------------------------
      template<typename Phen>
			void _master_eval(std::vector<boost::shared_ptr<Phen> >& pop, size_t begin, size_t end)
			{
      	dbg::trace trace("eval", DBG_HERE);

				assert(pop.size());
				assert(begin < pop.size());
				assert(end <= pop.size());

				// Number of eval iterations
				const size_t count = end - begin;

				LOG(INFO) << "Total: " << count << " | Batch: " << Params::image::batch << "\n";

				// Evaluate phenotypes in parallel using TBB.
				parallel::init();
				parallel::p_for(
						parallel::range_t(begin, end, Params::image::batch),
						sferes::eval::parallel_tbb_eval<Phen>(pop, Params::image::model_definition, Params::image::pretrained_model));

				// The barrier is implicitly set here after the for-loop in TBB.
			}

      boost::shared_ptr<boost::mpi::environment> _env;
      boost::shared_ptr<boost::mpi::communicator> _world;
    };

  }
}

#endif
