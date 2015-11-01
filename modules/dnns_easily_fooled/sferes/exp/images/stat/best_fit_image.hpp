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




#ifndef BEST_FIT_IMAGE_
#define BEST_FIT_IMAGE_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include "image_stat.hpp"
#include <sferes/fit/fitness.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>

// Headers specifics to the computations we need
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>

#include <exp/images/util/median.hpp>
#include <boost/filesystem.hpp>

namespace sferes {
  namespace stat {
    // assume that the population is sorted !
    SFERES_STAT(BestFitImage, ImageStat){
    public:
    	typedef boost::shared_ptr<Phen> indiv_t;
    	typedef std::vector<indiv_t> pop_t;

      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        //_best = *ea.pop().begin();

        // Create the log file
        this->_create_log_file(ea, "bestfit.dat");

        if (ea.dump_enabled() && ( ea.gen() % Params::pop::dump_period == 0 ))
        {
        	// The data for which we wish to calculate median as the boost accumulator does not.
					std::vector< double > data;

					// Calculate all stats: mean, max, min
					boost::accumulators::accumulator_set<double, boost::accumulators::stats<
					boost::accumulators::tag::mean,
					boost::accumulators::tag::max,
					boost::accumulators::tag::min
					> > stats;

        	int best_id = 0;
        	float best_fitness = 0.0f;
        	int id = 0;

        	BOOST_FOREACH(indiv_t i, ea.pop())
        	{
        		// With the stats object in hand, all we need is to push in the data.
        		float fitness = i->fit().value();
        		stats(fitness);
        		data.push_back(fitness);							// Add it to the list for calculating median later

        		// Get the best individual by fitness
        		if (fitness > best_fitness)
        		{
        			best_id = id;
        			best_fitness = fitness;
        		}

        		++id;
        	}

        	// Best individual in the current pop
        	_best = ea.pop()[best_id];

        	assert(data.size() == Params::pop::size);	// Make sure the list of values is of correct size
					double median = sferes::util::Median::calculate_median(data); // Get the mdian

        	// Dump best_fit.dat file
          (*this->_log_file) << ea.gen()
          		<< " " << median
          		<< " " << boost::accumulators::mean(stats)
          		<< " " << boost::accumulators::max(stats)
          		<< " " << boost::accumulators::min(stats)
          		<< std::endl;

          // Dump best image
					if (Params::log::best_image)
					{
						std::string image_fitness = boost::lexical_cast<std::string>(_best->fit().value());
						std::string image_gen = boost::lexical_cast<std::string>(ea.gen());
						std::string image_file = ea.res_dir() + "/" + image_gen + "_" + image_fitness;

						_best->log_best_image_fitness(image_file);
					}

					// Save this generation population to a file
					// Clear all individuals
					_pop.clear();

					// The mixed population (before selection)
					for (size_t i = 0; i < ea.pop().size(); ++i)
					{
						_pop.push_back(ea.pop()[i]);
					}
        }
      }
      void show(std::ostream& os, size_t k) {
        _best->develop();
        _best->show(os);
        _best->fit().set_mode(fit::mode::view);
        _best->fit().eval(*_best);

      }
      const boost::shared_ptr<Phen> best() const {
        return _best;
      }

      template<class Archive>
			void serialize(Archive& ar, const unsigned int version)
			{
				ar & BOOST_SERIALIZATION_NVP(_pop);
				ar & BOOST_SERIALIZATION_NVP(_best);
			}

      const pop_t& getPopulation() const
			{
				return _pop;
			}

    protected:
      indiv_t _best;
      pop_t _pop;
    };
  }
}
#endif
