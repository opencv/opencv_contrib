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

#ifndef MAP_ELITE_HPP_
#define MAP_ELITE_HPP_

#include <algorithm>
#include <limits>

#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>

#include <sferes/stc.hpp>
#include <exp/images/ea/ea_custom.hpp>
#include <sferes/fit/fitness.hpp>

#include <exp/images/continue_run/continue_run.hpp>

namespace sferes
{
	namespace ea
	{
		// Main class
		SFERES_EA(MapElite, EaCustom){
public:

	typedef boost::shared_ptr<Phen> indiv_t;

	typedef std::vector<indiv_t> raw_pop_t;
	typedef MapElite<Phen, Eval, Stat, FitModifier, Params, Exact> this_t;

	typedef typename std::vector<indiv_t> pop_t;
	typedef typename pop_t::iterator it_t;
	typedef typename std::vector<std::vector<indiv_t> > front_t;
	typedef boost::array<float, 2> point_t;
	typedef boost::shared_ptr<Phen> phen_t;
	typedef boost::multi_array<phen_t, 2> array_t;
//	typedef boost::shared_ptr<Stat> stat_t;

	static const size_t res_x = Params::ea::res_x;
	static const size_t res_y = Params::ea::res_y;

	typedef Stat stat_t;

	MapElite() :
	_array(boost::extents[res_x][res_y]),
	_array_parents(boost::extents[res_x][res_y]),
	_jumps(0)
	{
	}

	void random_pop()
	{
		// parallel::init(); We are not using TBB

		// Continuing a run
		sferes::cont::Continuator<this_t, Params> continuator;

		bool continue_run = continuator.enabled() || this->_gen_file_path != "";

		// Continuing a run manually from command line or continuing a run automatically if the job was pre-empted
		if(continue_run)
		{
			// Load the population file
			raw_pop_t raw_pop;
			if (this->_gen_file_path == "")
			{
				raw_pop = continuator.getPopulationFromFile(*this);
			}
			else
			{
				raw_pop = continuator.getPopulationFromFile(*this, this->_gen_file_path);
			}

			// Assign this pop also to the current map
			for (size_t i = 0; i < raw_pop.size(); ++i)
			{
				_add_to_archive(raw_pop[i], raw_pop[i]);
			}

			// Get the number of population to continue with
			const size_t init_size = raw_pop.size();

			// Resize the current population archive
			this->_pop.resize(init_size);

			// Add loaded individuals to the new population
			int i = 0;
			BOOST_FOREACH(boost::shared_ptr<Phen>&indiv, this->_pop)
			{
				indiv = boost::shared_ptr<Phen>(new Phen(*raw_pop[i]));
				++i;
			}
		}
		else	// Run normally from gen = 0
		{
			// Original Map-Elites code
			// Intialize a random population
			this->_pop.resize(Params::pop::init_size);
			BOOST_FOREACH(boost::shared_ptr<Phen>&indiv, this->_pop)
			{
				indiv = boost::shared_ptr<Phen>(new Phen());
				indiv->random();
			}
		}

		// Evaluate the initialized population
		this->_eval.eval(this->_pop, 0, this->_pop.size());
		BOOST_FOREACH(boost::shared_ptr<Phen>&indiv, this->_pop)
		_add_to_archive(indiv, indiv);

		// Continue a run from a specific generation
		if(continue_run)
		{
			if (this->_gen_file_path == "")
			{
				continuator.run_with_current_population(*this);
			}
			else
			{
				continuator.run_with_current_population(*this, this->_gen_file_path);
			}
		}
	}

	//ADDED
	void setGen(size_t gen)
	{
		this->_gen = gen;
	}
	//ADDED END

	void epoch()
	{
#ifdef PHELOGENETIC_TREE
		// We start with only 1 organism in order to construct the phylogenetic tree
		// Thus, no evolution happening at generation 0
		if (this->_gen == 0) return;
#endif

		this->_pop.clear();

		for (size_t i = 0; i < res_x; ++i)
		{
			for (size_t j = 0; j < res_y; ++j)
			{
				if (_array[i][j])
				{
					this->_pop.push_back(_array[i][j]);
				}
			}
		}

		pop_t ptmp, p_parents;
		for (size_t i = 0; i < Params::pop::size / 2; ++i)
		{
			indiv_t p1 = _selection(this->_pop);
			indiv_t p2 = _selection(this->_pop);
			boost::shared_ptr<Phen> i1, i2;
			p1->cross(p2, i1, i2);
			i1->mutate();
			i2->mutate();

			/*
			 Phenotypes are to be developed in eval() called below
			 this->_eval.eval(ptmp, 0, ptmp.size());
			 So no need to develop them here.
			 // i1->develop();
			 // i2->develop();
			 */

			// Add the generation when these two new organisms are created (mutated)
			i1->set_created_gen(this->_gen);
			i2->set_created_gen(this->_gen);

			ptmp.push_back(i1);
			ptmp.push_back(i2);
			p_parents.push_back(p1);
			p_parents.push_back(p2);
		}

		this->_eval.eval(ptmp, 0, ptmp.size());

		assert(ptmp.size() == p_parents.size());

		for (size_t i = 0; i < ptmp.size(); ++i)
		{
			_add_to_archive(ptmp[i], p_parents[i]);
		}
	}

	const array_t& archive() const
	{	return _array;}
	const array_t& parents() const
	{	return _array_parents;}

	const unsigned long jumps() const
	{	return _jumps;}

protected:
	array_t _array;
	array_t _prev_array;
	array_t _array_parents;
	unsigned long _jumps;

	bool _add_to_archive(indiv_t i1, indiv_t parent)
	{
		bool added = false;	// Flag raised when the individual is added to the archive in any cell

		// We have a map of 1x1000 for the total of 1000 categories
		assert(1 == res_x);
		assert(i1->fit().desc().size() == res_y);

		// Compare this individual with every top individual in every cell.
		// If this individual is better, replace the current cell occupant with it.
		for (int x = 0; x < res_x; ++x)
		{
			for (int y = 0; y < res_y; ++y)
			{
				float i1_fitness = i1->fit().value(y);

				if (!_array[x][y] || i1_fitness > _array[x][y]->fit().value(y))
				{
					// Replace the current cell occupant with new individual and its parent
					_array[x][y] = i1;
					_array_parents[x][y] = parent;

					added = true;

					// Record a jump of an indiv to a cell
					// One indiv could jump to many cells
					_jumps++;
				}
			}
		}

		return added;
	}

	indiv_t _selection(const pop_t& pop)
	{
		int x1 = misc::rand< int > (0, pop.size());
		return pop[x1];
	}

};
}
}
#endif

