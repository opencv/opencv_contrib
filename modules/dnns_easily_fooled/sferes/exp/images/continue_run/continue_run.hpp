/*
 * continue_run.hpp
 *
 *  Created on: Aug 14, 2014
 *      Author: joost
 */

#ifndef CONTINUE_RUN_HPP_
#define CONTINUE_RUN_HPP_

#include <boost/fusion/container.hpp>
#include <boost/fusion/include/vector.hpp>

#include <sferes/dbg/dbg.hpp>

#include "global_options.hpp"
#include <exp/images/stat/stat_map_image.hpp>

namespace sferes
{
	namespace cont
	{

		template<typename EAType, typename Params>
		class Continuator
		{

		public:
			typedef std::vector<typename EAType::indiv_t> pop_t;

			bool enabled()
			{
				return options::vm.count("continue");
			}

			pop_t getPopulationFromFile(EAType& ea)
			{
				ea.load(options::vm["continue"].as<std::string>());

				return boost::fusion::at_c<Params::cont::getPopIndex>(ea.stat()).getPopulation();
			}

			pop_t getPopulationFromFile(EAType& ea, const std::string& path_gen_file)
			{
				ea.load(path_gen_file);

				return boost::fusion::at_c<Params::cont::getPopIndex>(ea.stat()).getPopulation();
			}

			void run_with_current_population(EAType& ea, const std::string filename)
			{
				// Read the number of generation from gen file. Ex: gen_450
				int start = 0;
				std::string gen_prefix("gen_");
				std::size_t pos = filename.rfind(gen_prefix) + gen_prefix.size();
				std::string gen_number = filename.substr(pos);
				std::istringstream ss(gen_number);
				ss >> start;
				start++;
				dbg::out(dbg::info, "continue") << "File name: " << filename << " number start: " << pos << " gen number: " << gen_number << " result: " << start << std::endl;

				// Similar to the run() function in <sferes/ea/ea.hpp>
				for (int _gen = start; _gen < Params::pop::nb_gen; ++_gen)
				{
					ea.setGen(_gen);
					ea.epoch();
					ea.update_stats();
					if (_gen % Params::pop::dump_period == 0)
					{
						ea.write();
					}
				}

				std::cout << "Finished all the runs.\n";
				exit(0);
			}

			void run_with_current_population(EAType& ea)
			{
				const std::string filename = options::vm["continue"].as<std::string>();
				run_with_current_population(ea, filename);
			}

		};

	}
}
#endif /* CONTINUE_RUN_HPP_ */
