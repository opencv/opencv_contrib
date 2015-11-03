/*
 * global_options.hpp
 *
 *  Created on: Aug 14, 2014
 *      Author: Joost Huizinga
 *
 *  Header file created to allow for custom command-line options to be added to an experiment.
 *  Requires you to replace run_ea with options::run_ea to works.
 *  Options can be added by:
 *  options::add()("option_name","description");
 */

#ifndef GLOBAL_OPTIONS_HPP_
#define GLOBAL_OPTIONS_HPP_

#include <boost/program_options.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/foreach.hpp>

#include <sferes/eval/parallel.hpp>
//#include <sferes/dbg/dbg.hpp>

namespace sferes
{
	namespace options
	{

		boost::program_options::variables_map vm;

		template<typename Ea>
		static void run_ea ( int argc, char **argv, Ea& ea,
				const boost::program_options::options_description& add_opts =
						boost::program_options::options_description(),
				bool init_rand = true )
		{
			namespace po = boost::program_options;
			std::cout << "sferes2 version: " << VERSION << std::endl;
			if (init_rand)
			{
				time_t t = time(0) + ::getpid();
				std::cout << "seed: " << t << std::endl;
				srand(t);
			}
			po::options_description desc("Allowed sferes2 options");
			desc.add(add_opts);
			desc.add_options()("help,h", "produce help message")("stat,s",
					po::value<int>(), "statistic number")("out,o",
					po::value<std::string>(), "output file")("number,n", po::value<int>(),
					"number in stat")("load,l", po::value<std::string>(),
					"load a result file")("verbose,v",
					po::value<std::vector<std::string> >()->multitoken(),
					"verbose output, available default streams : all, ea, fit, phen, trace");

		//		po::variables_map vm;
			po::store(po::parse_command_line(argc, argv, desc), vm);
			po::notify(vm);

			if (vm.count("help"))
			{
				std::cout << desc << std::endl;
				return;
			}
			if (vm.count("verbose"))
			{
				dbg::init();
				std::vector < std::string > streams = vm["verbose"].as<
						std::vector<std::string> >();
				attach_ostream(dbg::warning, std::cout);
				attach_ostream(dbg::error, std::cerr);
				attach_ostream(dbg::info, std::cout);
				bool all = std::find(streams.begin(), streams.end(), "all")
						!= streams.end();
				bool trace = std::find(streams.begin(), streams.end(), "trace")
						!= streams.end();
				if (all)
				{
					streams.push_back("ea");
					streams.push_back("fit");
					streams.push_back("phen");
					streams.push_back("eval");
				}
				BOOST_FOREACH(const std::string& s, streams){
				dbg::enable(dbg::all, s.c_str(), true);
				dbg::attach_ostream(dbg::info, s.c_str(), std::cout);
				if (trace)
				dbg::attach_ostream(dbg::tracing, s.c_str(), std::cout);
			}
				if (trace)
					attach_ostream(dbg::tracing, std::cout);
			}

			parallel::init();
			if (vm.count("load"))
			{
				ea.load(vm["load"].as<std::string>());

				if (!vm.count("out"))
				{
					std::cerr << "You must specifiy an out file" << std::endl;
					return;
				}
				else
				{
					int stat = 0;
					int n = 0;
					if (vm.count("stat"))
						stat = vm["stat"].as<int>();
					if (vm.count("number"))
						n = vm["number"].as<int>();
					std::ofstream ofs(vm["out"].as<std::string>().c_str());
					ea.show_stat(stat, ofs, n);
				}
			}
			else
				ea.run();
		}

	}
}
#endif /* GLOBAL_OPTIONS_HPP_ */
