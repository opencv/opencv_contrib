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


#ifndef EA_CUSTOM_HPP_
#define EA_CUSTOM_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <sferes/ea/ea.hpp>

namespace sferes {
  namespace ea {

  	SFERES_EA(EaCustom, Ea) {
  	protected:
  		std::string _gen_file_path;

  	public:
  		EaCustom () : _gen_file_path("")
			{
				this->_make_res_dir();
			}

			void _make_res_dir()
			{
				if (Params::pop::dump_period == -1)
				{
					return;
				}

				// Delete the unused folder by Ea
				std::string to_delete = misc::hostname() + "_" + misc::date() + "_" + misc::getpid();

				if (boost::filesystem::is_directory(to_delete) && boost::filesystem::is_empty(to_delete))
				{
					boost::filesystem::remove(to_delete);
				}

				// Check if such a folder already exists
				this->_res_dir = "mmm";	// Only one folder regardless which platform the program is running on

				boost::filesystem::path my_path(this->_res_dir);

				// Create a new folder if it doesn't exist
				if (!boost::filesystem::exists(boost::filesystem::status(my_path)))
				{
					// Create a new folder if it does not exist
					boost::filesystem::create_directory(my_path);
				}
				// Run experiment from that folder
				else
				{
					std::vector<std::string> gens;

					// The file to find
					int max = 0;

					// Find a gen file
					for(boost::filesystem::directory_entry& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(my_path), {}))
					{
						// Find out if '/gen_' exists in the filename
						std::string e = entry.path().string();

						std::string prefix = this->_res_dir + "/gen_";
						size_t found = e.find(prefix);

						if (found != std::string::npos)
						{
							// Extract out the generation number
							std::string number = std::string(e).replace(found, prefix.length(), "");

							// Remove double quotes
//							number = boost::replace_all_copy(number, "\"", "");.string()

							int gen = boost::lexical_cast<int>(number);
							if (gen > max)
							{
								max = gen;
								_gen_file_path = e;
							}

						} // end if
					} // end for-loop

					// Start run from that gen file
//					_continue_run = boost::filesystem::current_path().string() + "/" + _continue_run;
					std::cout << "[A]: " <<  _gen_file_path << "\n";
				}
			}
  	};
  }
}
#endif
