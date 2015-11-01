#ifndef STAT_MAP_IMAGE_HPP
#define STAT_MAP_IMAGE_HPP

#include <numeric>
#include <boost/multi_array.hpp>
#include <sferes/stat/stat.hpp>
#include <glog/logging.h>
#include <boost/filesystem.hpp>

namespace sferes
{
  namespace stat
  {
    SFERES_STAT(MapImage, Stat)
    {
    public:
      typedef boost::shared_ptr<Phen> phen_t;
      typedef boost::multi_array<phen_t, 2> array_t;
      typedef boost::array<float, 2> point_t;

      MapImage() : _xs(0), _ys(0) {}
      template<typename E>
			void refresh(const E& ea)
			{
				_archive.clear();
				_xs = ea.archive().shape()[0];
				_ys = ea.archive().shape()[1];
				assert(_xs == Params::ea::res_x);
				assert(_ys == Params::ea::res_y);

				for (size_t i = 0; i < _xs; ++i)
				for (size_t j = 0; j < _ys; ++j)
				{
					phen_t p = ea.archive()[i][j];
					_archive.push_back(p);
				}

				// Report current generation every 10 generations
				if (ea.gen() % 10 == 0)
				{
					std::cout << "gen.. " << ea.gen() << std::endl;
				}

				if (ea.gen() % Params::pop::dump_period == 0)
				{
					_write_archive(ea.archive(), ea.parents(), std::string("archive_"), ea, ea.gen());

			#ifdef MAP_WRITE_PARENTS
					_write_parents(ea.archive(), ea.parents(), std::string("parents_"), ea);
			#endif
				}
			}

      const std::vector<phen_t>& getPopulation() const
			{
				return _archive;
			}

    	const std::vector<phen_t>& archive() const
    	{
    		return _archive;
    	}

      void show(std::ostream& os, size_t k)
			{
				std::cerr << "loading "<< k / _ys << "," << k % _ys << std::endl;
				if (_archive[k])
				{
					_archive[k]->develop();
					_archive[k]->show(os);
					_archive[k]->fit().set_mode(fit::mode::view);
					_archive[k]->fit().eval(*_archive[k]);
				}
				else
				std::cerr << "Warning, no point here" << std::endl;
			}

      template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
			{
				ar & BOOST_SERIALIZATION_NVP(_archive);
				ar & BOOST_SERIALIZATION_NVP(_xs);
				ar & BOOST_SERIALIZATION_NVP(_ys);
			}

    protected:
      std::vector<phen_t> _archive;
      int _xs, _ys;

      template<typename EA>
      void _write_parents(const array_t& array,
			  const array_t& p_array,
			  const std::string& prefix,
			  const EA& ea) const
			{
				std::cout << "writing..." << prefix << ea.gen() << std::endl;
				std::string fname = ea.res_dir() + "/"
				+ prefix
				+ boost::lexical_cast<
				std::string>(ea.gen())
				+ std::string(".dat");
				std::ofstream ofs(fname.c_str());
				for (size_t i = 0; i < _xs; ++i)
				for (size_t j = 0; j < _ys; ++j)
				if (array[i][j] && p_array[i][j])
				{
					point_t p = _get_point(p_array[i][j]);
					size_t x = round(p[0] * _xs);
					size_t y = round(p[1] * _ys);
					ofs << i / (float) _xs
					<< " " << j / (float) _ys
					<< " " << p_array[i][j]->fit().value()
					<< " " << x / (float) _xs
					<< " " << y / (float) _ys
					<< " " << array[i][j]->fit().value()
					<< std::endl;
				}
			}

      std::string _make_gen_dir(const std::string& res_dir, const int gen) const
      {
      	std::string gen_dir = res_dir + std::string("/map_gen_") + boost::lexical_cast<std::string>(gen);
				boost::filesystem::path my_path(gen_dir);
				boost::filesystem::create_directory(my_path);

				return gen_dir;
      }

      template<typename EA>
      void _write_archive(const array_t& array,
      	const array_t& p_array,
			  const std::string& prefix,
			  const EA& ea,
			  const int gen) const
			{
				std::cout << "writing..." << prefix << ea.gen() << std::endl;
				std::string fname = ea.res_dir() + "/"
				+ prefix
				+ boost::lexical_cast<
				std::string>(ea.gen())
				+ std::string(".dat");

				std::ofstream ofs(fname.c_str());
				for (size_t i = 0; i < _xs; ++i)
				{
					for (size_t j = 0; j < _ys; ++j)
					{
						if (array[i][j])
						{
							float fitness = array[i][j]->fit().value(j);

							ofs
							<< " " << j		// This dimension is categorical (1-1000). No need to normalize to be [0, 1].
							<< " " << fitness;

							// CPPN genome info
//							<< " " << array[i][j]->gen().get_nb_neurons()
//							<< " " << array[i][j]->gen().get_nb_connections();

							if (Params::image::record_lineage)
							{
								ofs << " " << array[i][j]->id();			// Record the id of this organism

								// Only print out the parent if this is a newly created organism
								if (array[i][j]->created_gen() == ea.gen())
								{
									ofs << " " << p_array[i][j]->id();		// Record the id of this organism's parent
								}
							}

							ofs << std::endl;	// End of line

							bool dump_map = false;

							// Always print the map in the first generation and last generation
//							if (
//									(gen == 0 || gen == Params::pop::nb_gen - Params::pop::dump_period)
									// Print out only when there is an improvement of 0.1
//									|| (fitness - p_array[i][j]->fit().value(j) >= 0.3)
//									)
							{
								dump_map = true;
							}

							// Check if we should print out
							if (dump_map)
							{
								// Create the directory
								const std::string gen_dir = _make_gen_dir(ea.res_dir(), gen);

								// Print out images at the current generation
								std::string image_gen = boost::lexical_cast<std::string>(ea.gen());
								std::string category = boost::lexical_cast<std::string>(j);
								std::string image_fitness = boost::lexical_cast<std::string>(fitness);
								std::string image_file = gen_dir + "/map_" + image_gen + "_" + category + "_" + image_fitness;

								array[i][j]->log_best_image_fitness(image_file);
							}
						}
					}
				}
			}

    };
  }
}

#endif
