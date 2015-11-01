#ifndef STAT_MAP_HPP_
#define STAT_MAP_HPP_

#include <numeric>
#include <boost/multi_array.hpp>
#include <sferes/stat/stat.hpp>

namespace sferes
{
  namespace stat
  {
    SFERES_STAT(Map, Stat)
    {
    public:
      typedef boost::shared_ptr<Phen> phen_t;
      typedef boost::multi_array<phen_t, 2> array_t;
      typedef boost::array<float, 2> point_t;

      Map() : _xs(0), _ys(0) {}
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

	if (ea.gen() % Params::pop::dump_period == 0)
	  {
	    _write_archive(ea.archive(), std::string("archive_"), ea);
#ifdef MAP_WRITE_PARENTS
	    _write_parents(ea.archive(), ea.parents(), std::string("parents_"), ea);
#endif
	  }
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
	std::string fname =  ea.res_dir() + "/"
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

      template<typename EA>
      void _write_archive(const array_t& array,
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
	    if (array[i][j])
	      {
		ofs << i / (float) _xs
		    << " " << j / (float) _ys
		    << " " << array[i][j]->fit().value()
		    << std::endl;
	      }
      }


    };
  }
}

#endif
