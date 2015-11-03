#include <iostream>

#ifdef MPI_ENABLED
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/rank_simple.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/mean_fit.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#include <boost/program_options.hpp>
#include <sferes/eval/mpi.hpp>

using namespace sferes;
using namespace sferes::gen::evo_float;

struct Params
{
  struct evo_float
  {
    SFERES_CONST float cross_rate = 0.5f;
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 10.0f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
  };
  struct pop
  {
    SFERES_CONST unsigned size = 200;
    SFERES_CONST unsigned nb_gen = 40;
    SFERES_CONST int dump_period = 5;
    SFERES_CONST int initial_aleat = 1;
    SFERES_CONST float coeff = 1.1f;
    SFERES_CONST float keep_rate = 0.6f;    
  };
  struct parameters
  {
    SFERES_CONST float min = -10.0f;
    SFERES_CONST float max = 10.0f;
  };
};

SFERES_FITNESS(FitTest, sferes::fit::Fitness)
{
 public:
  FitTest()
    {}
  template<typename Indiv>
    void eval(const Indiv& ind) 
  {
    float v = 0;
    for (unsigned i = 0; i < ind.size(); ++i)
      {
	float p = ind.data(i);
	v += p * p * p * p;
      }
    // slow down to simulate a slow fitness
    usleep(1e4);
    this->_value = -v;
  }
};



int main(int argc, char **argv)
{
  dbg::out(dbg::info)<<"running ex_ea ... try --help for options (verbose)"<<std::endl;
  std::cout << "To run this example, you need to use mpirun" << std::endl;
  std::cout << "mpirun -x LD_LIBRARY_PATH=/home/creadapt/lib  -np 50 -machinefile machines.pinfo build/debug/examples/ex_ea_mpi" << std::endl;
  typedef gen::EvoFloat<10, Params> gen_t;
  typedef phen::Parameters<gen_t, FitTest<Params>, Params> phen_t;
  typedef eval::Mpi<Params> eval_t;
  typedef boost::fusion::vector<stat::BestFit<phen_t, Params>, stat::MeanFit<Params> >  stat_t;
  typedef modif::Dummy<> modifier_t;
  typedef ea::RankSimple<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
  ea_t ea;

  run_ea(argc, argv, ea);

  std::cout<<"==> best fitness ="<<ea.stat<0>().best()->fit().value()<<std::endl;
//   std::cout<<"==> mean fitness ="<<ea.stat<1>().mean()<<std::endl;
  return 0;
}
#else
#warning MPI is disabled, ex_ea_mpi is not compiled
int main()
{
  std::cerr<<"MPI is disabled"<<std::endl;
  return 0;
}
#endif
