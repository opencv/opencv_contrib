#include <iostream>
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/eps_moea.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#include <boost/program_options.hpp>

using namespace sferes;
using namespace sferes::gen::evo_float;

struct Params
{
  struct evo_float
  {

    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.5f;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 10.0f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
  };
  struct pop
  {
    SFERES_CONST unsigned size = 200;
    SFERES_CONST int dump_period = 50;
    SFERES_ARRAY(float, eps, 0.0075f, 0.0075f);
    SFERES_ARRAY(float, min_fit, 0.0f, 0.0f);
    SFERES_CONST size_t grain = size / 4;
    SFERES_CONST unsigned nb_gen = 2000;
  };

  struct parameters
  {
    SFERES_CONST float min = 0.0f;
    SFERES_CONST float max = 1.0f;
  };
};


template<typename Indiv>
float _g(const Indiv &ind)
{
  float g = 0.0f;
  assert(ind.size() == 30);
  for (size_t i = 1; i < 30; ++i)
    g += ind.data(i);
  g = 9.0f * g / 29.0f;
  g += 1.0f;
  return g;
}

SFERES_FITNESS(FitZDT2, sferes::fit::Fitness)
{
 public:
  FitZDT2()  {}
  template<typename Indiv>
    void eval(Indiv& ind) 
  {
    this->_objs.resize(2);
    float f1 = ind.data(0);
    float g = _g(ind);
    float h = 1.0f - pow((f1 / g), 2.0);
    float f2 = g * h;
    this->_objs[0] = -f1;
    this->_objs[1] = -f2;
  }
};




int main(int argc, char **argv)
{
  std::cout<<"running "<<argv[0]<<" ... try --help for options (verbose)"<<std::endl;

  typedef gen::EvoFloat<30, Params> gen_t;
  typedef phen::Parameters<gen_t, FitZDT2<Params>, Params> phen_t;
  typedef eval::Eval<Params> eval_t;
  typedef boost::fusion::vector<stat::ParetoFront<phen_t, Params> >  stat_t;
  typedef modif::Dummy<> modifier_t;
  typedef ea::EpsMOEA<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
  ea_t ea;

  run_ea(argc, argv, ea);

  return 0;
}
