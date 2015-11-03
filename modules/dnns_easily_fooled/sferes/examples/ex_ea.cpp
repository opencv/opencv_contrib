#include <iostream>
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/rank_simple.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/mean_fit.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>

using namespace sferes;
using namespace sferes::gen::evo_float;

struct Params
{
  struct evo_float
  {
    // we choose the polynomial mutation type
    SFERES_CONST mutation_t mutation_type = polynomial;
    // we choose the polynomial cross-over type
    SFERES_CONST cross_over_t cross_over_type = sbx;
    // the mutation rate of the real-valued vector
    SFERES_CONST float mutation_rate = 0.1f;
    // the cross rate of the real-valued vector
    SFERES_CONST float cross_rate = 0.5f;
    // a parameter of the polynomial mutation
    SFERES_CONST float eta_m = 15.0f;
    // a parameter of the polynomial cross-over
    SFERES_CONST float eta_c = 10.0f;
  };
  struct pop
  {
    // size of the population
    SFERES_CONST unsigned size = 200;
    // number of generations
    SFERES_CONST unsigned nb_gen = 2000;
    // how often should the result file be written (here, each 5
    // generation)
    SFERES_CONST int dump_period = 5;
    // how many individuals should be created during the random
    // generation process?
    SFERES_CONST int initial_aleat = 1;
    // used by RankSimple to select the pressure
    SFERES_CONST float coeff = 1.1f;
    // the number of individuals that are kept from on generation to
    // another (elitism)
    SFERES_CONST float keep_rate = 0.6f;
  };
  struct parameters
  {
    // maximum value of parameters
    SFERES_CONST float min = -10.0f;
    // minimum value
    SFERES_CONST float max = 10.0f;
  };
};

SFERES_FITNESS(FitTest, sferes::fit::Fitness)
{
 public:
  // indiv will have the type defined in the main (phen_t)
  template<typename Indiv>
    void eval(const Indiv& ind)
  {
    float v = 0;
    for (unsigned i = 0; i < ind.size(); ++i)
      {
	float p = ind.data(i);
	v += p * p * p * p;
      }
    this->_value = -v;
  }
};



int main(int argc, char **argv)
{
  // Our fitness is the class FitTest (see above), that we will call
  // fit_t. Params is the set of parameters (struct Params) defined in
  // this file.
  typedef FitTest<Params> fit_t;
  // We define the genotype. Here we choose EvoFloat (real
  // numbers). We evolve 10 real numbers, with the params defined in
  // Params (cf the beginning of this file)
  typedef gen::EvoFloat<10, Params> gen_t;
  // This genotype should be simply transformed into a vector of
  // parameters (phen::Parameters). The genotype could also have been
  // transformed into a shape, a neural network... The phenotype need
  // to know which fitness to use; we pass fit_t.
  typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
  // The evaluator is in charge of distributing the evaluation of the
  // population. It can be simple eval::Eval (nothing special),
  // parallel (for multicore machines, eval::Parallel) or distributed
  // (for clusters, eval::Mpi).
  typedef eval::Eval<Params> eval_t;
  // Statistics gather data about the evolutionary process (mean
  // fitness, Pareto front, ...). Since they can also stores the best
  // individuals, they are the container of our results. We can add as
  // many statistics as required thanks to the boost::fusion::vector.
  typedef boost::fusion::vector<stat::BestFit<phen_t, Params>, stat::MeanFit<Params> >  stat_t;
  // Modifiers are functors that are run once all individuals have
  // been evalutated. Their typical use is to add some evolutionary
  // pressures towards diversity (e.g. fitness sharing). Here we don't
  // use this feature. As a consequence we use a "dummy" modifier that
  // does nothing.
  typedef modif::Dummy<> modifier_t;
  // We can finally put everything together. RankSimple is the
  // evolutianary algorithm. It is parametrized by the phenotype, the
  // evaluator, the statistics list, the modifier and the general params.
  typedef ea::RankSimple<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
  // We now have a special class for our experiment: ea_t. The next
  // line instantiate an object of this class
  ea_t ea;
  // we can now process the comannd line options an run the
  // evolutionary algorithm (if a --load argument is passed, the file
  // is loaded; otherwise, the algorithm is launched).
  run_ea(argc, argv, ea);
  //
  return 0;
}
