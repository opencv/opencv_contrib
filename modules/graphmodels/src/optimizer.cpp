#include "optimizer.hpp"
#include <sstream>
#include <iostream>

using namespace std;

namespace cv
{
namespace graphmodels
{

Optimizer* Optimizer::ChooseOptimizer(const config::Optimizer& config) {
  Optimizer* opt = NULL;
  switch (config.optimizer_type()) {
    case config::Optimizer::STOCHASTIC_GRADIENT_DESCENT :
      opt = new SGDOptimizer(config);
      break;
    case config::Optimizer::LBFGS :
      opt = new LBFGSOptimizer(config);
      break;
    default:
      cerr << "Undefined optimizer." << endl;
      exit(1);
  }
  return opt;
}

Optimizer::Optimizer(const config::Optimizer& optimizer_config) :
  epsilon_decay_type_(optimizer_config.epsilon_decay()),
  epsilon_(optimizer_config.epsilon()),
  minimum_epsilon_(optimizer_config.minimum_epsilon()),
  epsilon_decay_timescale_(optimizer_config.epsilon_decay_timescale()),
  start_optimization_after_(optimizer_config.start_optimization_after()),
  l2_decay_(optimizer_config.l2_decay()),
  weight_norm_limit_(optimizer_config.weight_norm_limit()),  // impose upper limit.
  weight_norm_constraint_(optimizer_config.weight_norm_constraint()),  // impose equality.
  step_(0) 
{}

Optimizer::~Optimizer() {}
void Optimizer::AllocateMemory(const int rows, const int cols) {}
void Optimizer::LoadParameters(hid_t file, const string& prefix) {}
void Optimizer::SaveParameters(hid_t file, const string& prefix) {}
void Optimizer::ReduceLearningRate(float factor) {
  epsilon_ *= factor;
}

void Optimizer::ApplyConstraints(Matrix& parameter) {
  if (weight_norm_constraint_ > 0) {  // Make the norm of the incoming weights have this value.
    parameter.NormLimitByAxis(1, weight_norm_constraint_, true);
  } else if (weight_norm_limit_ > 0) {  // Limit the norm of the incoming weights to this value.
    parameter.NormLimitByAxis(1, weight_norm_limit_, false);
  }
}

float Optimizer::GetDecayedEpsilon() const {
  float decayed_epsilon = epsilon_;
  if (epsilon_decay_timescale_ > 0 &&
      epsilon_decay_timescale_ != config::Optimizer::NONE) {
    float f = ((float)step_) / epsilon_decay_timescale_;
    if (epsilon_decay_type_ == config::Optimizer::EXPONENTIAL) {
      decayed_epsilon = epsilon_ * exp(-f);
    } else if (epsilon_decay_type_ == config::Optimizer::INVERSE_T) {
     decayed_epsilon = epsilon_ / (1 + f);
    } else if (epsilon_decay_type_ == config::Optimizer::LINEAR) {
      decayed_epsilon = (f < 1) ? (epsilon_ * (1-f) + minimum_epsilon_ * f)
                                  : minimum_epsilon_;
    } else {
      cerr << "Unknown epsilon decay rule." << endl;
      exit(1);
    }
  }
  if (decayed_epsilon < minimum_epsilon_) decayed_epsilon = minimum_epsilon_;
  return decayed_epsilon;
}

SGDOptimizer::SGDOptimizer(const config::Optimizer& optimizer_config) :
  Optimizer(optimizer_config),
  gradient_clip_(optimizer_config.gradient_clip()),
  initial_momentum_(optimizer_config.initial_momentum()),
  final_momentum_(optimizer_config.final_momentum()),
  momentum_transition_timescale_(
      optimizer_config.momentum_transition_timescale()) {}

void SGDOptimizer::AllocateMemory(const int rows, const int cols) {
  gradient_history_.AllocateGPUMemory(rows, cols);
}

void SGDOptimizer::LoadParameters(hid_t file, const string& prefix) {
  stringstream ss;
  ss << prefix << "_" << "gradient_history";
  gradient_history_.ReadHDF5(file, ss.str());
  ss.str("");
  ss << prefix << "_" << "step";
  ReadHDF5IntAttr(file, ss.str(), &step_);
}

void SGDOptimizer::SaveParameters(hid_t file, const string& prefix) {
  stringstream ss;
  ss << prefix << "_" << "gradient_history";
  gradient_history_.WriteHDF5(file, ss.str());
  ss.str("");
  ss << prefix << "_" << "step";
  WriteHDF5IntAttr(file, ss.str(), &step_);
}

float SGDOptimizer::GetMomentum() const {
  if (momentum_transition_timescale_ > 0) {
    return initial_momentum_ + (final_momentum_ - initial_momentum_) * 
      (1 - exp(-((float)step_)/momentum_transition_timescale_));
  } else {
    return final_momentum_;
  }
}

void SGDOptimizer::Optimize(Matrix& gradient, Matrix& parameter) {
  if (step_ >= start_optimization_after_) {
    float epsilon = GetDecayedEpsilon(), momentum = GetMomentum();

    gradient_history_.Mult(momentum);
   
    // L2 decay.
    if (l2_decay_ > 0) gradient_history_.Add(parameter, l2_decay_);
   
    // Clip gradients to prevent explosions.
    if (gradient_clip_ > 0) gradient.UpperBoundMod(gradient_clip_);

    gradient_history_.Add(gradient, 1.0);
    parameter.Add(gradient_history_, -epsilon);
    ApplyConstraints(parameter);
  }
  step_++;
}

LBFGSOptimizer::LBFGSOptimizer(const config::Optimizer& optimizer_config) :
  Optimizer(optimizer_config),
  m_(optimizer_config.lbfgs_memory()),
  start_(0) {
  if (m_ <= 0) {
    cerr << "LBFGS has zero memory!" << endl;
    exit(1);
  }
  rho_.resize(m_);
  alpha_.resize(m_);
  beta_.resize(m_);
  s_.resize(m_);
  y_.resize(m_);
}

void LBFGSOptimizer::AllocateMemory(const int rows, const int cols) {
  q_.AllocateGPUMemory(rows, cols);
  last_w_.AllocateGPUMemory(rows, cols);
  last_q_.AllocateGPUMemory(rows, cols);
  last_w_.FillWithRandn();
  last_q_.FillWithRandn();
  for (int i = 0; i < m_; i++) {
    s_[i].AllocateGPUMemory(rows, cols);
    y_[i].AllocateGPUMemory(rows, cols);
  }
}

void LBFGSOptimizer::SaveParameters(hid_t file, const string& prefix) {
  last_w_.WriteHDF5(file, prefix + "_lbfgs_w");
  last_q_.WriteHDF5(file, prefix + "_lbfgs_q");
  stringstream ss;
  for (int j = 0; j < m_; j++) {
    ss << prefix << "_lbfgs_s_" << j << endl;
    s_[j].WriteHDF5(file, ss.str());
    ss.str("");
    ss << prefix << "_lbfgs_y_" << j << endl;
    y_[j].WriteHDF5(file, ss.str());
    ss.str("");
  }
  WriteHDF5IntAttr(file, prefix + "_lbfgs_start", &start_);
}

void LBFGSOptimizer::LoadParameters(hid_t file, const string& prefix) {
  last_w_.ReadHDF5(file, prefix + "_lbfgs_w");
  last_q_.ReadHDF5(file, prefix + "_lbfgs_q");
  stringstream ss;
  for (int j = 0; j < m_; j++) {
    ss << prefix << "_lbfgs_s_" << j << endl;
    s_[j].ReadHDF5(file, ss.str());
    ss.str("");
    ss << prefix << "_lbfgs_y_" << j << endl;
    y_[j].ReadHDF5(file, ss.str());
    ss.str("");
  }
  ReadHDF5IntAttr(file, prefix + "_lbfgs_start", &start_);
}


// Modifies parameter.
void LBFGSOptimizer::Optimize(Matrix& gradient, Matrix& parameter) {
  q_.Set(gradient);

  if (l2_decay_ > 0) q_.Add(parameter, l2_decay_);

  // Update memory.
  parameter.Subtract(last_w_, s_[start_]);
  q_.Subtract(last_q_, y_[start_]);
  float norm = s_[start_].VDot(y_[start_]);
  if (norm == 0) {
    cerr<< "Error: Norm was 0." << endl;
    exit(1);
  }
  rho_[start_] = 1 / norm;
  last_w_.Set(parameter);
  last_q_.Set(q_);

  // Compute update.
  int i;
  for (int j = 0; j < m_; j++) {
    i = start_ - j;
    if (i < 0) i += m_;
    alpha_[i] = rho_[i] * q_.VDot(s_[i]);
    q_.Add(y_[i], -alpha_[i]);
  }
  float h = y_[start_].VDot(s_[start_]) / y_[start_].VDot(y_[start_]);
  q_.Mult(h);
  for (int j = m_ - 1; j >= 0; j--) {
    i = start_ - j;
    if (i < 0) i += m_;
    beta_[i] = rho_[i] * y_[i].VDot(q_);
    q_.Add(s_[i], alpha_[i] - beta_[i]);
  }
  float epsilon = GetDecayedEpsilon();  // TODO:line search here.
  parameter.Add(q_, -epsilon);
  //w.Add(q_, -1);
  ApplyConstraints(parameter);

  start_ = (start_ + 1) % m_;
  step_++;
}

}
}
