// terminate_action::operator() is licensed under the following license.
//
// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, H. Strasdat, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "terminate_action.hpp"

#include <g2o/core/sparse_optimizer.h>

namespace cv::slam {
namespace optimize {

g2o::HyperGraphAction* terminate_action::operator()(
    const g2o::HyperGraph* graph, Parameters* parameters) {
    assert(dynamic_cast<const g2o::SparseOptimizer*>(graph) && "graph is not a SparseOptimizer");
    assert(dynamic_cast<g2o::HyperGraphAction::ParametersIteration*>(parameters) && "error casting parameters");

    const g2o::SparseOptimizer* optimizer = static_cast<const g2o::SparseOptimizer*>(graph);
    g2o::HyperGraphAction::ParametersIteration* params = static_cast<g2o::HyperGraphAction::ParametersIteration*>(parameters);

    const_cast<g2o::SparseOptimizer*>(optimizer)->computeActiveErrors();
    if (params->iteration < 0) {
        // let the optimizer run for at least one iteration
        // Hence, we reset the stop flag
        setOptimizerStopFlag(optimizer, false);
        stopped_by_terminate_action_ = false;
    }
    else if (params->iteration == 0) {
        // first iteration, just store the chi2 value
        _lastChi = optimizer->activeRobustChi2();
    }
    else {
        // compute the gain and stop the optimizer in case the
        // gain is below the threshold or we reached the max
        // number of iterations
        bool stopOptimizer = false;
        if (params->iteration < _maxIterations) {
            auto currentChi = optimizer->activeRobustChi2();
            auto gain = (_lastChi - currentChi) / currentChi;
            _lastChi = currentChi;
            if (gain >= 0 && gain < _gainThreshold)
                stopOptimizer = true;
        }
        else {
            stopOptimizer = true;
        }
        if (stopOptimizer) { // tell the optimizer to stop
            setOptimizerStopFlag(optimizer, true);
            stopped_by_terminate_action_ = true;
        }
    }
    return this;
}

} // namespace optimize
} // namespace cv::slam
