// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "craam/definitions.hpp"
#include <memory>

namespace craam {

#ifdef GUROBI_USE

#include <gurobi_c++.h>

using uint = unsigned int;

/// Represents different Gurobi environments depending on the purpose
/// for which it is being used
enum class OptimizerType : uint {
    /// Used when computing Bellman updates with robust optimization
    NatureUpdate = 0,
    /// Linear programming MDP solution (and similar)
    LinearProgramMDP = 1,
    /// Mixed Integer and Nonconvex formulations
    NonconvexOptimization = 2,
    /// Other use: Must be the last item with the highest index
    Other = 3
};

/**
 * Constructs a static instance of the gurobi object.
 * The returned object should not be used concurrently,
 * but is definitely being used that way!
 *
 * The construction is also not thread-safe, but seems to work.
 *
 * @param What purpose the optimizer is being used for. There is a different
 *        environment for each purpose.
 */
inline std::shared_ptr<GRBEnv> get_gurobi(OptimizerType purpose) {
    using SPG = std::shared_ptr<GRBEnv>;
    static std::array<SPG, uint(OptimizerType::Other) + 1> envirs;

    SPG& env = envirs[uint(purpose)];
    if (env == nullptr) {
        try {
            env = std::make_shared<GRBEnv>();
        } catch (std::exception& e) {
            std::cerr << "Problem constructing Gurobi object: " << std::endl
                      << e.what() << std::endl;
            throw e;
        } catch (...) {
            std::cerr << "Unknown exception while creating a gurobi object. Could be a "
                         "license problem."
                      << std::endl;
            throw;
        }
        env->set(GRB_IntParam_OutputFlag, 0);
    }
    return env;
}
#endif

} // namespace craam
