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

#include "craam/Transition.hpp"
#include "craam/definitions.hpp"

#include <cmath>

namespace craam {

/**
 * A set of values that represent a solution to a plain MDP.
 *
 * @tparam PolicyType Type of the policy used (int deterministic, numvec
 * stochastic, but could also have multiple components (such as an action and
 * transition probability) )
 */
template <class PolicyType> struct Solution {
    /// Value function
    numvec valuefunction;
    /// Policy of the decision maker (and nature if applicable) for each state
    vector<PolicyType> policy;
    /// Bellman residual of the computation
    prec_t residual;
    /// Number of iterations taken
    long iterations;
    /// Time taken to solve the problem
    prec_t time;
    /// Status (0 means OK, 1 means timeout, 2 means internal error)
    int status;

    /// Constructs an empty solution with an invalid return value.
    ///
    /// WARNING: The default status of the solution is that it is a result
    /// of an internal error
    Solution()
        : valuefunction(0), policy(0), residual(-1), iterations(-1), time(std::nan("")),
          status(2) {}

    /// Empty solution for a problem with statecount states
    /// @param statecount Number of states in the final solution (0 means empty)
    /// @param status The status of the solution (0 means that the solution is optimal)
    Solution(size_t statecount, int status)
        : valuefunction(statecount, 0.0), policy(statecount), residual(-1),
          iterations(-1), time(nan("")), status(status) {}

    /// Empty solution for a problem with a given value function and policy
    Solution(numvec valuefunction, vector<PolicyType> policy, prec_t residual = -1,
             long iterations = -1, double time = nan(""), int status = 2)
        : valuefunction(move(valuefunction)), policy(move(policy)), residual(residual),
          iterations(iterations), time(time), status(status) {}

    /**
     * Computes the total return of the solution given the initial
     * distribution.
     * @param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const {
        if (initial.max_index() >= (long)valuefunction.size())
            throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};

/// A solution with a deterministic policy
/// The policy is:
///  1) index for the action
using DetermSolution = Solution<long>;

/// A solution to an MDP with a stochastic policy
/// The policy is:
///  1) distribution over actions
using RandSolution = Solution<numvec>;

/// Solution to an S,A rectangular robust problem to an MDP
/// The policy is:
///  1) index for the action
///  2) distribution over *reachable* states (with non-zero nominal probability)
using SARobustSolution = Solution<pair<long, numvec>>;

/// Solution to an S,A rectangular robust problem to an MDPO
/// The policy is:
///  1) index for the action
///  2) distribution over outcomes
using SARobustOutcomeSolution = Solution<pair<long, numvec>>;

/// Solution to an S-rectangular robust problem to an MDP
/// The policy is:
///  1) distribution over actions
///  2) list of distributions for all actions (not only the active ones)
///         over *reachable* states (with non-zero nominal probability)
using SRobustSolution = Solution<pair<numvec, numvecvec>>;

/// Solution to an S-rectangular robust problem to an MDP
/// The policy is:
///  1) distribution over actions
///  2) distribution over outcomes
using SRobustOutcomeSolution = Solution<pair<numvec, numvec>>;

} // namespace craam
