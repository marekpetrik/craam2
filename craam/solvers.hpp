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

#include "craam/algorithms/iteration_methods.hpp"

namespace craam {

// **************************************************************************
// Value iteration methods
// **************************************************************************

/**
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state
MDP. If the states are ordered correctly, one iteration is enough to compute the
optimal value function. Since the value function is updated from the last state
to the first one, the states should be ordered in the temporal order.

@param mdp The MDP to solve
@param discount Discount factor.
@param valuefunction Initial value function. Passed by value, because it is
modified. Optional, use all zeros when not provided. Ignored when size is 0.
@param policy Partial policy specification. Optimize only actions that are
policy[state] = -1
@param iterations Maximal number of iterations to run
@param maxresidual Stop when the maximal residual falls below this value.


@returns Solution that can be used to compute the total return, or the optimal
policy.
*/
template <class SType>
inline DeterministicSolution
solve_vi(const GMDP<SType>& mdp, prec_t discount, numvec valuefunction = numvec(0),
         const indvec& policy = indvec(0), unsigned long iterations = MAXITER,
         prec_t maxresidual = SOLPREC) {
    return vi_gs<SType, PlainBellman<SType>>(mdp, discount, move(valuefunction),
                                             PlainBellman<SType>(policy), iterations,
                                             maxresidual);
}

/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi *
iterations_vi
@param type Type of realization of the uncertainty
@param discount Discount factor
@param valuefunction Initial value function
@param policy Partial policy specification. Optimize only actions that are
policy[state] = -1
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops
below this threshold.
@param iterations_vi Maximal number of inner loop value iterations
@param maxresidual_vi Stop policy evaluation when the policy residual drops
below maxresidual_vi * last_policy_residual
@param print_progress Whether to report on progress during the computation
@return Computed (approximate) solution
 */
template <class SType>
inline DeterministicSolution
solve_mpi(const GRMDP<SType>& mdp, prec_t discount,
          const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
          unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
          bool print_progress = false) {

    return mpi_jac<SType, PlainBellman<SType>>(
        mdp, discount, valuefunction, PlainBellman<SType>(policy), iterations_pi,
        maxresidual_pi, iterations_vi, maxresidual_vi, print_progress);
}

// **************************************************************************
// Wrapper methods
// **************************************************************************

/**
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state
MDP. If the states are ordered correctly, one iteration is enough to compute
the optimal value function. Since the value function is updated from the last
state to the first one, the states should be ordered in the temporal order.

This is a simplified method interface. Use vi_gs with PolicyNature for full
functionality.

@param mdp      The MDP to solve
@param discount Discount factor.
@param nature   Response of nature, the function is the same for all states
and actions.
@param thresholds Parameters passed to nature response functions. One
value per state and then one value per action.
@param valuefunction Initial value function. Passed by value, because it is
modified. Optional, use all zeros when not provided. Ignored when size is 0.
@param policy  Partial policy specification. Optimize only actions that are
policy[state] = -1. Use policy length 0 to optimize all actions.
@param iterations Maximal number of iterations to run
@param maxresidual Stop when the maximal residual falls below this value.

\returns Solution that can be used to compute the total return, or the optimal
policy.
*/
template <class SType>
inline SARobustSolution
rsolve_vi(const GRMDP<SType>& mdp, prec_t discount, SANature nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC) {

    return vi_gs<SType, SARobustBellman<SType>>(
        mdp, discount, move(valuefunction), SARobustBellman<SType>(move(nature), policy),
        iterations, maxresidual);
}

/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

This is a simplified method interface. Use mpi_jac with PolicyNature for full
functionality.

WARNING: There is no proof of convergence for this method. This is not the
same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
the discussion in the paper on methods like this one (e.g. Seid, White)

Note that the total number of iterations will be bounded by iterations_pi *
iterations_vi
@param type Type of realization of the uncertainty
@param discount Discount factor
@param nature Response of nature; the same value is used for all states and
actions.
@param valuefunction Initial value function
@param policy Partial policy specification. Optimize only actions that are
policy[state] = -1
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops
below this threshold.
@param iterations_vi Maximal number of inner loop value iterations
@param maxresidual_vi Stop the inner policy iteration when the
residual drops below this threshold. This value should be smaller than
maxresidual_pi
@param print_progress Whether to report on progress during the
computation

@return Computed (approximate) solution
*/
template <class SType>
inline SARobustSolution
rsolve_mpi(const GRMDP<SType>& mdp, prec_t discount, SANature nature,
           const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
           unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
           unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
           bool print_progress = false) {
    return mpi_jac<SType, SARobustBellman<SType>>(
        mdp, discount, valuefunction, SARobustBellman<SType>(move(nature), policy),
        iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi, print_progress);
}

/**
Gauss-Seidel variant of value iteration (not parallelized). S-rectangular
nature.

This function is suitable for computing the value function of a finite state
MDP. If the states are ordered correctly, one iteration is enough to compute
the optimal value function. Since the value function is updated from the last
state to the first one, the states should be ordered in the temporal order.

This is a simplified method interface. Use vi_gs with PolicyNature for full
functionality.

@param mdp      The MDP to solve
@param discount Discount factor.
@param nature   Response of nature, the function is the same for all states
and actions.
@param thresholds Parameters passed to nature response functions. One value per
state and then one value per action.
@param valuefunction
Initial value function. Passed by value, because it is modified. Optional, use
all zeros when not provided. Ignored when size is 0.
@param policy    Partial policy specification. Optimize only actions that are
policy[state] = -1. Use policy length 0 to optimize all actions.
@param iterations Maximal number of iterations to run
@param maxresidual Stop when the maximal residual falls below this value.


\returns Solution that can be used to compute the total return, or the optimal
policy.
*/
template <class SType>
inline SRobustSolution
rsolve_s_vi(const GRMDP<SType>& mdp, prec_t discount, SNature nature,
            numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
            unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC) {
    return vi_gs<SType, SRobustBellman<SType>>(
        mdp, discount, move(valuefunction), SRobustBellman<SType>(move(nature), policy),
        iterations, maxresidual);
}

/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.
S-rectangular nature.

WARNING: There is no proof of convergence for this method. This is not the
same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
the discussion in the paper on methods like this one (e.g. Seid, White)

This is a simplified method interface. Use mpi_jac with PolicyNature for full
functionality.

Note that the total number of iterations will be bounded by iterations_pi *
iterations_vi @param type Type of realization of the uncertainty @param
discount Discount factor @param nature Response of nature; the same value is
used for all states and actions. @param valuefunction Initial value function
@param policy Partial policy specification. Optimize only actions that are
policy[state] = -1
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops
below this threshold. @param iterations_vi Maximal number of inner loop value
iterations @param maxresidual_vi Stop policy evaluation when the policy
residual drops below maxresidual_vi * last_policy_residual @param
print_progress Whether to report on progress during the computation \return
Computed (approximate) solution
*/
template <class SType>
inline SRobustSolution
rsolve_s_mpi(const GRMDP<SType>& mdp, prec_t discount, SNature nature,
             const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
             unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
             unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
             bool print_progress = false) {
    return mpi_jac<SType, SRobustBellman<SType>>(
        mdp, discount, valuefunction, SRobustBellman<SType>(move(nature), policy),
        iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi, print_progress);
}

} // namespace craam
