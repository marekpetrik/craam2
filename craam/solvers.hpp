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

#include "craam/MDP.hpp"
#include "craam/MDPO.hpp"
#include "craam/Solution.hpp"
#include "craam/algorithms/bellman_mdp.hpp"
#include "craam/algorithms/bellman_mdpo.hpp"
#include "craam/algorithms/iteration_methods.hpp"
#include "craam/algorithms/nature_declarations.hpp"

namespace craam {

/**
 * @defgroup Value Iteration
 *
 * Gauss-Seidel variant of value iteration (not parallelized).
 *
 * This function is suitable for computing the value function of a finite state
 * MDP. If the states are ordered correctly, one iteration is enough to compute the
 * optimal value function. Since the value function is updated from the last state
 * to the first one, the states should be ordered in the temporal order.
 * @param mdp The MDP to solve
 * @param discount Discount factor.
 * @param valuefunction Initial value function. Passed by value, because it is
 * modified. Optional, use all zeros when not provided. Ignored when size is 0.
 * @param policy Partial policy specification. Optimize only actions that are policy[state] = -1
 * @param iterations Maximal number of iterations to run
 * @param maxresidual Stop when the maximal residual falls below this value.
 *
 * @returns Solution that can be used to compute the total return, or the optimal
policy.
 */

/**
 *  Modified policy iteration using Jacobi value iteration in the inner loop.
 * This method generalizes modified policy iteration to robust MDPs.
 * In the value iteration step, both the action *and* the outcome are fixed.
 *
 * Note that the total number of iterations will be bounded by iterations_pi *
 * iterations_vi
 * @param type Type of realization of the uncertainty
 * @param discount Discount factor
 * @param valuefunction Initial value function
 * @param policy Partial policy specification. Optimize only actions that are
 * policy[state] = -1
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param iterations_vi Maximal number of inner loop value iterations
 * @param maxresidual_vi Stop policy evaluation when the policy residual drops
 * below maxresidual_vi * last_policy_residual
 * @param print_progress Whether to report on progress during the computation
 *
 * @return Computed (approximate) solution
 */

// **************************************************************************
// Plain MDP methods
// **************************************************************************

/**
 * \ingroup ValueIteration
 */
inline DetermSolution solve_vi(const MDP& mdp, prec_t discount,
                               numvec valuefunction = numvec(0),
                               const indvec& policy = indvec(0),
                               unsigned long iterations = MAXITER,
                               prec_t maxresidual = SOLPREC) {
    return algorithms::vi_gs(algorithms::PlainBellman(mdp, policy), discount,
                             move(valuefunction), iterations, maxresidual);
}

/**
 * @ingroup ModifiedPolicyIteration
 */
inline DetermSolution
solve_mpi(const MDP& mdp, prec_t discount, const numvec& valuefunction = numvec(0),
          const indvec& policy = indvec(0), unsigned long iterations_pi = MAXITER,
          prec_t maxresidual_pi = SOLPREC, unsigned long iterations_vi = MAXITER,
          prec_t maxresidual_vi = 0.9, bool print_progress = false) {

    return algorithms::mpi_jac(algorithms::PlainBellman(mdp, policy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, print_progress);
}

/**
Computes occupancy frequencies using matrix representation of transition
probabilities. This method requires computing a matrix inverse.

\param init Initial distribution (alpha)
\param discount Discount factor (gamma)
\param policies The policy (indvec) or a pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically
        a randomized policy
*/
inline numvec occupancies(const MDP& mdp, const Transition& initial, prec_t discount,
                          const indvec& policy) {

    return algorithms::occfreq_mat(algorithms::PlainBellman(mdp), initial, discount,
                                   policy);
}

// **************************************************************************
// Robust MDP methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s,a-rectangular nature.
 */
inline SARobustSolution
rsolve_vi(const MDP& mdp, prec_t discount, const algorithms::SANature& nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC) {

    return algorithms::vi_gs(algorithms::SARobustBellman(mdp, move(nature), policy),
                             discount, move(valuefunction), iterations, maxresidual);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s,a-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 */
inline SARobustSolution
rsolve_mpi(const MDP& mdp, prec_t discount, const algorithms::SANature&& nature,
           const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
           unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
           unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
           bool print_progress = false) {
    return algorithms::mpi_jac(algorithms::SARobustBellman(mdp, nature, policy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, print_progress);
}

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s-rectangular nature.
 */
inline SRobustSolution
rsolve_s_vi(const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
            numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
            unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC) {
    return algorithms::vi_gs(algorithms::SRobustBellman(mdp, nature, policy), discount,
                             move(valuefunction), iterations, maxresidual);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 */
inline SRobustSolution
rsolve_s_mpi(const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
             const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
             unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
             unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
             bool print_progress = false) {
    return algorithms::mpi_jac(algorithms::SRobustBellman(mdp, nature, policy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, print_progress);
}

// **************************************************************************
// Plain MDPO methods
// **************************************************************************

/**
 * \ingroup ValueIteration
 */
inline DetermSolution solve_vi(const MDPO& mdp, prec_t discount,
                               numvec valuefunction = numvec(0),
                               const indvec& policy = indvec(0),
                               unsigned long iterations = MAXITER,
                               prec_t maxresidual = SOLPREC) {
    auto solution = algorithms::vi_gs(
        algorithms::SARobustOutcomeBellman(mdp, algorithms::nats::average(), policy),
        discount, move(valuefunction), iterations, maxresidual);

    return DetermSolution(solution.valuefunction, unzip(solution.policy).first,
                          solution.residual, solution.iterations, solution.time);
}

/**
 * @ingroup ModifiedPolicyIteration
 */
inline DetermSolution
solve_mpi(const MDPO& mdp, prec_t discount, const numvec& valuefunction = numvec(0),
          const indvec& policy = indvec(0), unsigned long iterations_pi = MAXITER,
          prec_t maxresidual_pi = SOLPREC, unsigned long iterations_vi = MAXITER,
          prec_t maxresidual_vi = 0.9, bool print_progress = false) {

    auto solution = algorithms::mpi_jac(
        algorithms::SARobustOutcomeBellman(mdp, algorithms::nats::average(), policy),
        discount, valuefunction, iterations_pi, maxresidual_pi, iterations_vi,
        maxresidual_vi, print_progress);

    return DetermSolution(solution.valuefunction, unzip(solution.policy).first,
                          solution.residual, solution.iterations, solution.time);
}

// **************************************************************************
// Robust MDPO methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s,a-rectangular nature.
 */
inline SARobustSolution
rsolve_vi(const MDPO& mdp, prec_t discount, const algorithms::SANature& nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC) {

    return algorithms::vi_gs(
        algorithms::SARobustOutcomeBellman(mdp, move(nature), policy), discount,
        move(valuefunction), iterations, maxresidual);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s,a-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 */
inline SARobustSolution
rsolve_mpi(const MDPO& mdp, prec_t discount, const algorithms::SANature&& nature,
           const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
           unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
           unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
           bool print_progress = false) {
    return algorithms::mpi_jac(algorithms::SARobustOutcomeBellman(mdp, nature, policy),
                               discount, valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, print_progress);
}

// **************************************************************************
// Compute Occupancy Frequency
// **************************************************************************

} // namespace craam
