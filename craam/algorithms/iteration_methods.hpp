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

#include "craam/Solution.hpp"
#include "craam/algorithms/matrices.hpp"

#include <chrono>

namespace craam { namespace algorithms {

namespace internal {

/// An empty progress function, always returns true
inline bool empty_progress(size_t iteration, prec_t residual) { return true; }

} // namespace internal

/**
 * Gauss-Seidel variant of value iteration (not parallelized). See solve_vi for a
 * simplified interface.
 *
 * This function is suitable for computing the value function of a finite state
 * MDP. If the states are ordered correctly, one iteration is enough to compute the
 * optimal value function. Since the value function is updated from the last state
 * to the first one, the states should be ordered in the temporal order.
 *
 * @tparam ResponseType Class responsible for computing the Bellman updates. Should
 * be compatible with PlainBellman
 *
 * @param response Using PolicyResponce allows to specify a partial policy. Only
 * the actions that not provided by the partial policy are included in the
 * optimization. Using a class of a different types enables computing other
 * @param discount Discount factor.
 * @param valuefunction Initial value function. Passed by value, because it is
 * modified. Optional, use all zeros when not provided. Ignored when size is 0.
 * objectives, such as robust or risk averse ones.
 * @param iterations Maximal number of iterations to run
 * @param maxresidual Stop when the maximal residual falls below this value.
 * @param progress An optional function for reporting progress and can
 *                 return false to stop computation
 *
 * @returns Solution that can be used to compute the total return, or the optimal
 * policy.
 */
template <class ResponseType>
inline Solution<typename ResponseType::policy_type>
vi_gs(const ResponseType& response, prec_t discount, numvec valuefunction = numvec(0),
      unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
      const std::function<bool(size_t, prec_t)>& progress = internal::empty_progress) {
    using policy_type = typename ResponseType::policy_type;

    // just quit if there are no states
    if (response.state_count() == 0) return Solution<policy_type>(0);

    // time the computation
    auto start = chrono::steady_clock::now();

    if (valuefunction.empty()) { valuefunction.resize(response.state_count(), 0.0); }

    vector<policy_type> policy(response.state_count());

    // initialize values
    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i; // iterations defined outside to make them reportable

    for (i = 0; i < iterations && residual > maxresidual && progress(i, residual); i++) {
        residual = 0;

        for (size_t s = 0l; s < response.state_count(); s++) {
            prec_t newvalue;
            tie(newvalue, policy[s]) =
                response.policy_update(long(s), valuefunction, discount);

            residual = max(residual, abs(valuefunction[s] - newvalue));
            valuefunction[s] = newvalue;
        }
    }

    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    int status = residual <= maxresidual ? 0 : 1;
    return Solution<policy_type>(move(valuefunction), move(policy), residual, i,
                                 duration.count(), status);
}

/**
 * Modified policy iteration using Jacobi value iteration in the inner loop. See
 * solve_mpi for a simplified interface. This method can also be applied directly
 * to robust MDPs, but it is not guaranteed to converge (may loop)
 * for the robust objective. See:
 *
 * Condon, A. (1993). On algorithms for simple stochastic games.
 * Advances in Computational Complexity Theory, DIMACS Series in
 *  Discrete Mathematics and Theoretical Computer Science, 13, 51–71.
 *
 * It probably converges just fine for the optimistic objective.
 *
 * In the value iteration step, both the action *and* the
 * outcome are fixed.
 *
 * @tparam ResponseType Class responsible for computing the Bellman updates. Should
 *
 * Note that the total number of iterations will be bounded by iterations_pi *
 * iterations_vi
 * @param type Type of realization of the uncertainty
 * @param discount Discount factor
 * @param valuefunction Initial value function
 * @param response Using PolicyResponce allows to specify a partial policy. Only
 * the actions that not provided by the partial policy are included in the
 * optimization. Using a class of a different types enables computing other
 * objectives, such as robust or risk averse ones.
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param iterations_vi Maximal number of inner loop value iterations
 * @param maxresidual_vi_rel Stop policy evaluation when the policy residual drops
 * below maxresidual_vi_rel * last_policy_residual
 * @param progress An optional function for reporting progress and can
 *                 return false to stop computation
 *
 * @return Computed (approximate) solution
 */
template <class ResponseType>
inline Solution<typename ResponseType::policy_type>
mpi_jac(const ResponseType& response, prec_t discount,
        const numvec& valuefunction = numvec(0), unsigned long iterations_pi = MAXITER,
        prec_t maxresidual_pi = SOLPREC, unsigned long iterations_vi = MAXITER,
        prec_t maxresidual_vi_rel = 0.9,
        const std::function<bool(size_t, prec_t)>& progress = internal::empty_progress) {
    using policy_type = typename ResponseType::policy_type;
    // just quit if there are no states
    if (response.state_count() == 0) { return Solution<policy_type>(0); }

    // time the computation
    auto start = chrono::steady_clock::now();

    // intialize the policy
    vector<policy_type> policy(response.state_count());

    numvec sourcevalue = valuefunction; // value function to compute the update
    // resize if the the value function is empty and initialize to 0
    if (sourcevalue.empty()) sourcevalue.resize(response.state_count(), 0.0);
    numvec targetvalue = sourcevalue; // value function to hold the updated values

    numvec residuals(response.state_count());

    // residual in the policy iteration part
    static_assert(std::numeric_limits<prec_t>::has_infinity == true);
    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    for (i = 0; i < iterations_pi; i++) {
        // this just swaps pointers
        swap(targetvalue, sourcevalue);

        prec_t residual_vi = numeric_limits<prec_t>::infinity();

        // update policies
#pragma omp parallel for
        for (auto s = 0l; s < long(response.state_count()); s++) {
            prec_t newvalue;
            tie(newvalue, policy[s]) = response.policy_update(s, sourcevalue, discount);

            residuals[s] = abs(sourcevalue[s] - newvalue);
            targetvalue[s] = newvalue;
        }
        residual_pi = *max_element(residuals.cbegin(), residuals.cend());

        // the residual is sufficiently small
        if (residual_pi <= maxresidual_pi || !progress(i, residual_pi)) break;

        // compute values using value iteration

        for (size_t j = 0;
             j < iterations_vi && residual_vi > maxresidual_vi_rel * residual_pi; j++) {

            swap(targetvalue, sourcevalue);

#pragma omp parallel for
            for (auto s = 0l; s < long(response.state_count()); s++) {
                prec_t newvalue =
                    response.compute_value(policy[s], s, sourcevalue, discount);
                residuals[s] = abs(sourcevalue[s] - newvalue);
                targetvalue[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(), residuals.end());
        }
    }
    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    int status = residual_pi <= maxresidual_pi ? 0 : 1;
    return Solution<policy_type>(move(targetvalue), move(policy), residual_pi, i,
                                 duration.count(), status);
}

/**
 * Policy iteration. See solve_pi for a simplified interface. In the value iteration
 * step, both the action *and* the
 * outcome are fixed.
 *
 * This method can also be applied directly
 * to robust MDPs, but it is not guaranteed to converge (may loop)
 * for the robust objective. See:
 *
 * Condon, A. (1993). On algorithms for simple stochastic games.
 * Advances in Computational Complexity Theory, DIMACS Series in
 *  Discrete Mathematics and Theoretical Computer Science, 13, 51–71.
 *
 * It probably converges just fine for the optimistic objective.
 *
 * @tparam ResponseType Class responsible for computing the Bellman updates. Should
 *         be compatible with PlainBellman
 *
 * Note that the total number of iterations will be bounded by iterations_pi *
 * iterations_vi
 * @param type Type of realization of the uncertainty
 * @param discount Discount factor
 * @param valuefunction Initial value function. Used to compute the first policy
 * @param response Using PolicyResponce allows to specify a partial policy. Only
 * the actions that not provided by the partial policy are included in the
 * optimization. Using a class of a different types enables computing other
 * objectives, such as robust or risk averse ones.
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param iterations_vi Maximal number of inner loop value iterations
 * @param maxresidual_vi_rel Stop policy evaluation when the policy residual drops
 * below maxresidual_vi_rel * last_policy_residual
 * @param progress An optional function for reporting progress and can
 *                 return false to stop computation
 *
 *
 * @return Computed (approximate) solution
 */
template <class ResponseType>
inline Solution<typename ResponseType::policy_type>
pi(const ResponseType& response, prec_t discount, numvec valuefunction = numvec(0),
   unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
   const std::function<bool(size_t, prec_t)>& progress = internal::empty_progress) {
    const auto n = response.state_count();

    using policy_type = typename ResponseType::policy_type;
    // just quit if there are no states
    if (n == 0) { return Solution<policy_type>(0); }
    // time the computation
    auto start = chrono::steady_clock::now();
    // intialize the policy
    vector<policy_type> policy(n);
    // keep the old policy around to detect no change
    vector<policy_type> policy_old = policy;
    // resize if the the value function is empty and initialize to 0
    if (valuefunction.empty()) valuefunction.resize(n, 0.0);

    numvec residuals(n);

    // residual in the policy iteration part
    prec_t residual_pi = numeric_limits<prec_t>::infinity();
    size_t i; // defined here to be able to report the number of iterations

    // NOTE: could be sped up by keeping I - gamma * P instead of transition probabilities

    // first udate the policy
#pragma omp parallel for
    for (size_t s = 0; s < n; ++s) {
        tie(std::ignore, policy[s]) = response.policy_update(s, valuefunction, discount);
    }

    // **discounted** matrix of transition probabilities
    MatrixXd trans_discounted = transition_mat(response, policy, false, discount);

    for (i = 0; i < iterations_pi; ++i) {

        const numvec rw = rewards_vec(response, policy);
        // get transition matrix and construct (I - gamma * P)
        // TODO: this step could be eliminated by keeping I - gamma P (this is an unnecessary copy)
        MatrixXd t_mat =
            MatrixXd::Identity(n, n) - transition_mat(response, policy, false, discount);
        // compute and store the value function
        Map<VectorXd, Unaligned>(valuefunction.data(), valuefunction.size()) =
            HouseholderQR<MatrixXd>(t_mat).solve(
                Map<const VectorXd, Unaligned>(rw.data(), rw.size()));

        // update policy
        swap(policy, policy_old);
#pragma omp parallel for
        for (size_t s = 0; s < n; ++s) {
            prec_t newvalue;
            tie(newvalue, policy[s]) = response.policy_update(s, valuefunction, discount);
            residuals[s] = abs(valuefunction[s] - newvalue);
        }
        // TODO: change this to a span seminorm (in all algorithms)
        residual_pi = *max_element(residuals.cbegin(), residuals.cend());

        // the residual is sufficiently small
        if (residual_pi <= maxresidual_pi || policy == policy_old ||
            !progress(i, residual_pi))
            break;

        // ** now compute the value function
        // 1. update the transition probabilities
        update_transition_mat(response, trans_discounted, policy, policy_old, false,
                              discount);
    }
    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    int status = residual_pi <= maxresidual_pi ? 0 : 1;
    return Solution<policy_type>(move(valuefunction), move(policy), residual_pi, i,
                                 duration.count(), status);
}

/**
 * Robust partial policy iteration, proposed in Ho 2019. Converges in robust settings to
 * the optimal solution, and probably converges in the robust setting.
 *
 * @tparam ResponseType Class responsible for computing the Bellman updates. This works
 *                      with SARobustBellman and SRobustBellman
 *
 * The algorithm is meant for solving robust MDPs.
 *
 * @param response The policy provided to the response is not respected by this method
 * @param residual Target Bellman residual (when to stop)
 * @param rob_residual_init Initial target residual for solving the robust problem
 * @param rob_residual_rate Multiplicative coefficient that controls the
 * decrese in the target rate. It needs to be smaller than the discount factor.
 *
 * @return Computed (approximate) solution
 */
template <class ResponseType>
inline Solution<typename ResponseType::policy_type>
rppi(ResponseType response, prec_t discount, numvec valuefunction = numvec(0),
     unsigned long iterations_pi = MAXITER, prec_t maxresidual = SOLPREC,
     const prec_t rob_residual_init = 1.0, const prec_t rob_residual_rate = 0.5,
     const std::function<bool(size_t, prec_t)>& progress = internal::empty_progress) {
    using policy_type = typename ResponseType::policy_type;
    using dec_policy_type = typename ResponseType::dec_policy_type;

    // just quit if there are no states
    if (response.state_count() == 0) { return Solution<policy_type>(0); }

    // make sure that the value function is the right size
    if (valuefunction.empty()) { valuefunction.resize(response.state_count(), 0.0); }

    // time the computation
    auto start = chrono::steady_clock::now();

    // keep track of the target residual achieved in the policy iteration
    prec_t target_residual = rob_residual_init;

    // intialize the policy (the policy could be randomized or deterministic)
    vector<dec_policy_type> dec_policy(response.state_count());
    // this an array that holds the output policy (only used for the output)
    vector<policy_type> output_policy(response.state_count());

    // initial bellman residual = to check for the stopping criterion
    static_assert(std::numeric_limits<prec_t>::has_infinity == true);
    prec_t residual_pi = std::numeric_limits<prec_t>::infinity();
    numvec residuals(response.state_count());

    unsigned long iterations = 1;
    do {
        // *** robust policy evaluation ***
        // set the dec_policy to prevent optimization
        // count the inner policy iterations too
        bool inner_continue = true; // propagate a termination request from
                                    // the inner method
        response.set_decision_policy(dec_policy);
        Solution<policy_type> solution_rob =
            pi(response, discount, valuefunction, iterations_pi - iterations,
               target_residual,
               [iterations, residual_pi, &progress, &inner_continue](size_t iters,
                                                                     prec_t res) {
                   inner_continue = progress(iterations + iters, residual_pi);
                   return inner_continue;
               });
        // remember that the number of iterations is already offset here
        iterations += solution_rob.iterations - iterations;
        valuefunction = move(solution_rob.valuefunction);
        if (!inner_continue) break;

        // *** robust policy update ***
        // set the dec policy to empty to optimize it
        response.set_decision_policy();
#pragma omp parallel for
        for (auto s = 0l; s < long(response.state_count()); s++) {
            // the new value is only used to compute the residual
            // otherwise this only about the policy
            prec_t newvalue;
            tie(newvalue, output_policy[s]) =
                response.policy_update(s, valuefunction, discount);
            // update the policy of the decision maker (to be used in the evaluation)
            // assume that the policy type is a tuple: [dec policy, nat policy]
            dec_policy[s] = output_policy[s].first;
            residuals[s] = abs(valuefunction[s] - newvalue);
        }
        residual_pi = *max_element(residuals.cbegin(), residuals.cend());
        ++iterations;

        // check whether there is no reson to interrupt
        if (!progress(iterations, residual_pi)) break;
    } while (residual_pi > maxresidual && iterations <= iterations_pi);

    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    int status = residual_pi <= residual_pi ? 0 : 1;
    return Solution<policy_type>(move(valuefunction), move(output_policy), residual_pi,
                                 iterations, duration.count(), status);
}

}} // namespace craam::algorithms
