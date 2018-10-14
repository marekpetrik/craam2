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
Gauss-Seidel variant of value iteration (not parallelized). See solve_vi for a
simplified interface.

This function is suitable for computing the value function of a finite state
MDP. If the states are ordered correctly, one iteration is enough to compute the
optimal value function. Since the value function is updated from the last state
to the first one, the states should be ordered in the temporal order.

@tparam ResponseType Class responsible for computing the Bellman updates. Should
be compatible with PlainBellman

@param response Using PolicyResponce allows to specify a partial policy. Only
the actions that not provided by the partial policy are included in the
optimization. Using a class of a different types enables computing other
@param discount Discount factor.
@param valuefunction Initial value function. Passed by value, because it is
modified. Optional, use all zeros when not provided. Ignored when size is 0.
objectives, such as robust or risk averse ones.
@param iterations Maximal number of iterations to run
@param maxresidual Stop when the maximal residual falls below this value.
@param progress An optional function for reporting progress and can
                return false to stop computation

@returns Solution that can be used to compute the total return, or the optimal
policy.
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
    return Solution<policy_type>(move(valuefunction), move(policy), residual, i,
                                 duration.count());
}

/**
Modified policy iteration using Jacobi value iteration in the inner loop. See
solve_mpi for a simplified interface. This method generalizes modified policy
iteration to robust MDPs. In the value iteration step, both the action *and* the
outcome are fixed.

@tparam ResponseType Class responsible for computing the Bellman updates. Should

Note that the total number of iterations will be bounded by iterations_pi *
iterations_vi
@param type Type of realization of the uncertainty
@param discount Discount factor
@param valuefunction Initial value function
@param response Using PolicyResponce allows to specify a partial policy. Only
the actions that not provided by the partial policy are included in the
optimization. Using a class of a different types enables computing other
objectives, such as robust or risk averse ones.
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops
below this threshold.
@param iterations_vi Maximal number of inner loop value iterations
@param maxresidual_vi_rel Stop policy evaluation when the policy residual drops
below maxresidual_vi_rel * last_policy_residual
@param progress An optional function for reporting progress and can
                return false to stop computation

@return Computed (approximate) solution
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
    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    for (i = 0; i < iterations_pi; i++) {

        // this should use move semantics and therefore be very efficient
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
    return Solution<policy_type>(move(targetvalue), move(policy), residual_pi, i,
                                 duration.count());
}

/**
Policy iteration. See solve_pi for a simplified interface. In the value iteration
step, both the action *and* the
outcome are fixed.

@tparam ResponseType Class responsible for computing the Bellman updates. Should
        be compatible with PlainBellman

Note that the total number of iterations will be bounded by iterations_pi *
iterations_vi
@param type Type of realization of the uncertainty
@param discount Discount factor
@param valuefunction Initial value function. Used to compute the first policy
@param response Using PolicyResponce allows to specify a partial policy. Only
the actions that not provided by the partial policy are included in the
optimization. Using a class of a different types enables computing other
objectives, such as robust or risk averse ones.
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops
below this threshold.
@param iterations_vi Maximal number of inner loop value iterations
@param maxresidual_vi_rel Stop policy evaluation when the policy residual drops
below maxresidual_vi_rel * last_policy_residual
@param progress An optional function for reporting progress and can
                return false to stop computation


@return Computed (approximate) solution
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

    // TODO: could be sped up by keeping I - gamma * P instead of transition probabilities
    // **discounted** matrix of transition probabilities
    MatrixXd trans_discounted = transition_mat(response, policy, false, discount);

    for (i = 0; i < iterations_pi; ++i) {

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

        const numvec rw = rewards_vec(response, policy);
        // get transition matrix and construct (I - gamma * P)
        // TODO: this step could be eliminated by keeping I - gamma P (this is an unnecessary copy)
        MatrixXd t_mat =
            MatrixXd::Identity(n, n) - transition_mat(response, policy, false, discount);
        // compute and store the value function
        Map<VectorXd, Unaligned>(valuefunction.data(), valuefunction.size()) =
            HouseholderQR<MatrixXd>(t_mat).solve(
                Map<const VectorXd, Unaligned>(rw.data(), rw.size()));
    }
    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    return Solution<policy_type>(move(valuefunction), move(policy), residual_pi, i,
                                 duration.count());
}

}} // namespace craam::algorithms
