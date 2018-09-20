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

#include "craam/GMDP.hpp"
#include "craam/Transition.hpp"

#include <eigen3/Eigen/Dense>
#include <rm/range.hpp>

namespace craam { namespace algorithms {

using namespace std;
using namespace Eigen;

/// Internal helper functions
namespace internal {

/// Helper function to deal with variable indexing
template <class SType>
inline Transition mean_transition_state(const SType& state, long index,
                                        const pair<indvec, vector<numvec>>& policies) {
    return state.mean_transition(policies.first[index], policies.second[index]);
}

/// Helper function to deal with variable indexing
template <class SType>
inline Transition mean_transition_state(const SType& state, long index,
                                        const indvec& policy) {
    return state.mean_transition(policy[index]);
}

/// Helper function to deal with variable indexing
/// \param state
/// \param index
/// \param policies
template <class SType>
inline prec_t mean_reward_state(const SType& state, long index,
                                const pair<indvec, vector<numvec>>& policies) {

    return state.mean_reward(policies.first[index], policies.second[index]);
}

// TODO: this function should be called by the Bellman operator
/// Helper function to deal with variable indexing
template <class SType>
inline prec_t mean_reward_state(const SType& state, long index, const indvec& policy) {
    return state[policy[index]].mean_reward();
}
} // namespace internal

/**
Constructs the transition (or its transpose) matrix for the policy.

\tparam SType Type of the state in the MDP (regular vs robust)
\tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic
                policy and a randomized policy of the nature
\param rmdp Regular or robust MDP
\param policies The policy (indvec) or the pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically
        a randomized policy
\param transpose (optional, false) Whether to return the transpose of the
transition matrix. This is useful for computing occupancy frequencies
*/
template <typename SType, typename Policies>
inline MatrixXd transition_mat(const GMDP<SType>& rmdp, const Policies& policies,
                               bool transpose = false) {
    const size_t n = rmdp.state_count();
    MatrixXd result = MatrixXd::Zero(n, n);

#pragma omp parallel for
    for (size_t s = 0; s < n; s++) {
        const Transition&& t = internal::mean_transition_state(rmdp[s], s, policies);

        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        if (!transpose) {
            for (size_t j = 0; j < t.size(); j++)
                result(s, indexes[j]) = probabilities[j];
        } else {
            for (size_t j = 0; j < t.size(); j++)
                result(indexes[j], s) = probabilities[j];
        }
    }
    return result;
}

/**
 * @brief Updates transition probabilities according to the provided policy.
 * @param response BellmanOperator class (e.g. PlainBellman)
 * @param transition Transition probabilities for @a old_policy
 * @param new_policy Policy used to update transition probabilities
 * @param old_policy Policy that corresponds to values in @a transition. The
 *          parameter can be length 0 if the transitions matrix is invalid.
 * @param transpose If true, transposes the probaility matrix
 */
template <typename BellmanResponse>
inline void
update_transition_mat(const BellmanResponse& response, MatrixXd& transitions,
                      const vector<typename BellmanResponse::policy_type>& new_policy,
                      const vector<typename BellmanResponse::policy_type>& old_policy,
                      bool transpose = false) {

    assert(transitions.rows() == transitions.cols());
    assert(transitions.rows() == new_policy.size());
    assert(old_policy.empty() || new_policy.size() == old_policy.size());

    const size_t n = response.state_count();

#pragma omp parallel for
    for (size_t s = 0; s < n; s++) {
        const Transition& t = response.mean_transition(s, new_policy[s]);

        // if the policy has not changed then do nothing
        if (!old_policy.empty() && old_policy[s] == new_policy[s]) continue;

        // add transition probabilities to the matrix
        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        if (!transpose) {
            for (size_t j = 0; j < t.size(); j++)
                transitions(s, indexes[j]) = probabilities[j];
        } else {
            for (size_t j = 0; j < t.size(); j++)
                transitions(indexes[j], s) = probabilities[j];
        }
    }
}

/**
 * @brief Creates a transition probability matrix for the Bellman response operator
 * @param response BellmanOperator class (e.g. PlainBellman)
 * @param policy Policy used to construct transition probabilities
 * @param transpose If true, transposes the probaility matrix
 */
template <typename BellmanResponse>
inline MatrixXd
transition_mat(const BellmanResponse& response,
               const vector<typename BellmanResponse::policy_type>& policy,
               bool transpose = false) {

    const size_t n = response.state_count();
    MatrixXd result(n, n);

    update_transition_mat(response, result, policy,
                          vector<typename BellmanResponse::policy_type>(0), transpose);
}

/**
Constructs the rewards vector for each state for the RMDP.

\tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic
                policy and a randomized policy of the nature

\param rmdp Regular or robust MDP
\param policies The policy (indvec) or the pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically
        a randomized policy
 */
template <typename SType, typename Policy>
inline numvec rewards_vec(const GMDP<SType>& rmdp, const Policy& policies) {

    const auto n = rmdp.state_count();
    numvec rewards(n);

#pragma omp parallel for
    for (size_t s = 0; s < n; s++) {
        const SType& state = rmdp[s];
        if (state.is_terminal())
            rewards[s] = 0;
        else
            rewards[s] = internal::mean_reward_state(state, s, policies);
    }
    return rewards;
}

/**
Computes occupancy frequencies using matrix representation of transition
probabilities. This method may not scale well


\tparam SType Type of the state in the MDP (regular vs robust)
\tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic
                policy and a randomized policy of the nature
\param init Initial distribution (alpha)
\param discount Discount factor (gamma)
\param policies The policy (indvec) or a pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically
        a randomized policy
*/
template <typename SType, typename Policies>
inline numvec occfreq_mat(const GMDP<SType>& rmdp, const Transition& init,
                          prec_t discount, const Policies& policies) {
    const auto n = rmdp.state_count();

    // initial distribution
    const numvec& ivec = init.probabilities_vector(n);
    const VectorXd initial_vec = Map<const VectorXd, Unaligned>(ivec.data(), ivec.size());

    // get transition matrix and construct (I - gamma * P^T)
    MatrixXd t_mat =
        MatrixXd::Identity(n, n) - discount * transition_mat(rmdp, policies, true);

    // solve set of linear equations
    numvec result(n, 0);
    Map<VectorXd, Unaligned>(result.data(), result.size()) =
        HouseholderQR<MatrixXd>(t_mat).solve(initial_vec);

    return result;
}
}} // namespace craam::algorithms
