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
#include "craam/MDP.hpp"
#include "craam/Transition.hpp"

#include <eigen3/Eigen/Dense>
#include <rm/range.hpp>

namespace craam { namespace algorithms {

using namespace std;
using namespace Eigen;

/**
 * Updates transition probabilities according to the provided policy.
 *
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
                      bool transpose = false, prec_t discount = 1.0) {

    assert(transitions.rows() == transitions.cols());
    assert(size_t(transitions.rows()) == new_policy.size());
    assert(old_policy.empty() || new_policy.size() == old_policy.size());

    const size_t n = response.state_count();

    bool openmp_error = false;
#pragma omp parallel for
    for (size_t s = 0; s < n; s++) {
        try {
            const Transition& t = response.transition(s, new_policy[s]);

            // if the policy has not changed then do nothing
            if (!old_policy.empty() && old_policy[s] == new_policy[s]) continue;

            // add transition probabilities to the matrix
            const auto& indexes = t.get_indices();
            const auto& probabilities = t.get_probabilities();

            if (!transpose) {
                for (size_t j = 0; j < t.size(); j++)
                    transitions(s, indexes[j]) = discount * probabilities[j];
            } else {
                for (size_t j = 0; j < t.size(); j++)
                    transitions(indexes[j], s) = discount * probabilities[j];
            }
        } catch (const exception& e) {
            // only run this once per loop
            if (!openmp_error) {
                internal::openmp_exception_handler(e, "update_transition_mat");
                openmp_error = true;
            }
        }
    }
    if (openmp_error) throw runtime_error("Failed with an exception in OPENMP block.");
}

/**
 * @brief Creates a transition probability matrix for the Bellman response operator
 *
 * @param response BellmanOperator class (e.g. PlainBellman)
 * @param policy Policy used to construct transition probabilities
 * @param transpose If yes, then the source states are columns (non-standard)
 * @param discount An optional discount factor that multiplies each row
 */
template <typename BellmanResponse,
          typename policy_type = typename BellmanResponse::policy_type>
inline Eigen::MatrixXd transition_mat(const BellmanResponse& response,
                                      const vector<policy_type>& policy,
                                      bool transpose = false, prec_t discount = 1.0) {

    const size_t n = response.state_count();
    Eigen::MatrixXd result = MatrixXd::Zero(n, n);

    update_transition_mat(response, result, policy,
                          vector<typename BellmanResponse::policy_type>(0), transpose,
                          discount);
    return result;
}

/**
 * Constructs the rewards vector for each state for the RMDP.
 *
 * @tparam BellmanResponse An implementation similar to PlainBellman
 *
 * @param response An instance that generates rewards
 * @param policy The current policy
 */
template <typename BellmanResponse,
          typename policy_type = typename BellmanResponse::policy_type>
inline numvec rewards_vec(const BellmanResponse& response,
                          const vector<policy_type>& policy) {

    const auto n = response.state_count();
    numvec rewards(n);

    bool openmp_error = false;
#pragma omp parallel for
    for (size_t s = 0; s < n; s++) {
        try {
            rewards[s] = response.reward(s, policy[s]);
        } catch (const exception& e) {
            // only run this once per loop
            if (!openmp_error) {
                internal::openmp_exception_handler(e, "rewards_vec");
                openmp_error = true;
            }
        }
    }
    if (openmp_error) throw runtime_error("Failed with an exception in OPENMP block.");
    return rewards;
}

/**
Computes occupancy frequencies using matrix representation of transition
probabilities. This method requires computing a matrix inverse.

@tparam Methods for computing Bellman responses, similar to PlainBellman

@param init Initial distribution (alpha)
@param discount Discount factor (gamma)
@param policies The policy (indvec) or a pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically
        a randomized policy
*/
template <typename BellmanResponse,
          typename policy_type = typename BellmanResponse::policy_type>
inline numvec occfreq_mat(const BellmanResponse& response, const Transition& init,
                          prec_t discount, const policy_type& policy) {
    const auto n = response.state_count();

    // initial distribution
    const numvec& ivec = init.probabilities_vector(n);
    const VectorXd initial_vec = Map<const VectorXd, Unaligned>(ivec.data(), ivec.size());

    // get transition matrix and construct (I - gamma * P^T)
    MatrixXd t_mat =
        MatrixXd::Identity(n, n) - transition_mat(response, policy, true, discount);

    // solve set of linear equations
    numvec result(n, 0);
    Map<VectorXd, Unaligned>(result.data(), result.size()) =
        HouseholderQR<MatrixXd>(t_mat).solve(initial_vec);

    return result;
}

/**
 * Computes the value function of a policy by solving a system of linear equations
 *
 * @param response Bellman response that provides the transition probabilities and rewards
 * @param discount discount factor
 *
 *
 */
template <typename BellmanResponse,
          typename policy_type = typename BellmanResponse::policy_type>
inline numvec valuefunction_mat(const BellmanResponse& response, prec_t discount,
                                const policy_type& policy) {

    const auto n = response.state_count();
    const numvec rewards = rewards_vec(response, policy);
    const VectorXd rewards_vec =
        Map<const VectorXd, Unaligned>(rewards.data(), rewards.size());

    // get transition matrix and construct (I - gamma * P)
    MatrixXd t_mat =
        MatrixXd::Identity(n, n) - transition_mat(response, policy, true, discount);

    // solve set of linear equations
    numvec result(n, 0);
    Map<VectorXd, Unaligned>(result.data(), result.size()) =
        HouseholderQR<MatrixXd>(t_mat).solve(rewards_vec);

    return result;
}

/**
  * Constructs the LP matrix and a reward vector for an MDP.
  *
  * The output are a matrix A and a vector b such that
  *
  * A = [  I- gamma P_1 ; I - gamma P_2 ; .... ]
  * and
  * b = [ r_1 ; r_2 ; ...]
  * where P_a and r_a are the tansition probabilities and rewards (respectively for each action).
  * The vector of rewards r_1 is computed as an average over the target states. That is,
  * r_a(s) = sum_s p(s,a,s') r(s,a,s')
  * Note that this is not normalized, which matter when the transition probabilities do not
  * sum to one
  *
  * @param mdp MDP definition
  * @param discount The discount factor
  * @returns A tuple with (A,b,stateactionids) where A, b are described as above and
  *     stateactionsids represent which state and action each row
  *     represents
  */
inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd, vector<std::pair<long, long>>>
lp_matrix(const MDP& mdp, prec_t discount) {

    const auto nstates = mdp.size();

    // count the number of state-action pairs (this is the number of rows)
    size_t nstateactions = 0;
    for (const auto& s : mdp)
        nstateactions += s.size();

    // Initialize output values
    Eigen::MatrixXd A(nstateactions, nstates);
    Eigen::VectorXd b(nstateactions);
    std::vector<std::pair<long, long>> stateactionids(nstateactions, {-1, -1});

    size_t row_index = 0;
    // add objectives and constraints for each state
    for (size_t si = 0; si < nstates; ++si) {
        assert(row_index < nstateactions && row_index >= 0);
        const State& s = mdp[si];

        for (size_t ai = 0; ai < s.size(); ++ai) {
            const Action& a = s[ai];
            stateactionids[row_index] = {si, ai};

            double reward_sa = 0; // use to aggregate rewards
            for (size_t ti = 0; ti < a.size(); ++ti) {
                const size_t to_si =
                    a.get_indices()[ti]; // index of the destination state
                const long prob = a.get_probabilities()[ti];
                const prec_t rew = a.get_rewards()[ti];

                reward_sa += rew * prob;
                // compute I - gamma P
                A(row_index, to_si) = (si == to_si ? 1.0 : 0.0) - discount * prob;
            }
            b(row_index) = reward_sa;
            ++row_index;
        }
    }
    return {move(A), move(b), move(stateactionids)};
}

}} // namespace craam::algorithms
