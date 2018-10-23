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

/**
Value-function based methods (value iteration and policy iteration) style
algorithms. Provides abstractions that allow generalization to both robust and
regular MDPs.
*/
#pragma once

#include "craam/MDP.hpp"
#include "craam/algorithms/nature_declarations.hpp"

#include <functional>

/// Main namespace for algorithms that operate on MDPs and MDPOs
namespace craam { namespace algorithms {

using namespace std;

// *******************************************************
// Plain Action
// *******************************************************

/**
Computes the average value of the action.

@param action Action for which to compute the value
@param valuefunction State value function to use
@param discount Discount factor
@return Action value
*/
inline prec_t value_action(const Action& action, const numvec& valuefunction,
                           prec_t discount) {
    return action.value(valuefunction, discount);
}

/**
Computes a value of the action for a given distribution. This function can be
used to evaluate a robust solution which may modify the transition
probabilities.

The new distribution may be non-zero only for states for which the original
distribution is not zero.

@param action Action for which to compute the value
@param valuefunction State value function to use
@param discount Discount factor
@param distribution New distribution. The length must match the number of
            states to which the original transition probabilities are strictly
greater than 0. The order of states is the same as in the underlying transition.
@return Action value
*/
inline prec_t value_action(const Action& action, const numvec& valuefunction,
                           prec_t discount, const numvec& distribution) {
    return action.value(valuefunction, discount, distribution);
}

// *******************************************************
// Robust Action
// *******************************************************

/**
 * The function computes the value of each transition by adding the
 * reward function to the discounted value function
 * @param action Action for which to compute the z-values
 * @param valuefunction Value function over ALL states
 * @param discount Discount facto
 * @return The length of the zvalues is the same as the number of
 *          transitions with positive probabilities.
 */
inline numvec compute_zvalues(const Action& action, const numvec& valuefunction,
                              prec_t discount) {
    const numvec& rewards = action.get_rewards();
    const indvec& nonzero_indices = action.get_indices();

    numvec zvalues(rewards.size()); // values for individual states - used by nature.

#pragma omp simd
    for (size_t i = 0; i < rewards.size(); i++) {
        zvalues[i] = rewards[i] + discount * valuefunction[nonzero_indices[i]];
    }

    return zvalues;
}

/**
Computes an ambiguous value (e.g. robust) of the action, depending on the type
of nature that is provided.

@param action Action for which to compute the value
@param valuefunction State value function to use
@param discount Discount factor
@param nature Method used to compute the response of nature.
*/
inline vec_scal_t value_action(const Action& action, const numvec& valuefunction,
                               prec_t discount, long stateid, long actionid,
                               const SANature& nature) {
    numvec zvalues = compute_zvalues(action, valuefunction, discount);
    return nature(stateid, actionid, action.get_probabilities(), zvalues);
}

// *******************************************************
// State methods
// *******************************************************

/**
 * Constructs and returns a vector of nominal probabilities for each
 * state and positive transition probabilities.
 * @param state The state for which to compute the nominal probabilities
 * @return The length of the outer vector is the number of actions, the length
 *          of the inner vector is the number of non-zero transition
 * probabilities
 */
inline vector<numvec> compute_probabilities(const State& state) {
    vector<numvec> result;
    result.reserve(state.size());

    for (const auto& action : state.get_actions()) {
        result.push_back(action.get_probabilities());
    }
    return result;
}

/**
 * Constructs and returns a vector of z-values for each action in the state
 * @param state The state for which to compute the nominal probabilities
 * @param value function over the entire state space
 * @param discount The discount factor
 * @return The length of the outer vector is the number of actions, the length
 *          of the inner vector is the number of non-zero transition
 * probabilities
 */
inline vector<numvec> compute_zvalues(const State& state, const numvec& valuefunction,
                                      prec_t discount) {
    numvecvec result;
    result.reserve(state.size());

    for (const auto& action : state.get_actions()) {
        if (!action.is_valid()) throw invalid_argument("an action is invalid");
        result.push_back(compute_zvalues(action, valuefunction, discount));
    }
    return result;
}

/**
Computes the value of a fixed action and fixed response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actiondist Distribution over actions
@param distribution New distribution over states with non-zero nominal
probabilities

@return Value of state, 0 if it's terminal regardless of the action index
*/
inline prec_t value_fix_state(const State& state, numvec const& valuefunction,
                              prec_t discount, const numvec& actiondist,
                              const numvec& distribution) {
    // this is the terminal state, return 0
    if (state.is_terminal()) return 0;

    assert(actiondist.size() == state.size());
    assert((1.0 - accumulate(actiondist.cbegin(), actiondist.cend(), 0.0) - 1.0) < 1e-5);

    prec_t result = 0.0;
    for (size_t actionid = 0; actionid < state.size(); actionid++) {
        const auto& action = state[actionid];
        // cannot assume that the action is valid
        if (!state.is_valid(actionid))
            throw invalid_argument("Cannot take an invalid action");

        result += actiondist[actionid] *
                  value_action(action, valuefunction, discount, distribution);
    }
    return result;
}

}} // namespace craam::algorithms
