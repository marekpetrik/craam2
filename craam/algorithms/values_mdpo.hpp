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
 * Robust MDP methods for computing value functions.
 */
#pragma once

#include "craam/MDPO.hpp"
#include "craam/algorithms/nature_declarations.hpp"

namespace craam { namespace algorithms {

using namespace std;

// *******************************************************
// ActionO computation methods
// *******************************************************

/**
 * Computes the maximal outcome distribution constraints on the nature's
 * distribution. Does not work when the number of outcomes is zero.
 *
 * @param action Action for which the value is computed
 * @param valuefunction Value function reference
 * @param discount Discount factor
 * @param nature Method used to compute the response of nature.
 *
 * @return Outcome distribution and the mean value for the choice of the nature
 */
inline vec_scal_t value_action(const ActionO& action, const numvec& valuefunction,
                               prec_t discount, long stateid, long actionid,
                               const SANature& nature) {
    assert(action.get_distribution().size() == action.get_outcomes().size());
    if (action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");
    // loop over all outcomes
    numvec outcomevalues(action.size());
#pragma omp simd
    for (size_t i = 0; i < action.size(); i++)
        outcomevalues[i] = action[i].value(valuefunction, discount);
    return nature(stateid, actionid, action.get_distribution(), outcomevalues);
}

// *******************************************************
// Robust computation methods
// *******************************************************

/**
 * Computes the average outcome using the provided distribution.
 *
 * @param action Action for which the value is computed
 * @param valuefunction Updated value function
 * @param discount Discount factor
 * @return Mean value of the action
 */
inline prec_t value_action(const ActionO& action, numvec const& valuefunction,
                           prec_t discount) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if (action.get_outcomes().empty()) throw invalid_argument("ActionO with no outcomes");
    prec_t averagevalue = 0.0;
    const numvec& distribution = action.get_distribution();
#pragma omp simd reduction(+ : averagevalue)
    for (size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += distribution[i] * action[i].value(valuefunction, discount);
    return averagevalue;
}

/**
 * Computes the action value for a fixed index outcome.
 *
 * @param action Action for which the value is computed
 * @param valuefunction Updated value function
 * @param discount Discount factor
 * @param distribution Custom distribution that is selected by nature.
 * @return Value of the action
 */
inline prec_t value_action(const ActionO& action, numvec const& valuefunction,
                           prec_t discount, const numvec& distribution) {
    assert(distribution.size() == action.get_outcomes().size());
    if (action.get_outcomes().empty()) throw invalid_argument("ActionO with no outcomes");
    prec_t averagevalue = 0.0;
#pragma omp simd reduction(+ : averagevalue)
    for (size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += distribution[i] * action[i].value(valuefunction, discount);
    return averagevalue;
}

/**
 * The function computes the value of each outcome by adding the
 * reward function to the discounted value function
 * @param action Action for which to compute the z-values
 * @param valuefunction Value function over ALL states
 * @param discount Discount facto
 * @return The length of the zvalues is the same as the number of
 *          transitions with positive probabilities.
 */
inline numvec compute_zvalues(const ActionO& action, const numvec& valuefunction,
                              prec_t discount) {
    if (action.get_outcomes().empty()) throw invalid_argument("ActionO with no outcomes");
    numvec z_outcome(action.size());
#pragma omp simd
    for (size_t i = 0; i < action.size(); i++)
        z_outcome[i] = action[i].value(valuefunction, discount);
    return z_outcome;
}

// **********************************************************
// State methods
// **********************************************************

/**
 * Computes the value of a fixed action and fixed response of nature.
 *
 * Nature computes the average over outcomes, not states directly.
 *
 * @param state State to compute the value for
 * @param valuefunction Value function to use in computing value of states.
 * @param discount Discount factor
 * @param actiondist Distribution over actions
 * @param distribution New distribution over states with non-zero nominal
 * probabilities
 *
 * @return Value of state, 0 if it's terminal regardless of the action index
 */
inline prec_t value_fix_state(const StateO& state, numvec const& valuefunction,
                              prec_t discount, const numvec& actiondist,
                              const numvec& distribution) {
    // this is the terminal state, return 0
    if (state.is_terminal()) return 0;
    assert(actiondist.size() == state.size());
    assert((1.0 - accumulate(actiondist.cbegin(), actiondist.cend(), 0.0) - 1.0) < 1e-5);

    prec_t result = 0.0;
#pragma omp simd reduction(+ : result)
    for (size_t actionid = 0; actionid < state.size(); actionid++)
        result += actiondist[actionid] *
                  value_action(state[actionid], valuefunction, discount, distribution);
    return result;
}

/**
 * Constructs and returns a vector of nominal probabilities for each outcome.
 *
 * @param state The state for which to compute the nominal probabilities
 * @return The length of the outer vector is the number of actions, the length
 *          of the inner vector is the number of outcomes
 */
inline vector<numvec> compute_probabilities(const StateO& state) {
    vector<numvec> result;
    result.reserve(state.size());

    for (const auto& action : state.get_actions()) {
        result.push_back(action.get_distribution());
    }
    return result;
}

/**
 * Constructs and returns a vector of z-values for each action in the state.
 * The z-values are not over next states, but over outcomes.
 *
 * @param state The state for which to compute the nominal probabilities
 * @param value function over the entire state space
 * @param discount The discount factor
 * @return The length of the outer vector is the number of actions, the length
 *          of the inner vector is the number of non-zero transition
 *          probabilities
 */
inline vector<numvec> compute_zvalues(const StateO& state, const numvec& valuefunction,
                                      prec_t discount) {
    numvecvec result;
    result.reserve(state.size());

    for (const auto& action : state.get_actions()) {
        result.push_back(compute_zvalues(action, valuefunction, discount));
    }
    return result;
}

}} // namespace craam::algorithms
