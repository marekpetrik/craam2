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

#include "craam/State.hpp"
#include "craam/algorithms/nature_declarations.hpp"

namespace craam { namespace algorithms {
// *******************************************************
// State computation methods
// *******************************************************

/**
Finds the action with the maximal average return. The return is 0 with no
actions. Such state is assumed to be terminal.

@param state State to compute the value for
@param valuefunction Value function to use for the following states
@param discount Discount factor

@return (Index of best action, value), returns 0 if the state is terminal.
*/
template <class AType>
inline pair<long, prec_t> value_max_state(const SAState<AType>& state,
                                          const numvec& valuefunction, prec_t discount) {
    if (state.is_terminal()) return make_pair(-1, 0.0);
    // skip invalid state.get_actions()

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;

    for (size_t i = 0; i < state.size(); i++) {
        auto const& action = state[i];

        if (!state.is_valid(i))
            throw invalid_argument("cannot have an invalid state and action");

        auto value = value_action(action, valuefunction, discount);
        if (value >= maxvalue) {
            maxvalue = value;
            result = i;
        }
    }

    // if the result has not been changed, that means that all actions are invalid
    if (result == -1) throw invalid_argument("all actions are invalid.");

    return make_pair(result, maxvalue);
}

/**
Computes the value of a fixed (and valid) action. Performs validity checks.

@param state State to compute the value for
@param valuefunction Value function to use for the following states
@param discount Discount factor

@return Value of state, 0 if it's terminal regardless of the action index
*/
template <class AType>
inline prec_t value_fix_state(const SAState<AType>& state, numvec const& valuefunction,
                              prec_t discount, long actionid) {
    // this is the terminal state, return 0
    if (state.is_terminal()) return 0;
    if (actionid < 0 || actionid >= (long)state.size())
        throw range_error("invalid actionid: " + to_string(actionid) +
                          " for action count: " + to_string(state.get_actions().size()));

    const auto& action = state[actionid];
    // cannot assume invalid state.get_actions()
    if (!state.is_valid(actionid))
        throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount);
}

/**
Computes the value of a fixed action and fixed response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actionid Action prescribed by the policy
@param distribution New distribution over states with non-zero nominal
probabilities

@return Value of state, 0 if it's terminal regardless of the action index
*/
template <class AType>
inline prec_t value_fix_state(const SAState<AType>& state, numvec const& valuefunction,
                              prec_t discount, long actionid, numvec distribution) {
    // this is the terminal state, return 0
    if (state.is_terminal()) return 0;

    assert(actionid >= 0 && actionid < long(state.size()));

    // if(actionid < 0 || actionid >= long(state.size())) throw
    // range_error("invalid actionid: "
    //    + to_string(actionid) + " for action count: " +
    //    to_string(state.get_actions().size()) );

    const auto& action = state[actionid];

    return value_action(action, valuefunction, discount, distribution);
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
template <class AType>
inline prec_t value_fix_state(const SAState<AType>& state, numvec const& valuefunction,
                              prec_t discount, numvec actiondist, numvec distribution) {
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

/**
Computes the value of a fixed action and fixed response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actiondist Distribution over actions
@param distributions New distribution over states with non-zero nominal
probabilities and for actions that have a positive actiondist probability

@return Value of state, 0 if it's terminal regardless of the action index
*/
template <class AType>
inline prec_t value_fix_state(const SAState<AType>& state, numvec const& valuefunction,
                              prec_t discount, numvec actiondist,
                              vector<numvec> distributions) {
    // this is the terminal state, return 0
    if (state.is_terminal()) return 0;

    assert(actiondist.size() == state.size());
    assert((1.0 - accumulate(actiondist.cbegin(), actiondist.cend(), 0.0) - 1.0) < 1e-5);
    assert(distributions.size() == actiondist.size());

    prec_t result = 0.0;
    for (size_t actionid = 0; actionid < state.size(); actionid++) {
        const auto& action = state[actionid];
        // cannot assume that the action is valid
        if (!state.is_valid(actionid))
            throw invalid_argument("Cannot take an invalid action");

        if (actiondist[actionid] <= EPSILON) continue;

        result += actiondist[actionid] *
                  value_action(action, valuefunction, discount, distributions[actionid]);
    }
    return result;
}

// *******************************************************
// State computation methods
// *******************************************************

/**
Computes the value of a fixed action and any response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actionid Which action to take
@param stateid Which state it is
@param nature Instance of a nature optimizer

\return Value of state, 0 if it's terminal regardless of the action index
*/
template <class SType>
inline vec_scal_t value_fix_state(const SType& state, numvec const& valuefunction,
                                  prec_t discount, long actionid, long stateid,
                                  const SANature& nature) {
    // this is the terminal state, return 0
    if (state.is_terminal()) return make_pair(numvec(0), 0);

    assert(actionid >= 0 && actionid < long(state.size()));

    if (actionid < 0 || actionid >= long(state.size()))
        throw range_error("invalid actionid: " + to_string(actionid) +
                          " for action count: " + to_string(state.get_actions().size()));

    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if (!state.is_valid(actionid))
        throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount, stateid, actionid, nature);
}

/**
  * Finds the greedy action and its value for the given value function.
  * This function assumes a robust or optimistic response by nature depending on the
  * provided ambiguity.
  *
  * When there are no actions, the state is assumed to be terminal and the return is
  * 0.
  * @param state State to compute the value for
  * @param valuefunction Value function to use in computing value of states.
  * @param discount Discount factor
  * @param stateid Number of the state in the MDP
  * @param natures Method used to compute the response of nature; one for each
  * action available in the state.
  *
  * \return (Action index, outcome index, value), 0 if it's terminal regardless of
  * the action index
  */
template <typename SType>
inline ind_vec_scal_t value_max_state(const SType& state, const numvec& valuefunction,
                                      prec_t discount, long stateid,
                                      const SANature& nature) {
    // can finish immediately when the state is terminal
    if (state.is_terminal()) return make_tuple(-1, numvec(), 0);

    // make sure that the number of natures is the same as the number of actions

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();

    long result = -1;
    numvec result_outcome;

    for (size_t i = 0; i < state.size(); i++) {
        const auto& action = state[i];

        if (!state.is_valid(i)) throw invalid_argument("Cannot have an invalid action.");

        auto value =
            value_action(action, valuefunction, discount, stateid, long(i), nature);
        if (value.second > maxvalue) {
            maxvalue = value.second;
            result = long(i);
            result_outcome = move(value.first);
        }
    }

    // if the result has not been changed, that means that all actions are invalid
    if (result == -1) throw invalid_argument("all actions are invalid.");

    return make_tuple(result, result_outcome, maxvalue);
}

}} // namespace craam::algorithms
