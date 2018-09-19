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

#include "craam/MDPO.hpp"
#include "craam/algorithms/nature_declarations.hpp"

namespace craam { namespace algorithms {

/**
 * The class abstracts some operations of value / policy iteration in order to
 * generalize to various types of robust MDPs. It can be used in place of
 * response in mpi_jac or vi_gs to solve robust MDP objectives for s-rectangular
 * ambiguity. When a policy is specified for a given state then it evolves
 * simply according to the nominal transition probabilities.
 *
 * This Bellman update computes result as a worst case over a set of possible
 * choices of transition probabilities. Each of the available choices is
 * called **outcome**. The transition probabilities of the MDP itself are ignored.
 *
 * The policy consists an index for the state and distribution for the policy
 *
 * The update for a state value recomputes the worst case, which should ensure
 * the convergence of the modified policy policy iteration in robust cases.
 */
template <class SType = State> class SARobustOutcomeBellman {
public:
    /// Action index and distribution over outcomes
    using policy_type = std::pair<long, numvec>;

    /// Constructs an empty object
    SARobustOutcomeBellman() {}

    /**
         * @param responses Possible responses for each state and action
         */
    SARobustOutcomeBellman(SANature nature) : nature(move(nature)) {}

    /**
     * Computes the Bellman update.
     *
     * @param solution Solution to update
     * @param state State for which to compute the Bellman update
     * @param stateid Index of the state
     * @param valuefunction The full value function
     * @param discount Discount factor
     *
     * @returns Best value and action (decision maker and nature)
     */
    pair<prec_t, policy_type> policy_update(const SType& state, long stateid,
                                            const numvec& valuefunction,
                                            prec_t discount) const {
        assert(stateid >= 0 && stateid < responses.size());
        // can finish immediately when the state is terminal
        if (state.is_terminal()) return make_pair(-1, make_pair(-1, numvec()));

        // make sure that the number of natures is the same as the number of actions
        prec_t maxvalue = -numeric_limits<prec_t>::infinity();

        long result = -1;
        numvec result_outcome;

        for (size_t i = 0; i < state.size(); i++) {
            const auto& action = state[i];

            if (!state.is_valid(i))
                throw invalid_argument("Cannot have an invalid action.");

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

        return make_pair(maxvalue, make_pair(result, result_outcome));
    }

    /** Computes the Bellman update for a given policy.
            The function is called for the particular state. */
    prec_t compute_value(const policy_type& action_pol, const SType& state, long stateid,
                         const numvec& valuefunction, prec_t discount) const {
        assert(stateid >= 0 && stateid < responses.size());

        long actionid = action_pol.first;
        assert(actionid >= 0 && actionid < responses[stateid].size());

        // this is the terminal state, return 0
        if (state.is_terminal()) return 0;

        assert(actionid >= 0 && actionid < long(state.size()));

        if (actionid < 0 || actionid >= long(state.size()))
            throw range_error("invalid actionid: " + to_string(actionid) +
                              " for action count: " + to_string(state.size()));

        const auto& action = state[actionid];
        // cannot assume that the action is valid
        if (!state.is_valid(actionid))
            throw invalid_argument("Cannot take an invalid action");

        return value_action(action, valuefunction, discount, stateid, actionid, nature);
    }

    vector<vector<ActionO>>& get_responses() { return responses; }

protected:
    /// Lists of outcomes for each state and action
    vector<vector<ActionO>> responses;
    /// How to combine the values from a robust solution
    SANature nature;
};

}} // namespace craam::algorithms
