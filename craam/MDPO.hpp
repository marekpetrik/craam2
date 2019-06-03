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

#include "craam/ActionO.hpp"
#include "craam/GMDP.hpp"
#include "craam/State.hpp"

namespace craam {

/// State with uncertain outcomes with L1 constraints on the distribution
typedef SAState<ActionO> StateO;

/**
 *An uncertain MDP with outcomes and weights. See craam::L1RobustState.
*/
using MDPO = GMDP<StateO>;

/**
Adds a transition probability and reward for a particular outcome.

\param mdp model to add the transition to
\param fromid Starting state ID
\param actionid Action ID
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
*/
inline void add_transition(MDPO& mdp, long fromid, long actionid, long outcomeid,
                           long toid, prec_t probability, prec_t reward) {
    // make sure that the destination state exists
    mdp.create_state(toid);
    auto& state_from = mdp.create_state(fromid);
    auto& action = state_from.create_action(actionid);
    Transition& outcome = action.create_outcome(outcomeid);
    outcome.add_sample(toid, probability, reward);
}

/**
Sets the distribution for outcomes for each state and
action to be uniform.
*/
inline void set_uniform_outcome_dst(MDPO& mdp) {
    for (size_t si = 0; si < mdp.size(); ++si) {
        auto& s = mdp[si];
        for (size_t ai = 0; ai < s.size(); ++ai) {
            auto& a = s[ai];
            numvec distribution(a.size(), 1.0 / static_cast<prec_t>(a.size()));

            a.set_distribution(distribution);
        }
    }
}

/**
Sets the distribution of outcomes for the given state and action.
*/
inline void set_outcome_dst(MDPO& mdp, size_t stateid, size_t actionid,
                            const numvec& dist) {
    assert(stateid >= 0 && stateid < mdp.size());
    assert(actionid >= 0 && actionid < mdp[stateid].size());
    mdp[stateid][actionid].set_distribution(dist);
}

/**
Checks whether outcome distributions sum to 1 for all states and actions.

This function only applies to models that have outcomes, such as ones using
"ActionO" or its derivatives.

*/
inline bool is_outcome_dst_normalized(const MDPO& mdp) {
    for (size_t si = 0; si < mdp.size(); ++si) {
        auto& state = mdp[si];
        for (size_t ai = 0; ai < state.size(); ++ai) {
            if (!state[ai].is_distribution_normalized()) return false;
        }
    }
    return true;
}

/**
Normalizes outcome distributions for all states and actions.

This function only applies to models that have outcomes, such as ones using
"ActionO" or its derivatives.
*/
inline void normalize_outcome_dst(MDPO& mdp) {
    for (size_t si = 0; si < mdp.size(); ++si) {
        auto& state = mdp[si];
        for (size_t ai = 0; ai < state.size(); ++ai)
            state[ai].normalize_distribution();
    }
}

} // namespace craam
