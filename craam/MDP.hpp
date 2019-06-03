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
#include "craam/State.hpp"

#include <functional>

namespace craam {

/// Regular MDP state with no outcomes
typedef SAState<Action> State;

/**
 * Regular MDP with discrete actions and one outcome per action
 */
using MDP = GMDP<State>;

/**
 * Adds a transition probability and reward for an MDP model.
 *
 * @param mdp model to add the transition to
 * @param fromid Starting state ID
 * @param actionid Action ID
 * @param toid Destination ID
 * @param probability Probability of the transition (must be non-negative)
 * @param reward The reward associated with the transition.
 * @param force Whether to force adding the probability even when it is 0 or even
 *                negative
*/
inline void add_transition(MDP& mdp, long fromid, long actionid, long toid,
                           prec_t probability, prec_t reward, bool force = false) {
    // make sure that the destination state exists
    mdp.create_state(toid);
    auto& state_from = mdp.create_state(fromid);
    auto& action = state_from.create_action(actionid);
    action.add_sample(toid, probability, reward, force);
}

} // namespace craam
