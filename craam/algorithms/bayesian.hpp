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

#include "craam/MDP.hpp"
#include "craam/MDPO.hpp"

namespace craam { namespace bayes {

enum class Norm { L1, L2, Linf };

/**
 * Computes the size of credibility intervals for sa-rectangular ambiguity sets.
 *
 *
 *
 * This is the cridibility interval for all states and actions
 * independently. That mean that each individual level is built with
 * credibility level delta:
 *
 * level = 1 - 1/(1-delta)/(states-action pairs)
 *
 * This method assumes that the rewards are independent of the outcomes.
 *
 * @param mdpo MDP with outcomes
 * @param delta Confidence level between 0 and 1 for the probability of the robust
 *              solution being a lower bound (or the optimistic solution being an upper bound)
 * @param norm The type of the norm to use for the confidence interval
 *
 * @return An MDP with the nominal points and the appropriate size of the confidence intervals
 *          for each state and action
 */
pair<MDP, numvecvec> confidence_intervals_sa(const MDPO& mdpo, prec_t delta, Norm norm) {
    assert(delta >= 0.0 && delta <= 1.0);

    MDP nominal;
    numvecvec nature;

    // count the number of state action pairs
    size_t stateactioncount = 0;
    for (size_t s = 0; s < mdpo.size(); ++s) {
        stateactioncount += mdpo[s].size();
    }
    if (stateactioncount == 0) { throw invalid_argument("Cannot use an empty MDPO"); }

    // compute the confidence level for each state-action pair
    prec_t level = 1 - 1 / (1 - delta) / prec_t(stateactioncount);
    assert(level >= 0.0 && level <= 1.0);

    // iterate the size over all state-action pairs
    for (size_t si = 0; si < mdpo.size(); ++si) {
        const auto& state = mdpo[si];
        for (size_t ai = 0; ai < state.size(); ++ai) {
            const auto& action = state[ai];
            // weight of each sample
            prec_t weight = 1.0 / prec_t(action.size());

            // compute the mean transition over all outcomes

            // compute the distances of the outcomes
        }
    }
}

}} // namespace craam::bayes
