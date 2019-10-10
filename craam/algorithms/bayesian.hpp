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

/// A metric (like a norm) that is used to determine
/// the distance between two probability distributions
using Metric = std::function<prec_t(Transition, Transition)>;

/**
 * Computes the size of credible regions for sa-rectangular ambiguity sets.
 *
 * This is the credible region for all states and actions
 * independently. That mean that each individual level is built with
 * credibility level delta_s:
 *
 * delta_s = 1 - 1/(1-delta)/(states-action pairs)
 *
 * This approach uses the union bound assuming no dependence among the samples
 * across states and actions.
 *
 * The method also assumes that the rewards are independent of the outcomes. The
 * rewards for the MDP are constructed as a mean across the posterior samples.
 * If rewards depend on the outcomes, one workaround would be to add additional states
 * to represent the rewards only.
 *
 * The credible regions are built around a center point that is  computed to be
 * the mean of the posterior probability distribution.
 *
 * @param mdpo MDP with outcomes. Each outcome represents a sample of the
 *              transition probabilities from the Bayesian posterior distribution.
 * @param delta Confidence level between 0 and 1 for the probability of the robust
 *              solution being a lower bound (or the optimistic solution being an upper bound)
 * @param norm The type of the norm to use for the confidence interval. The metric
 *              must satisfy the triangle inequality and probably also needs to be
 *              symmetric. Norms like L1, L2, Linfty and their weghted versions are
 *              good choices. Something like KL-divergence is unclear.
 *
 * @return An MDP with the nominal points and the appropriate size of the confidence intervals
 *          for each state and action in the MDP
 */
pair<MDP, numvecvec> credible_regions_sa(const MDPO& mdpo, prec_t delta, Metric norm) {
    assert(delta >= 0.0 && delta <= 1.0);

    // construct output values
    MDP nominal;                           // nominal transition probabilities
    numvecvec budgets(mdpo.state_count()); // budgets computed for all states and actions

    // count the number of state action pairs
    size_t stateactioncount = 0;
    for (size_t s = 0; s < mdpo.size(); ++s) {
        stateactioncount += mdpo[s].size();
    }
    if (stateactioncount == 0) { throw invalid_argument("Cannot use an empty MDPO"); }

    // compute the confidence level for each state-action pair
    prec_t salevel = 1.0 - 1.0 / (1.0 - delta) / prec_t(stateactioncount);
    assert(salevel >= 0.0 && salevel <= 1.0);

    // iterate the size over all state-action pairs
    for (size_t si = 0; si < mdpo.size(); ++si) {
        const auto& state = mdpo[si];
        budgets[si] = numvec(state.size());

        for (size_t ai = 0; ai < state.size(); ++ai) {
            const auto& action = state[ai];
            // weight of each sample
            prec_t weight = 1.0 / prec_t(action.size());

            // ** compute the mean transition over all outcomes
            Transition nominal_tran;
            prec_t sample_weight =
                1 / prec_t(action.size()); // weight for each posterior sample, uniform
            // iterate over outcomes (bayesian samples)
            for (size_t oi = 0; oi < action.size(); ++oi) {
                const auto& outcome = action[oi];
                nominal_tran.probabilities_add(sample_weight, outcome);
            }

            // ** compute the distances of the outcomes, and the corresponding size
            numvec distances(action.size());
            for (size_t oi = 0; oi < action.size(); ++oi) {
                const auto& outcome = action[oi];
                distances[oi] = norm(nominal_tran, outcome);
            }
            // sort the distances
            sizvec distance_index = sort_indexes(distances);
            prec_t budget = distance_index[size_t(ceil(distances.size() * salevel))];

            // ** add value to the output
            nominal.create_state(si).create_action(ai) = move(nominal_tran);
            budgets[si][ai] = budget;
        }
    }
    return {nominal, budgets};
}
}} // namespace craam::bayes
