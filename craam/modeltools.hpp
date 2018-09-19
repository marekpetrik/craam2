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

#include "craam/Action.hpp"
#include "craam/GMDP.hpp"
#include "craam/State.hpp"
#include "craam/Transition.hpp"
#include "craam/definitions.hpp"

#include <cassert>
#include <csv.h>
#include <fstream>
#include <functional>
#include <istream>
#include <memory>
#include <rm/range.hpp>
#include <sstream>
#include <string>
#include <vector>

// **********************************************************************
// ***********************    HELPER FUNCTIONS    ***********************
// **********************************************************************

namespace craam {

using namespace std;
using namespace util::lang;

/**
Adds uncertainty to a regular MDP. Turns transition probabilities to uncertain
outcomes and uses the transition probabilities as the nominal weights assigned
to the outcomes.

The input is an MDP:
\f$ \mathcal{M} = (\mathcal{S},\mathcal{A},P,r) ,\f$
where the states are \f$ \mathcal{S} = \{ s_1, \ldots, s_n \} \f$
The output MDPO is:
\f$ \bar{\mathcal{M}} = (\mathcal{S},\mathcal{A},\mathcal{B},
\bar{P},\bar{r},d), \f$ where the states and actions are the same as in the
original MDP and \f$ d : \mathcal{S} \times \mathcal{A} \rightarrow
\Delta^{\mathcal{B}} \f$ is the nominal probability of outcomes. Outcomes,
transition probabilities, and rewards depend on whether uncertain transitions to
zero-probability states are allowed:

When allowzeros = true, then \f$ \bar{\mathcal{M}} \f$ will also allow uncertain
transition to states that have zero probabilities in \f$ \mathcal{M} \f$.
- Outcomes are identical for all states and actions:
    \f$ \mathcal{B} = \{ b_1, \ldots, b_n \} \f$
- Transition probabilities are:
    \f$ \bar{P}(s_i,a,b_k,s_l) =  1 \text{ if } k = l, \text{ otherwise } 0  \f$
- Rewards are:
    \f$ \bar{r}(s_i,a,b_k,s_l) = r(s_i,a,s_k) \text{ if } k = l, \text{
otherwise } 0 \f$
- Nominal outcome probabilities are:
    \f$ d(s,a,b_k) = P(s,a,s_k) \f$

When allowzeros = false, then \f$ \bar{\mathcal{M}} \f$ will only allow
transitions to states that have non-zero transition probabilities in \f$
\mathcal{M} \f$. Let \f$ z_k(s,a) \f$ denote the \f$ k \f$-th state with a
non-zero transition probability from state \f$ s \f$ and action \f$ a \f$.
- Outcomes for \f$ s,a \f$ are:
    \f$ \mathcal{B}(s,a) = \{ b_1, \ldots, b_{|z(s,a)|} \}, \f$
    where \f$ |z(s,a)| \f$ is the number of positive transition probabilities in
\f$ P \f$.
- Transition probabilities are:
    \f$ \bar{P}(s_i,a,b_k,s_l) = 1 \text{ if } z_k(s_i,a) = l, \text{ otherwise
} 0  \f$
- Rewards are:
    \f$ \bar{r}(s_i,a,b_k,s_k) = r(s_i,a,s_{z_k(s_i,a)}) \f$
- Nominal outcome probabilities are:
    \f$ d(s,a,b_k) = P(s,a,z_k(s,a)) \f$

\param mdp MDP \f$ \mathcal{M} \f$ used as the input
\param allowzeros Whether to allow outcomes to states with zero
                    transition probability
\returns MDPO with nominal probabilitiesas weights
*/
inline MDPO robustify(const MDP& mdp, bool allowzeros = false) {
    // construct the result first
    MDPO rmdp;
    // iterate over all starting states (at t)
    for (size_t si : indices(mdp)) {
        const auto& s = mdp[si];
        auto& newstate = rmdp.create_state(si);
        for (size_t ai : indices(s)) {
            // make sure that the invalid actions are marked as such in the rmdp
            auto& newaction = newstate.create_action(ai);
            const Transition& t = s[ai];
            // iterate over transitions next states (at t+1) and add samples
            if (allowzeros) { // add outcomes for states with 0 transition probability
                numvec probabilities = t.probabilities_vector(mdp.state_count());
                numvec rewards = t.rewards_vector(mdp.state_count());
                for (size_t nsi : indices(probabilities)) {
                    // create the outcome with the appropriate weight
                    Transition& newoutcome =
                        newaction.create_outcome(newaction.size(), probabilities[nsi]);
                    // adds the single sample for each outcome
                    newoutcome.add_sample(nsi, 1.0, rewards[nsi]);
                }
            } else { // add outcomes only for states with non-zero probabilities
                for (size_t nsi : indices(t)) {
                    // create the outcome with the appropriate weight
                    Transition& newoutcome = newaction.create_outcome(
                        newaction.size(), t.get_probabilities()[nsi]);
                    // adds the single sample for each outcome
                    newoutcome.add_sample(t.get_indices()[nsi], 1.0,
                                          t.get_rewards()[nsi]);
                }
            }
        }
    }
    return rmdp;
}

} // namespace craam
