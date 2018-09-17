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
#include "craam/RMDP.hpp"
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
Adds a transition probability and reward for an MDPO model.

\param mdp model to add the transition to
\param fromid Starting state ID
\param actionid Action ID
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
*/
inline void add_transition(MDP& mdp, long fromid, long actionid, long toid,
                           prec_t probability, prec_t reward) {
    // make sure that the destination state exists
    mdp.create_state(toid);
    auto& state_from = mdp.create_state(fromid);
    auto& action = state_from.create_action(actionid);
    action.add_sample(toid, probability, reward);
}

/**
Loads an MDP definition from a simple csv file. States, actions, and
outcomes are identified by 0-based ids. The columns are separated by
commas, and rows by new lines.

The file is formatted with the following columns:
idstatefrom, idaction, idstateto, probability, reward
The file must have a header.

 */
MDP mdp_from_csv(io::CSVReader<5> in) {
    long idstatefrom, idaction, idstateto;
    double probability, reward;

    MDP mdp;
    in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto",
                   "probability", "reward");
    bool read_row = in.read_row(idstatefrom, idaction, idstateto, probability, reward);
    do {
        add_transition(mdp, idstatefrom, idaction, idstateto, probability, reward);
        read_row = in.read_row(idstatefrom, idaction, idstateto, probability, reward);
    } while (read_row);
    return mdp;
}

MDP mdp_from_csv(const string& file_name) {
    return mdp_from_csv(io::CSVReader<5>(file_name));
}

MDP mdp_from_csv(istream& input) {
    return mdp_from_csv(io::CSVReader<5>("temp_file", input));
}

/**
Loads an MDPO definition from a simple csv file. States, actions, and
outcomes are identified by 0-based ids. The columns are separated by
commas, and rows by new lines.

The file is formatted with the following columns:
idstatefrom, idaction, idoutcome, idstateto, probability, reward
The file must have a header.

 */
MDPO mdpo_from_csv(io::CSVReader<6> in) {
    long idstatefrom, idaction, idoutcome, idstateto;
    double probability, reward;

    MDPO mdp;

    in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idoutcome",
                   "idstateto", "probability", "reward");
    bool read_row =
        in.read_row(idstatefrom, idaction, idoutcome, idstateto, probability, reward);
    do {
        add_transition(mdp, idstatefrom, idaction, idoutcome, idstateto, probability,
                       reward);
        read_row =
            in.read_row(idstatefrom, idaction, idoutcome, idstateto, probability, reward);
    } while (read_row);
    return mdp;
}

MDPO mdpo_from_csv(const string& file_name) {
    return mdpo_from_csv(io::CSVReader<6>(file_name));
}

MDPO mdpo_from_csv(istream& input) {
    return mdpo_from_csv(io::CSVReader<6>("temp_file", input));
}

/**
Saves the MDPO model to a stream as a simple csv file. States, actions, and
outcomes are identified by 0-based ids. Columns are separated by commas, and
rows by new lines.

The file is formatted with the following columns:
idstatefrom, idaction, idoutcome, idstateto, probability, reward

Exported and imported MDP will be be slightly different. Since
action/transitions will not be exported if there are no actions for the state.
However, when there is data for action 1 and action 3, action 2 will be created
with no outcomes.

Note that underlying nominal distributions are not saved.

\param output Output for the stream
\param header Whether the header should be written as the
      first line of the file represents the header.
*/
void to_csv(const MDPO& rmdp, ostream& output, bool header = true) {

    // write header if so requested
    if (header) {
        output << "idstatefrom,"
               << "idaction,"
               << "idoutcome,"
               << "idstateto,"
               << "probability,"
               << "reward" << endl;
    }

    // idstatefrom
    for (size_t i = 0l; i < rmdp.size(); i++) {
        const auto& actions = rmdp[i].get_actions();
        // idaction
        for (size_t j = 0; j < actions.size(); j++) {

            const auto& outcomes = actions[j].get_outcomes();
            // idoutcome
            for (size_t k = 0; k < outcomes.size(); k++) {
                const auto& tran = outcomes[k];

                auto& indices = tran.get_indices();
                const auto& rewards = tran.get_rewards();
                const auto& probabilities = tran.get_probabilities();
                // idstateto
                for (size_t l = 0; l < tran.size(); l++) {
                    output << i << ',' << j << ',' << k << ',' << indices[l] << ','
                           << probabilities[l] << ',' << rewards[l] << endl;
                }
            }
        }
    }
}

/**
Saves the MDP model to a stream as a simple csv file. States, actions, and
outcomes are identified by 0-based ids. Columns are separated by commas, and
rows by new lines.

The file is formatted with the following columns:
idstatefrom, idaction, idstateto, probability, reward

Exported and imported MDP will be be slightly different. Since
action/transitions will not be exported if there are no actions for the state.
However, when there is data for action 1 and action 3, action 2 will be created
with no outcomes, but will be marked as invalid in the state.

\param output Output for the stream
\param header Whether the header should be written as the
      first line of the file represents the header.
*/
void to_csv(const MDP& mdp, ostream& output, bool header = true) {
    // write header if so requested
    if (header) {
        output << "idstatefrom,"
               << "idaction,"
               << "idstateto,"
               << "probability,"
               << "reward" << endl;
    }

    // idstatefrom
    for (size_t i = 0l; i < mdp.size(); i++) {
        // idaction
        for (size_t j = 0; j < mdp[i].size(); j++) {
            const auto& tran = mdp[i][j];

            const auto& indices = tran.get_indices();
            const auto& rewards = tran.get_rewards();
            const auto& probabilities = tran.get_probabilities();
            // idstateto
            for (size_t l = 0; l < tran.size(); l++) {
                output << i << ',' << j << ',' << indices[l] << ',' << probabilities[l]
                       << ',' << rewards[l] << endl;
            }
        }
    }
}

/**
Saves the transition probabilities and rewards to a CSV file. See to_csv for
a detailed description.

\param filename Name of the file
\param header Whether to create a header of the file too
 */
template <class Model>
void to_csv_file(const Model& mdp, const string& filename, bool header = true) {
    ofstream ofs(filename, ofstream::out);
    to_csv(mdp, ofs, header);
    ofs.close();
}

/**
Sets the distribution for outcomes for each state and
action to be uniform.
*/
inline void set_uniform_outcome_dst(MDPO& mdp) {
    for (const auto si : indices(mdp)) {
        auto& s = mdp[si];
        for (const auto ai : indices(s)) {
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
    for (auto si : indices(mdp)) {
        auto& state = mdp[si];
        for (auto ai : indices(state)) {
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
    for (auto si : indices(mdp)) {
        auto& state = mdp[si];
        for (auto ai : indices(state))
            state[ai].normalize_distribution();
    }
}

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

/**
 * Creates a vector of vectors with one entry for each state and action
 *
 * @tparam T Type of the method output.
 *
 * @param mdp The mdp to map
 * @param fun Function that takes a state and action as an input
 */
template <class T, class Model>
inline vector<vector<T>> map_sa(const Model& mdp,
                                std::function<T(const State&, const Action&)> fun) {
    vector<vector<T>> statesres(mdp.size());
    for (size_t i = 0; i < mdp.size(); i++) {
        const State& s = mdp[i];
        statesres[i] = vector<T>(s.size());
        for (size_t j = 0; j < s.size(); j++) {
            statesres[i][j] = fun(s, s[j]);
        }
    }
    return statesres;
}

} // namespace craam
