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
#include <csv.h>

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
Loads an MDPO definition from a simple csv file. States, actions, and
outcomes are identified by 0-based ids. The columns are separated by
commas, and rows by new lines.

The file is formatted with the following columns:
idstatefrom, idaction, idoutcome, idstateto, probability, reward
The file must have a header.

 */
inline MDPO mdpo_from_csv(io::CSVReader<6> in) {
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

inline MDPO mdpo_from_csv(const string& file_name) {
    return mdpo_from_csv(io::CSVReader<6>(file_name));
}

inline MDPO mdpo_from_csv(istream& input) {
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
inline void to_csv(const MDPO& rmdp, ostream& output, bool header = true) {

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
Saves the transition probabilities and rewards to a CSV file. See to_csv for
a detailed description.

@param filename Name of the file
@param header Whether to create a header of the file too
 */
inline void to_csv_file(const MDPO& mdp, const string& filename, bool header = true) {
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

} // namespace craam
