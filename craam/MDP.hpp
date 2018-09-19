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

#include <csv.h>
#include <functional>

namespace craam {

/// Regular MDP state with no outcomes
typedef SAState<Action> State;

/**
 * Regular MDP with discrete actions and one outcome per action
 */
using MDP = GMDP<State>;

/**
Adds a transition probability and reward for an MDP model.

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
inline MDP mdp_from_csv(io::CSVReader<5> in) {
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

inline MDP mdp_from_csv(const string& file_name) {
    return mdp_from_csv(io::CSVReader<5>(file_name));
}

inline MDP mdp_from_csv(istream& input) {
    return mdp_from_csv(io::CSVReader<5>("temp_file", input));
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
inline void to_csv(const MDP& mdp, ostream& output, bool header = true) {
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

@param filename Name of the file
@param header Whether to create a header of the file too
 */
inline void to_csv_file(const MDP& mdp, const string& filename, bool header = true) {
    ofstream ofs(filename, ofstream::out);
    to_csv(mdp, ofs, header);
    ofs.close();
}

/**
 * Creates a vector of vectors with one entry for each state and action
 *
 * @tparam T Type of the method output.
 *
 * @param mdp The mdp to map
 * @param fun Function that takes a state and action as an input
 */
template <class T>
inline vector<vector<T>> map_sa(const MDP& mdp,
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
