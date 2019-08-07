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

#include "config.hpp"

#include <Rcpp.h>
#include <eigen3/Eigen/Dense>

#include <string>

/**
 * Constructs a data frame from the MDP definition
 */
inline Rcpp::DataFrame mdp_to_dataframe(const craam::MDP& mdp) {
    craam::indvec idstatefrom, idaction, idstateto;
    craam::numvec probability, reward;

    for (size_t i = 0; i < mdp.size(); i++) {
        const auto& state = mdp[i];
        //idaction
        for (size_t j = 0; j < state.size(); j++) {
            const auto& tran = state[j];

            auto& indices = tran.get_indices();
            const auto& rewards = tran.get_rewards();
            const auto& probabilities = tran.get_probabilities();
            //idstateto
            for (size_t l = 0; l < tran.size(); l++) {
                idstatefrom.push_back(i);
                idaction.push_back(j);
                idstateto.push_back(indices[l]);
                probability.push_back(probabilities[l]);
                reward.push_back(rewards[l]);
            }
        }
    }

    // this is to make sure that the type on the dataframe is an integer
    auto idstatefrom_v =
        Rcpp::IntegerVector::import(idstatefrom.cbegin(), idstatefrom.cend());
    auto idaction_v = Rcpp::IntegerVector::import(idaction.cbegin(), idaction.cend());
    auto idstateto_v = Rcpp::IntegerVector::import(idstateto.cbegin(), idstateto.cend());
    auto probability_v =
        Rcpp::NumericVector::import(probability.cbegin(), probability.cend());
    auto reward_v = Rcpp::NumericVector::import(reward.cbegin(), reward.cend());

    return Rcpp::DataFrame::create(
        Rcpp::Named("idstatefrom") = idstatefrom_v, Rcpp::Named("idaction") = idaction_v,
        Rcpp::Named("idstateto") = idstateto_v,
        Rcpp::Named("probability") = probability_v, Rcpp::Named("reward") = reward_v);
}

/**
 * Constructs an R matrix from an Eigen matrix. All matrices are dense and the data is
 * copied. This method is not suitable for large matrices.
 */
inline Rcpp::NumericMatrix as_matrix(const Eigen::MatrixXd& matrix) {
    Rcpp::NumericMatrix result(matrix.rows(), matrix.cols());

    for (size_t i = 0; i < matrix.rows(); i++) {
        for (size_t j = 0; j < matrix.cols(); j++) {
            result(i, j) = matrix(i, j);
        }
    }
    return result;
}

/**
 * Parses a data frame  to an MDP.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with columns: idstatefrom, idaction, idstateto, reward, probability.
 *              Multiple state-action-state rows have summed probabilities and averaged rewards.
 * @param force Whether transitions with probability 0 should be focibly added to the transitions.
 *              This makes a difference with robust MDPs.
 *
 * @returns Corresponding MDP definition
 */
inline craam::MDP mdp_from_dataframe(const Rcpp::DataFrame& data, bool force = false) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    craam::MDP m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idstateto[i],
                              probability[i], reward[i], force);
    }
    return m;
}

/**
 * Parses a data frame  to an MDPO. Each outcome represents a possible outcome of nature
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with columns: idstatefrom, idaction, idoutcome, idstateto, reward, probability.
 *              Multiple state-action-outcome-state rows have summed probabilities and averaged rewards.
 * @param force Whether transitions with probability 0 should be focibly added to the transitions.
 *              This makes a difference with robust MDPs.
 *
 * @returns Corresponding MDPO definition
 */
inline craam::MDPO mdpo_from_dataframe(const Rcpp::DataFrame& data, bool force = false) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"], idoutcome = data["idoutcome"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    craam::MDPO m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idoutcome[i], idstateto[i],
                              probability[i], reward[i], force);
    }
    return m;
}

/**
 * Turns a dataframe `frame` to a matrix (array of arrays) of dimensions
 * dim1 x dim2. Index1 and index2 are the name of the columns with the
 * indices and value is the name of the value column. def_value is the
 * default value for any elements that are not provided.
 */
inline craam::numvecvec frame2matrix(const Rcpp::DataFrame& frame, uint dim1, uint dim2,
                                     const std::string& index1, const std::string& index2,
                                     const std::string& value, double def_value) {

    craam::numvecvec result(dim1);
    for (long i = 0; i < dim1; i++) {
        result[i] = craam::numvec(dim2, def_value);
    }

    Rcpp::IntegerVector idvec1 = frame[index1], idvec2 = frame[index2];
    Rcpp::NumericVector values = frame[value];

    for (long i = 0; i < idvec1.size(); i++) {
        long id1 = idvec1[i], id2 = idvec2[i];

        if (id1 < 0) Rcpp::stop("idstate must be non-negative");
        if (id1 > dim1)
            Rcpp::stop("idstate must be smaller than the number of MDP states");
        if (id2 < 0) Rcpp::stop("idaction must be non-negative");
        if (id2 > dim2)
            Rcpp::stop("idaction must be smaller than the number of actions for the "
                       "corresponding state");

        result[id1][id2] = values[i];
    }

    return result;
}

/**
 * Parses a data frame definition of values that correspond to states.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 2 comlumns, idstate, value. Here, idstate
 *              determines which value should be set.
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not
 *                  specify anything for the state action pair
 * @param value_column Name of the column with the value
 *
 * @tparam T Type of the value to parse
 *
 * @returns A vector over states with the included values
 */
template <class T>
inline std::vector<T> parse_s_values(size_t statecount, const Rcpp::DataFrame& frame,
                                     T def_value = 0,
                                     const std::string& value_column = "value") {
    std::vector<T> result(statecount);
    Rcpp::IntegerVector idstates = frame["idstate"];
    Rcpp::NumericVector values = frame[value_column];
    for (long i = 0; i < idstates.size(); i++) {
        long idstate = idstates[i];

        if (idstate < 0) Rcpp::stop("idstate must be non-negative");
        if (idstate > statecount)
            Rcpp::stop("idstate must be smaller than the number of MDP states");

        result[idstate] = values[i];
    }
    return result;
}

/**
* Parses a data frame definition of values that correspond to states and
* actions.
*
* Also checks whether the values passed are consistent with the MDP definition.
*
* @param mdp The definition of the MDP to know how many states and actions there are.
* @param frame Dataframe with 3 comlumns, idstate, idaction, value. Here, idstate and idaction
*              determine which value should be set.
*              Only the last value is used if multiple rows are present.
* @param def_value The default value for when frame does not specify anything for the state action pair
* @param val_name Name of the value column
*
* @returns A vector over states with an inner vector of actions
*/
inline craam::numvecvec parse_sa_values(const craam::MDP& mdp,
                                        const Rcpp::DataFrame& frame,
                                        double def_value = 0,
                                        const std::string val_name = "value") {

    craam::numvecvec result(mdp.size());
    for (long i = 0; i < mdp.size(); i++) {
        result[i] = craam::numvec(mdp[i].size(), def_value);
    }

    Rcpp::IntegerVector idstates = frame["idstate"], idactions = frame["idaction"];
    Rcpp::NumericVector values = frame[val_name];

    for (long i = 0; i < idstates.size(); i++) {
        long idstate = idstates[i], idaction = idactions[i];

        if (idstate < 0) Rcpp::stop("idstate must be non-negative");
        if (idstate > mdp.size())
            Rcpp::stop("idstate must be smaller than the number of MDP states");
        if (idaction < 0) Rcpp::stop("idaction must be non-negative");
        if (idaction > mdp[idstate].size())
            Rcpp::stop("idaction must be smaller than the number of actions for the "
                       "corresponding state");

        double value = values[i];
        result[idstate][idaction] = value;
    }

    return result;
}

/**
 * Parses a data frame definition of values that correspond to starting states, actions,
 * and taget states.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idstateto, value.
 *              Here, idstate(from,to) and idaction determine which value should be set
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not specify anything for the state action pair
 *
 * @returns A vector over states, action, with an inner vector of actions
 */
inline std::vector<craam::numvecvec> parse_sas_values(const craam::MDP& mdp,
                                                      const Rcpp::DataFrame& frame,
                                                      double def_value = 0) {

    std::vector<craam::numvecvec> result(mdp.size());
    for (long i = 0; i < mdp.size(); i++) {
        result[i] = craam::numvecvec(mdp[i].size());
        for (long j = 0; j < mdp[i].size(); j++) {
            // this is the number of non-zero transition probabilities
            result[i][j] = craam::numvec(mdp[i][j].size(), def_value);
        }
    }

    Rcpp::IntegerVector idstatesfrom = frame["idstatefrom"],
                        idactions = frame["idaction"], idstatesto = frame["idstateto"];
    Rcpp::NumericVector values = frame["value"];

    for (long i = 0; i < idstatesfrom.size(); i++) {
        long idstatefrom = idstatesfrom[i], idstateto = idstatesto[i],
             idaction = idactions[i];

        if (idstatefrom < 0) {
            Rcpp::warning("idstatefrom must be non-negative");
            continue;
        }
        if (idstatefrom > mdp.size()) {
            Rcpp::warning("idstatefrom must be smaller than the number of MDP states");
            continue;
        }
        if (idaction < 0) {
            Rcpp::warning("idaction must be non-negative");
            continue;
        }
        if (idaction > mdp[idstatefrom].size()) {
            Rcpp::warning("idaction must be smaller than the number of actions for the "
                          "corresponding state");
            continue;
        }
        if (idstateto < 0) {
            Rcpp::warning("idstateto must be non-negative");
            continue;
        }

        long indexto = mdp[idstatefrom][idaction].index_of(idstateto);
        //cout << idstatefrom << "," << idaction << "," << idstateto << "," << indexto
        //     << endl;

        if (indexto < 0) {
            Rcpp::warning("idstateto must be one of the states with non-zero probability."
                          "idstatefrom = " +
                          std::to_string(idstatefrom) +
                          ", idaction = " + std::to_string(idaction));
        } else {
            result[idstatefrom][idaction][indexto] = values[i];
        }
    }
    return result;
}
