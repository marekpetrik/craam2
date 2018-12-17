#pragma once

#define GUROBI_USE

#include "craam/MDP.hpp"
#include "craam/MDPO.hpp"

#include <Rcpp.h>
#include <eigen3/Eigen/Dense>

#include <string>

/**
 * A very simple test MDP.
 */
inline craam::MDP create_test_mdp() {
    craam::MDP rmdp(3);

    // nonrobust and deterministic
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and rewards [0,0,1.1]
    // action 0 has a transition matrix [[1,0,0],[1,0,0], [0,1,0]] and rewards [0,1.0,1.0]
    add_transition(rmdp, 0, 1, 1, 1.0, 0.0);
    add_transition(rmdp, 1, 1, 2, 1.0, 0.0);
    add_transition(rmdp, 2, 1, 2, 1.0, 1.1);

    add_transition(rmdp, 0, 0, 0, 1.0, 0.0);
    add_transition(rmdp, 1, 0, 0, 1.0, 1.0);

    add_transition(rmdp, 2, 0, 1, 1.0, 1.0);

    return rmdp;
}

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
 * Parses a data frame  to an MDP
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idstateto, reward, probability.
 *              Multiple state-action-state rows have summed probabilities and averaged rewards.
 *
 * @returns Corresponding MDP definition
 */
inline craam::MDP mdp_from_dataframe(const Rcpp::DataFrame& data) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    craam::MDP m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idstateto[i],
                              probability[i], reward[i]);
    }
    return m;
}

/**
 * Parses a data frame to an MDPO (mdp with outcomes)
 *
 * Also checks whether the values passed are consistent with the MDPO definition.
 *
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idoutcome, idstateto, reward, probability.
 *              Multiple state-action-state rows have summed probabilities and averaged rewards.
 *
 * @returns Corresponding MDP definition
 */
inline craam::MDPO mdpo_from_dataframe(const Rcpp::DataFrame& data) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"], idoutcome = data["idoutcome"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    craam::MDPO m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idoutcome[i], idstateto[i],
                              probability[i], reward[i]);
    }
    return m;
}
