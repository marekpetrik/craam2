#pragma once

#define GUROBI_USE

#include "craam/MDP.hpp"

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

    return Rcpp::DataFrame::create(
        Rcpp::_["idstatefrom"] = idstatefrom, Rcpp::_["idaction"] = idaction,
        Rcpp::_["idstateto"] = idstateto, Rcpp::_["probability"] = probability,
        Rcpp::_["reward"] = reward);
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
