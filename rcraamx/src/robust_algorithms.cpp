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

#include "utils.hpp"

#include "craam/Samples.hpp"
#include "craam/Solution.hpp"
#include "craam/algorithms/nature_declarations.hpp"
#include "craam/algorithms/nature_response.hpp"
#include "craam/definitions.hpp"
#include "craam/optimization/optimization.hpp"
#include "craam/solvers.hpp"

#include "craam/algorithms/bellman_mdp.hpp"
#include "craam/algorithms/matrices.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

// [[Rcpp::depends(RcppProgress)]]
namespace RcppProg {
#include <progress.hpp>
}

using namespace craam;
using namespace std;

namespace defaults {
constexpr size_t iterations = 1000;
constexpr double maxresidual = 0.0001;
constexpr double timeout = -1;
constexpr bool show_progress = true;
constexpr long mpi_vi_count = 50;
} // namespace defaults

class ComputeProgress {
protected:
    /// maximum number of iterations
    size_t max_iterations;
    /// the target Bellman residual
    prec_t min_residual;
    /// Progress bar visualization
    RcppProg::Progress progress;
    /// Timing the computation
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    /// Timeout
    double timeout_seconds;
    /// Show progress
    bool show_progress;

public:
    /**
     * @param max_iterations Maximum number of iterations
     * @param min_residual Desired level of Bellman residual
     * @param timeout_seconds Number of seconds before a timeout. 0 or less means
     *                          that there is no timeout
     */
    ComputeProgress(size_t max_iterations, prec_t min_residual, bool show_progress,
                    double timeout_seconds)
        : max_iterations(max_iterations), min_residual(min_residual),
          progress(max_iterations, show_progress), timeout_seconds(timeout_seconds),
          show_progress(show_progress){};

    /**
     * Reports progress of the computation and listens to interrupt requests
     * @param iterations Current number of iterations
     * @param residual Current residual achieved

     * @return Whether to continue with the computation
     */
    bool operator()(size_t iterations, prec_t residual) {
        //std::cerr << iterations << "," << residual << std::endl;
        if (RcppProg::Progress::check_abort()) { return false; }
        if (timeout_seconds > craam::EPSILON) {
            auto finish = chrono::steady_clock::now();
            chrono::duration<double> duration = finish - start;
            if (duration.count() > timeout_seconds) {
                Rcpp::warning("Computation timed out.");
                return false;
            }
        }
        progress.update(iterations);
        return true;
    }
};

/// Report the status of the solution, and stops if the solution is incorrect
/// @param solution Could be any of the (deterministic or randomized) solutions from craam
/// @tparam S solution class, must support status, iterations, and time
template <class S> void report_solution_status(const S& solution) {
    if (solution.status != 0) {
        if (solution.status == 1) {
            Rcpp::warning("Ran out of time or iterations. The solution may be "
                          "suboptimal. Residual " +
                          to_string(solution.residual) +
                          ", Time: " + to_string(solution.time) +
                          ", Iterations: " + to_string(solution.iterations));
        } else if (solution.status == 2) {
            Rcpp::stop("Internal error, could not compute a solution.");
        } else {
            Rcpp::stop("Unknown error, solution not computed.");
        }
    }
}

//' Computes the maximum distribution subject to L1 constraints
//'
//' @param value Random variable (objective)
//' @param reference_dst Reference distribution of the same size as value
//' @param budget Maximum L1 distance from the reference dst
//'
//' @returns A list with dst as the worstcase distribution,
// '         and value as the objective
// [[Rcpp::export]]
Rcpp::List worstcase_l1(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst,
                        double budget) {

    craam::numvec vz(value.begin(), value.end()),
        vq(reference_dst.begin(), reference_dst.end());

    craam::numvec p;  // resulting probability
    double objective; // resulting objective value
    std::tie(p, objective) = craam::worstcase_l1(vz, vq, budget);

    Rcpp::List result;
    result["dst"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["value"] = objective;

    return result;
}

//' Computes average value at risk
//'
//' @param value Random variable (as a vector over realizations)
//' @param reference_dst Reference distribution of the same size as value
//' @param alpha Confidence value. 0 is worst case, 1 is average
//'
//' @returns A list with dst as the distorted distribution,
// '         and value as the avar value
// [[Rcpp::export]]
Rcpp::List avar(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst,
                double alpha) {

    craam::numvec p;  // resulting probability
    double objective; // resulting objective value

    craam::numvec vz(value.begin(), value.end()),
        vq(reference_dst.begin(), reference_dst.end());
    std::tie(p, objective) = craam::avar(vz, vq, alpha);

    Rcpp::List result;
    result["dst"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["value"] = objective;

    return result;
}

/**
 * Turns a deterministic policy to a dataframe with state and action
 * as the columns
 *
 * @param policy Deterministic policy
 * @return Dataframe with idstate, idaction columns (idstate is the index)
 */
Rcpp::DataFrame output_policy(const indvec& policy) {
    Rcpp::IntegerVector states(policy.size());
    std::iota(states.begin(), states.end(), 0);
    auto result = Rcpp::DataFrame::create(Rcpp::Named("idstate") = states,
                                          Rcpp::Named("idaction") = as_intvec(policy));
    return result;
}

/**
 * Turns a nested structure of state-action values to a dataframe with
 * state and action as the columns
 *
 * @param policy A state-action value
 * @param value_column Name of the value column
 * @return Dataframe with idstae, idaction, probability columns
 *          (idstate, idaction are the index)
 */
Rcpp::DataFrame output_sa_values(const numvecvec& values, const string& value_column) {
    Rcpp::IntegerVector states, actions;
    Rcpp::NumericVector values_c;
    for (size_t s = 0; s < values.size(); ++s) {
        for (size_t a = 0; a < values[s].size(); ++a) {
            states.push_back(s);
            actions.push_back(a);
            values_c.push_back(values[s][a]);
        }
    }
    auto result = Rcpp::DataFrame::create(Rcpp::Named("idstate") = states,
                                          Rcpp::Named("idaction") = actions,
                                          Rcpp::Named(value_column) = values_c);
    return result;
}

/**
 * Turns a randomized policy to a dataframe with state and action
 * as the columns
 *
 * @param policy A randomized policy
 * @return Dataframe with idstae, idaction, probability columns
 *          (idstate, idaction are the index)
 */
Rcpp::DataFrame output_policy(const numvecvec& policy) {
    return output_sa_values(policy, "probability");
}

/**
 * Turns a value function to a dataframe. This is safer than
 * just an array because it serves as a reminder that the
 * first state is 0 and not 1
 *
 * @param value
 * @return
 */
Rcpp::DataFrame output_value_fun(numvec value) {
    Rcpp::IntegerVector idstates(value.size());
    std::iota(idstates.begin(), idstates.end(), 0);

    return Rcpp::DataFrame::create(Rcpp::_["idstate"] = idstates,
                                   Rcpp::_["value"] = value);
}

/**
 * Packs MDP actions to be consequitive
 */
//[[Rcpp::export]]
Rcpp::List pack_actions(Rcpp::DataFrame mdp) {
    Rcpp::List result;

    MDP m = mdp_from_dataframe(mdp);
    result["action_mapping"] = m.pack_actions();

    result["mdp"] = mdp_to_dataframe(m);
    return result;
}

//' Solves a plain Markov decision process.
//'
//' This method supports only deterministic policies. See solve_mdp_rand for a
//' method that supports randomized policies.
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param discount Discount factor in [0,1]
//' @param algorithm One of "mpi", "vi", "vi_j", "pi". Also supports "lp"
//'           when Gurobi is properly installed
//' @param policy_fixed States for which the  policy should be fixed. This
//'          should be a dataframe with columns idstate and idaction. The policy
//'          is optimized only for states that are missing, and the fixed policy
//'          is used otherwise. Both indices are 0-based.
//' @param maxresidual Residual at which to terminate
//' @param iterations Maximum number of iterations
//' @param timeout Maximum number of secods for which to run the computation
//' @param value_init A  dataframe that contains the initial value function used
//'          to initialize the method. The columns should be idstate and value.
//'          Any states that are not provided are initialized to 0.
//' @param pack_actions Whether to remove actions with no transition probabilities,
//'          and rename others for the same state to prevent gaps. The policy
//'          for the original actions can be recovered using ``action_map'' frame
//'          in the result
//' @param output_tran Whether to construct and return a matrix of transition
//'          probabilites and a vector of rewards
//' @param show_progress Whether to show a progress bar during the computation
//' @return A list with value function policy and other values
// [[Rcpp::export]]
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm = "mpi",
                     Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                     double maxresidual = 10e-4, size_t iterations = 10000,
                     double timeout = 300,
                     Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                     bool pack_actions = false, bool output_tran = false,
                     bool show_progress = true) {

    if (policy_fixed.isNotNull() && pack_actions) {
        Rcpp::warning(
            "Providing a policy_fixed and packing actions is a bad idea. When the "
            "actions are re-indexed, the provided policy may become invalid.");
    }

    // parse MDP from the dataframe
    MDP m = mdp_from_dataframe(mdp);

    if (mdp.size() == 0) return Rcpp::List();

    // Construct the output (to get the output from pack actions)
    Rcpp::List result;
    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // Solution to be constructed and returned
    DetermSolution sol;

    // initialized policies from the parameter
    indvec policy =
        policy_fixed.isNotNull()
            ? parse_s_values<long>(m.size(), policy_fixed.get(), -1, "idaction")
            : indvec(0);

    // initialized value function from the parameters
    numvec vf_init = value_init.isNotNull()
                         ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value")
                         : numvec(0);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    if (algorithm == "mpi") {
        // Modified policy iteration
        sol = solve_mpi(m, discount, vf_init, policy, iterations, maxresidual,
                        defaults::mpi_vi_count, 0.9, progress);
    } else if (algorithm == "vi_j") {
        // Jacobian value iteration

        sol = solve_mpi(m, discount, vf_init, policy, iterations, maxresidual, 1, 0.9,
                        progress);
    } else if (algorithm == "vi") {
        // Gauss-seidel value iteration

        sol = solve_vi(m, discount, vf_init, policy, iterations, maxresidual, progress);

    } else if (algorithm == "pi") {
        // Gauss-seidel value iteration
        sol = solve_pi(m, discount, vf_init, policy, iterations, maxresidual, progress);
    }
#ifdef GUROBI_USE
    else if (algorithm == "lp") {
        // Gauss-seidel value iteration
        if (vf_init.size() > 0)
            Rcpp::warning(
                "The initial value function is ignored whem using linear programming.");
        sol = solve_lp(m, discount, policy);
    }
#endif // GUROBI_USE
    else {
        Rcpp::stop("Unknown or unsupported algorithm type.");
    }

    if (output_tran) {

        auto pb = craam::algorithms::PlainBellman(m);
        auto tmat = craam::algorithms::transition_mat(pb, sol.policy);
        auto rew = craam::algorithms::rewards_vec(pb, sol.policy);
        result["transitions"] = as_matrix(tmat);
        result["rewards"] = rew;
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;
    result["policy"] = output_policy(sol.policy);
    result["valuefunction"] = output_value_fun(move(sol.valuefunction));
    result["status"] = sol.status;
    report_solution_status(sol);
    return result;
}

//' Solves a plain Markov decision process with randomized policies.
//'
//' The method can be provided with a randomized policy for some states
//' and the output policy is randomized.
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param discount Discount factor in [0,1]
//' @param algorithm One of "mpi", "vi", "vi_j", "pi"
//' @param policy_fixed States for which the  policy should be fixed. This
//'         should be a dataframe with columns idstate, idaction, probability.
//'          The policy is optimized only for states that are missing, and the
//'          fixed policy is used otherwise
//' @param maxresidual Residual at which to terminate
//' @param iterations Maximum number of iterations
//' @param timeout Maximum number of secods for which to run the computation
//' @param output_tran Whether to construct and return a matrix of transition
//'          probabilites and a vector of rewards
//' @param show_progress Whether to show a progress bar during the computation
//'
//' @return A list with value function policy and other values
// [[Rcpp::export]]
Rcpp::List solve_mdp_rand(Rcpp::DataFrame mdp, double discount,
                          Rcpp::String algorithm = "mpi",
                          Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                          double maxresidual = 10e-4, size_t iterations = 10000,
                          double timeout = 300, bool output_tran = false,
                          bool show_progress = true) {

    // Note: this method only makes sense to be called with a provided
    // fixed policy and in that case, packing the actions is quite nonsensical

    // Construct the output (to get the output from pack actions)
    Rcpp::List result;

    MDP m = mdp_from_dataframe(mdp);

    // use one of the solutions, stochastic or deterministic
    RandSolution rsol;

    // initialized policies from the parameter
    numvecvec rpolicy =
        policy_fixed.isNotNull()
            ? parse_sa_values(m, Rcpp::as<Rcpp::DataFrame>(policy_fixed.get()), 0.0,
                              "probability")
            : numvecvec(0);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    if (algorithm == "mpi") {
        // Modified policy iteration
        rsol = solve_mpi_r(m, discount, numvec(0), rpolicy, iterations, maxresidual,
                           defaults::mpi_vi_count, 0.9, progress);

    } else if (algorithm == "vi_j") {
        // Jacobian value iteration
        rsol = solve_mpi_r(m, discount, numvec(0), rpolicy, iterations, maxresidual, 0,
                           0.9, progress);
    } else if (algorithm == "vi") {
        // Gauss-seidel value iteration
        rsol = solve_vi_r(m, discount, numvec(0), rpolicy, iterations, maxresidual,
                          progress);
    } else if (algorithm == "pi") {
        // Gauss-seidel value iteration
        rsol = solve_pi_r(m, discount, numvec(0), rpolicy, iterations, maxresidual,
                          progress);
    } else {
        Rcpp::stop("Unknown algorithm type.");
    }

    if (output_tran) {
        auto pb = craam::algorithms::PlainBellmanRand(m);
        auto tmat = craam::algorithms::transition_mat(pb, rsol.policy);
        auto rew = craam::algorithms::rewards_vec(pb, rsol.policy);
        result["transitions"] = as_matrix(tmat);
        result["rewards"] = rew;
    }

    result["iters"] = rsol.iterations;
    result["residual"] = rsol.residual;
    result["time"] = rsol.time;
    result["policy_rand"] = output_policy(rsol.policy);
    result["valuefunction"] = output_value_fun(move(rsol.valuefunction));
    result["status"] = rsol.status;
    report_solution_status(rsol);

    return result;
}

//' Computes the function for the MDP for the given value function and discount factor
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param discount Discount factor in [0,1]
//' @param valuefunction A dataframe representation of the value function. Each row
//'             represents a state. The columns must be idstate, value
//'
//' @return Dataframe with idstate, idaction, qvalue columns
// [[Rcpp::export]]
Rcpp::DataFrame compute_qvalues(Rcpp::DataFrame mdp, double discount,
                                Rcpp::DataFrame valuefunction) {

    MDP m = mdp_from_dataframe(mdp);
    Rcpp::IntegerVector states = valuefunction["idstate"];
    auto minmax_els = std::minmax_element(states.cbegin(), states.cend());

    if (int(m.size()) != (*minmax_els.second + 1)) {
        Rcpp::stop("The maximum idstate in valuefunction must be the same as the maximum "
                   "MDP state id.");
        return Rcpp::DataFrame();
    }

    if ((*minmax_els.first) != 0) {
        Rcpp::stop("The minimum idstate in valuefunction must be 0.");
        return Rcpp::DataFrame();
    }

    vector<numvec> qvalue = craam::algorithms::compute_qfunction(
        m, Rcpp::as<numvec>(valuefunction["value"]), discount);

    return output_sa_values(qvalue, "qvalue");
}

/**
 * Parses the name and the parameter of the provided nature
 */
algorithms::SANature parse_nature_sa(const MDP& mdp, const string& nature,
                                     SEXP nature_par) {
    if (nature == "l1u") {
        return algorithms::nats::robust_l1u(Rcpp::as<double>(nature_par));
    } else if (nature == "l1") {
        numvecvec values =
            parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0, "budget");
        return algorithms::nats::robust_l1(values);
    } else if (nature == "l1w") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),
                                       0.0, "budget");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight");
        return algorithms::nats::robust_l1w(budgets, weights);
    } else if (nature == "evaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_var_exp_u(Rcpp::as<double>(par["alpha"]),
                                                  Rcpp::as<double>(par["beta"]));
    } else if (nature == "eavaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_avar_exp_u(Rcpp::as<double>(par["alpha"]),
                                                   Rcpp::as<double>(par["beta"]));
    }
#ifdef GUROBI_USE
    else if (nature == "l1_g") {
        vector<numvec> values =
            parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0, "budget");
        return algorithms::nats::robust_l1w_gurobi(values);
    } else if (nature == "l1w_g") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),
                                       0.0, "budget");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight");
        return algorithms::nats::robust_l1w_gurobi(budgets, weights);
    }
#endif // end gurobi
    else {
        Rcpp::stop("unknown nature");
    }
}

/// Parses the name and the parameter of the provided nature
algorithms::SANature parse_nature_sa(const MDPO& mdpo, const string& nature,
                                     SEXP nature_par) {

    if (nature == "exp") {
        return algorithms::nats::robust_exp();
    } else if (nature == "evaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_var_exp_u(Rcpp::as<double>(par["alpha"]),
                                                  Rcpp::as<double>(par["beta"]));
    } else if (nature == "eavaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_avar_exp_u(Rcpp::as<double>(par["alpha"]),
                                                   Rcpp::as<double>(par["beta"]));
    } else {
        Rcpp::stop("unknown nature");
    }
}

//' Solves a robust Markov decision process with state-action rectangular
//' ambiguity sets. The worst-case is computed with the MDP transition
//' probabilities treated as nominal values.
//'
//' NOTE: The algorithms: pi, mpi may cycle infinitely without converging to a solution,
//' when solving a robust MDP.
//' The algorithms ppi and mppi are guaranteed to converge to an optimal solution.
//'
//' Important: Worst-case transitions are allowed only to idstateto states that
//' are provided in the mdp dataframe, even when the transition
//' probability to those states is 0.
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param discount Discount factor in [0,1]
//' @param nature Algorithm used to select the robust outcome. See details for options.
//' @param nature_par Parameters for the nature. Varies depending on the nature.
//'                   See details for options.
//' @param algorithm One of "ppi", "mppi", "vppi", "mpi", "vi", "vi_j", "pi". MPI and PI may
//'           may not converge
//' @param policy_fixed States for which the  policy should be fixed. This
//'          should be a dataframe with columns idstate and idaction. The policy
//'          is optimized only for states that are missing, and the fixed policy
//'          is used otherwise
//' @param maxresidual Residual at which to terminate
//' @param iterations Maximum number of iterations
//' @param timeout Maximum number of secods for which to run the computation
//' @param value_init A  dataframe that contains the initial value function used
//'          to initialize the method. The columns should be idstate and value.
//'          Any states that are not provided are initialized to 0.
//' @param pack_actions Whether to remove actions with no transition probabilities,
//'          and rename others for the same state to prevent gaps. The policy
//'          for the original actions can be recovered using ``action_map'' frame
//'          in the result
//' @param output_tran Whether to construct and return a matrix of transition
//'          probabilites and a vector of rewards
//' @param show_progress Whether to show a progress bar during the computation
//'
//' @return A list with value function policy and other values
//'
//' @details
//'
//' The options for nature and the corresponding nature_par are:
//'    \itemize{
//'         \item "l1u" an l1 ambiguity set with the same budget for all s,a.
//'                nature_par is a float number representing the budget
//'         \item "l1" an ambiguity set with different budgets for each s,a.
//'                nature_par is dataframe with idstate, idaction, budget
//'         \item "l1w" an l1-weighted ambiguity set with different weights
//'                      and budgets for each state and action
//'                 nature_par is a list with two elements: budgets, weights.
//'                 budgets must be a dataframe with columns idstate, idaction, budget
//'                 and weights must be a dataframe with columns:
//'                 idstatefrom, idaction, idstateto, weight (for the l1 weighted norms)
//'         \item "evaru" a convex combination of expectation and V@R over
//'                 transition probabilites. Uniform over all states and actions
//'                 nature_par is a list with parameters (alpha, beta). The worst-case
//'                 response is computed as:
//'                 beta * var [z] + (1-beta) * E[z], where
//'                 var is inf{x \in R : P[X <= x] >= alpha}, with alpha = 0 being the
//'                 worst-case.
//'         \item "evaru" a convex combination of expectation and AV@R over
//'                 transition probabilites. Uniform over states
//'                 nature_par is a list with parameters (alpha, beta). The worst-case
//'                 response is computed as:
//'                 beta * var [z] + (1-beta) * E[z], where
//'                 var is AVaR(z,alpha) = 1/alpha * ( E[X I{X <= x_a} ] + x_a (alpha - P[X <= x_a])
//'                 where I is the indicator function and
//'                 x_a = inf{x \in R : P[X <= x] >= alpha} being the
//'                 worst-case.
//'    }
// [[Rcpp::export]]
Rcpp::List rsolve_mdp_sa(Rcpp::DataFrame mdp, double discount, Rcpp::String nature,
                         SEXP nature_par, Rcpp::String algorithm = "mppi",
                         Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                         double maxresidual = 10e-4, size_t iterations = 10000,
                         double timeout = 300,
                         Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                         bool pack_actions = false, bool output_tran = false,
                         bool show_progress = true) {

    Rcpp::List result;

    // make robust transitions to states with 0 probability possible
    MDP m = mdp_from_dataframe(mdp, true);

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    indvec policy =
        policy_fixed.isNotNull()
            ? parse_s_values<long>(m.size(), policy_fixed.get(), -1, "idaction")
            : indvec(0);

    // initialized value function from the parameters
    numvec vf_init = value_init.isNotNull()
                         ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value")
                         : numvec(0);

    SARobustSolution sol;
    algorithms::SANature natparsed = parse_nature_sa(m, nature, nature_par);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    // the default method is to use ppa
    if (algorithm == "ppi") {
        sol = rsolve_ppi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                         maxresidual, progress);
    } else if (algorithm == "mppi") {
        sol = rsolve_mppi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                          maxresidual, progress);
    } else if (algorithm == "vppi") {
        sol = rsolve_vppi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                          maxresidual, progress);
    } else if (algorithm == "mpi") {
        Rcpp::warning("The robust version of the mpi method may cycle forever "
                      "without converging.");
        sol = rsolve_mpi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                         maxresidual, defaults::mpi_vi_count, 0.5, progress);
    } else if (algorithm == "vi") {
        sol = rsolve_vi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                        maxresidual, progress);
    } else if (algorithm == "vi_j") {
        // Jacobian value iteration, simulated using mpi
        sol = rsolve_mpi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                         maxresidual, 0, 0.5, progress);
    } else if (algorithm == "pi") {
        Rcpp::warning("The robust version of the pi method may cycle forever without "
                      "converging.");
        sol = rsolve_pi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                        maxresidual, progress);
    } else {
        Rcpp::stop("Unknown solver type.");
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;

    // split natures policy away
#if __cplusplus >= 201703L
    auto [dec_pol, nat_pol] = unzip(sol.policy);
#else
    craam::indvec dec_pol;
    std::vector<craam::numvec> nat_pol;
    std::tie(dec_pol, nat_pol) = unzip(sol.policy);
#endif

    if (output_tran) {
        auto pb = craam::algorithms::SARobustBellman(m, natparsed);
        auto tmat = craam::algorithms::transition_mat(pb, sol.policy);
        auto rew = craam::algorithms::rewards_vec(pb, sol.policy);
        result["transitions"] = as_matrix(tmat);
        result["rewards"] = rew;
    }

    result["policy"] = output_policy(dec_pol);
    result["nature"] = sanature_todataframe(m, dec_pol, nat_pol);
    result["valuefunction"] = output_value_fun(move(sol.valuefunction));
    result["status"] = sol.status;
    report_solution_status(sol);
    return result;
}

//' Solves a robust Markov decision process with state-action rectangular
//' ambiguity sets.
//'
//' The worst-case is computed across the outcomes and not
//' the actual transition probabilities.
//'
//' NOTE: The algorithms  mpi and pi may cycle infinitely without converging to a solution,
//' when solving a robust MDP.
//' The algorithms ppi and mppi are guaranteed to converge to an optimal solution.
//'
//'
//' @param algorithm One of "ppi", "mppi", "mpi", "vi", "vi_j", "pi". MPI may
//'           may not converge
//' @param policy_fixed States for which the  policy should be fixed. This
//'          should be a dataframe with columns idstate and idaction. The policy
//'          is optimized only for states that are missing, and the fixed policy
//'          is used otherwise
//' @param maxresidual Residual at which to terminate
//' @param iterations Maximum number of iterations
//' @param timeout Maximum number of secods for which to run the computation
//' @param value_init A  dataframe that contains the initial value function used
//'          to initialize the method. The columns should be idstate and value.
//'          Any states that are not provided are initialized to 0.
//' @param pack_actions Whether to remove actions with no transition probabilities,
//'          and rename others for the same state to prevent gaps. The policy
//'          for the original actions can be recovered using ``action_map'' frame
//'          in the result
//' @param output_tran Whether to construct and return a matrix of transition
//'          probabilites and a vector of rewards
//' @param show_progress Whether to show a progress bar during the computation
//'
//' @return A list with value function policy and other values
//'
//' @details
//'
//' The options for nature and the corresponding nature_par are:
//'    \itemize{
//'         \item "exp" plain expectation over the outcomes
//'         \item "evaru" a convex combination of expectation and V@R over
//'                 transition probabilites. Uniform over all states and actions
//'                 nature_par is a list with parameters (alpha, beta). The worst-case
//'                 response is computed as:
//'                 beta * var [z] + (1-beta) * E[z], where
//'                 var is \eqn{VaR(z,\alpha) = \inf{x \in R : P[X <= x] >= \alpha}}, with \eqn{\alpha = 0} being the
//'                 worst-case.
//'         \item "evaru" a convex combination of expectation and AV@R over
//'                 transition probabilites. Uniform over states
//'                 nature_par is a list with parameters (alpha, beta). The worst-case
//'                 response is computed as:
//'                 beta * var [z] + (1-beta) * E[z], where
//'                 var is \eqn{AVaR(z,alpha) =  1/alpha * ( E[X I{X <= x_a} ] + x_a (alpha - P[X <= x_a] )}
//'                 where I is the indicator function and
//'                 \eqn{x_a = \inf{x \in R : P[X <= x] >= \alpha}} being the
//'                 worst-case.
//'    }
// [[Rcpp::export]]
Rcpp::List rsolve_mdpo_sa(Rcpp::DataFrame mdpo, double discount, Rcpp::String nature,
                          SEXP nature_par, Rcpp::String algorithm = "mppi",
                          Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                          double maxresidual = 10e-4, size_t iterations = 10000,
                          double timeout = 300,
                          Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                          bool pack_actions = false, bool output_tran = false,
                          bool show_progress = true) {

    Rcpp::List result;

    // What would be the point of forcing to add transitions even if
    // the  probabilities are 0?
    // perhaps only if the RMDP is transformed to an MDP
    // that is why this is set to true for now .... it is also easy to remove the 0s from
    // the dataframe
    MDPO m = mdpo_from_dataframe(mdpo, true);

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    indvec policy = policy_fixed.isNotNull()
                        ? parse_s_values<long>(m.size(), policy_fixed, -1, "idaction")
                        : indvec(0);

    // initialized value function from the parameters
    numvec vf_init = value_init.isNotNull()
                         ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value")
                         : numvec(0);

    SARobustSolution sol;
    algorithms::SANature natparsed = parse_nature_sa(m, nature, nature_par);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    if (algorithm == "mppi") {
        sol = rsolve_mppi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                          maxresidual, progress);
    } else if (algorithm == "ppi") {
        sol = rsolve_ppi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                         maxresidual, progress);
    } else if (algorithm == "mpi") {
        Rcpp::warning("The robust version of the mpi method may cycle forever "
                      "without converging.");
        sol = rsolve_mpi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                         maxresidual, defaults::mpi_vi_count, 0.5, progress);
    } else if (algorithm == "vi") {
        sol = rsolve_vi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                        maxresidual, progress);
    } else if (algorithm == "vi_j") {
        // Jacobian value iteration, simulated using mpi
        sol = rsolve_mpi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                         maxresidual, 0, 0.5, progress);
    } else if (algorithm == "pi") {
        Rcpp::warning("The robust version of the pi method may cycle forever without "
                      "converging.");
        sol = rsolve_pi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                        maxresidual, progress);
    } else {
        Rcpp::stop("Unknown solver type.");
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;

#if __cplusplus >= 201703L
    auto [dec_pol, nat_pol] = unzip(sol.policy);
#else
    craam::indvec dec_pol;
    std::vector<craam::numvec> nat_pol;
    std::tie(dec_pol, nat_pol) = unzip(sol.policy);
#endif

    if (output_tran) {
        auto pb = craam::algorithms::SARobustOutcomeBellman(m, natparsed);
        auto tmat = craam::algorithms::transition_mat(pb, sol.policy);
        auto rew = craam::algorithms::rewards_vec(pb, sol.policy);
        result["transitions"] = as_matrix(tmat);
        result["rewards"] = rew;
    }

    result["policy"] = output_policy(dec_pol);
    result["nature"] = sanature_out_todataframe(m, dec_pol, nat_pol);
    result["valuefunction"] = output_value_fun(move(sol.valuefunction));
    result["status"] = sol.status;
    report_solution_status(sol);

    return result;
}

/**
 * Parses the name and the parameter of the provided nature
 */
algorithms::SNature parse_nature_s(const MDP& mdp, const string& nature,
                                   SEXP nature_par) {
    if (nature == "l1u") {
        return algorithms::nats::robust_s_l1u(Rcpp::as<double>(nature_par));
    }
    if (nature == "l1") {
        numvec values = parse_s_values<prec_t>(
            mdp.size(), Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0, "budget");
        return algorithms::nats::robust_s_l1(values);
    }
    if (nature == "l1w") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_s_values(
            mdp.size(), Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0, "budget");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight");
        return algorithms::nats::robust_s_l1w(budgets, weights);
    }
    // ----- gurobi only -----
#ifdef GUROBI_USE
    if (nature == "l1_g") {
        numvec values = parse_s_values(mdp.size(), Rcpp::as<Rcpp::DataFrame>(nature_par),
                                       0.0, "budget");
        return algorithms::nats::robust_s_l1_gurobi(values);
    }
    if (nature == "l1w_g") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_s_values(
            mdp.size(), Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0, "budget");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight");
        return algorithms::nats::robust_s_l1w_gurobi(budgets, weights);
    }
#endif
    // ---- end gurobi -----
    else {
        Rcpp::stop("unknown nature");
    }
}

//' Solves a robust Markov decision process with state rectangular
//' ambiguity sets. The worst-case is computed with the MDP transition
//' probabilities treated as nominal values.
//'
//' NOTE: The algorithms: pi, mpi may cycle infinitely without converging to a solution,
//' when solving a robust MDP.
//' The algorithms ppi and mppi are guaranteed to converge to an optimal solution.
//'
//' Important: Worst-case transitions are allowed only to idstateto states that
//' are provided in the mdp dataframe, even when the transition
//' probability to those states is 0.
//'
//' @param algorithm One of "ppi", "mppi", "mpi", "vi", "vi_j", "pi". MPI may
//'           may not converge
//' @param policy_fixed States for which the  policy should be fixed. This
//'          should be a dataframe with columns idstate and idaction. The policy
//'          is optimized only for states that are missing, and the fixed policy
//'          is used otherwise
//' @param maxresidual Residual at which to terminate
//' @param iterations Maximum number of iterations
//' @param timeout Maximum number of secods for which to run the computation
//' @param value_init A  dataframe that contains the initial value function used
//'          to initialize the method. The columns should be idstate and value.
//'          Any states that are not provided are initialized to 0.
//' @param pack_actions Whether to remove actions with no transition probabilities,
//'          and rename others for the same state to prevent gaps. The policy
//'          for the original actions can be recovered using ``action_map'' frame
//'          in the result
//' @param output_tran Whether to construct and return a matrix of transition
//'          probabilites and a vector of rewards
//' @param show_progress Whether to show a progress bar during the computation
//'
//' @return A list with value function policy and other values
//' @details
//'
//' The options for nature and the corresponding nature_par are:
//'    \itemize{
//'         \item "l1u" an l1 ambiguity set with the same budget for all s.
//'                nature_par is a float number representing the budget
//'         \item "l1" an ambiguity set with different budgets for each s.
//'                nature_par is dataframe with idstate, budget
//'         \item "l1w" an l1-weighted ambiguity set with different weights
//'                      and budgets for each state and action
//'                 nature_par is a list with two elements: budgets, weights.
//'                 budgets must be a dataframe with columns idstate, budget
//'                 and weights must be a dataframe with columns:
//'                 idstatefrom, idaction, idstateto, weight (for the l1 weighted norms)
//'    }
// [[Rcpp::export]]
Rcpp::List rsolve_mdp_s(Rcpp::DataFrame mdp, double discount, Rcpp::String nature,
                        SEXP nature_par, Rcpp::String algorithm = "mppi",
                        Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                        double maxresidual = 10e-4, size_t iterations = 10000,
                        double timeout = 300,
                        Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                        bool pack_actions = false, bool output_tran = false,
                        bool show_progress = true) {
    Rcpp::List result;

    MDP m = mdp_from_dataframe(mdp, true);

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    numvecvec rpolicy = policy_fixed.isNotNull()
                            ? parse_sa_values(m, policy_fixed, 0.0, "probability")
                            : numvecvec(0);

    // initialized value function from the parameters
    numvec vf_init = value_init.isNotNull()
                         ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value")
                         : numvec(0);

    craam::SRobustSolution sol;
    algorithms::SNature natparsed = parse_nature_s(m, nature, nature_par);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    if (algorithm == "ppi") {
        sol = rsolve_s_ppi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                             iterations, maxresidual, progress);
    } else if (algorithm == "mppi") {
        sol = rsolve_s_mppi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                              iterations, maxresidual, progress);
    } else if (algorithm == "vppi") {
        sol = rsolve_s_vppi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                              iterations, maxresidual, progress);
    } else if (algorithm == "mpi") {
        sol = rsolve_s_mpi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                             iterations / defaults::mpi_vi_count, maxresidual,
                             defaults::mpi_vi_count, 0.5, progress);
    } else if (algorithm == "vi") {
        sol = rsolve_s_vi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                            iterations, maxresidual, progress);
    } else if (algorithm == "vi_j") {
        // Jacobian value iteration, simulated using mpi
        sol = rsolve_s_mpi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                             iterations, maxresidual, 0, 0.5, progress);
    } else if (algorithm == "pi") {
        sol = rsolve_s_pi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                            iterations, maxresidual, progress);
    } else {
        Rcpp::stop("Unknown algorithm type: " + std::string(algorithm.get_cstring()));
    }
    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;
#if __cplusplus >= 201703L
    auto [dec_pol, nat_pol] = unzip(sol.policy);
#else
    std::vector<craam::numvec> dec_pol;
    std::vector<std::vector<craam::numvec>> nat_pol;
    std::tie(dec_pol, nat_pol) = unzip(sol.policy);
#endif

    if (output_tran) {
        auto pb = craam::algorithms::SRobustBellman(m, natparsed);
        auto tmat = craam::algorithms::transition_mat(pb, sol.policy);
        auto rew = craam::algorithms::rewards_vec(pb, sol.policy);
        result["transitions"] = as_matrix(tmat);
        result["rewards"] = rew;
    }

    result["policy_rand"] = output_policy(dec_pol);
    result["nature"] = sasnature_todataframe(m, nat_pol);
    result["valuefunction"] = output_value_fun(move(sol.valuefunction));
    result["status"] = sol.status;
    report_solution_status(sol);

    return result;
}

/**
 * Sets the number of threads for parallelization.
 */
// [[Rcpp::export]]
void set_rcraam_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    Rcpp::stop("Compiled without OPENMP support, cannot set the number of threads.");
#endif
}

//'  Builds an MDP from samples
//'
// [[Rcpp::export]]
Rcpp::DataFrame mdp_from_samples(Rcpp::DataFrame samples_frame) {
    Rcpp::IntegerVector idstatefrom = samples_frame["idstatefrom"],
                        idaction = samples_frame["idaction"],
                        idstateto = samples_frame["idstateto"];
    Rcpp::NumericVector reward = samples_frame["reward"];

    craam::msen::DiscreteSamples samples;

    // values from the last sample
    int last_step, last_state, last_action;
    double last_reward;

    for (int i = 0; i < samples_frame.nrows(); ++i) {
        samples.add_sample(idstatefrom[i], idaction[i], idstateto[i], reward[i], 1.0, i,
                           0);
    }

    craam::msen::SampledMDP smdp;
    smdp.add_samples(samples);

    return mdp_to_dataframe(*smdp.get_mdp());
}
