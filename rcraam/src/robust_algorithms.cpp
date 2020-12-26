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
#include "craam/algorithms/soft_robust.hpp"
#include "craam/definitions.hpp"
#include "craam/optimization/optimization.hpp"
#include "craam/solvers.hpp"

#include "craam/algorithms/bellman_mdp.hpp"
#include "craam/algorithms/matrices.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>

// [[Rcpp::depends(RcppProgress)]]
namespace RcppProg {
#include <progress.hpp>
}

using namespace craam;
using namespace std;

namespace defaults {
// these constants do not work because Rcpp cannot parse default
// parameters values then
//constexpr size_t iterations = 1000;
//constexpr double maxresidual = 0.0001;
//constexpr double timeout = -1;
//constexpr bool show_progress = true;
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
    /// Show progress level (verbosity)
    int show_progress;

public:
    /**
     * @param max_iterations Maximum number of iterations
     * @param min_residual Desired level of Bellman residual
     * @param timeout_seconds Number of seconds before a timeout. 0 or less means
     *                          that there is no timeout
     * @param show_progress The level of progress to report. Non-positive value indicates
     *        no progress shown, 1 means a progress bar, and 2 is detailed report for
     *        each iteration (and no progress bar)
     */
    ComputeProgress(size_t max_iterations, prec_t min_residual, int show_progress,
                    double timeout_seconds)
        : max_iterations(max_iterations), min_residual(min_residual),
          progress(max_iterations, show_progress == 1), timeout_seconds(timeout_seconds),
          show_progress(show_progress){};

    /**
     * Reports progress of the computation and listens to interrupt requests
     * @param iterations Current number of iterations
     * @param residual Current residual achieved

     * @return Whether to continue with the computation
     */
    bool operator()(size_t iterations, prec_t residual, const std::string& location,
                    const std::string& sublocation, const std::string& message) {

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
        if (show_progress == 2) {

            if (sublocation.length() == 0) {
                Rcpp::Rcout << iterations << ":[" << location << "]: " << residual << " "
                            << message << std::endl;
            } else {
                Rcpp::Rcout << "    " << iterations << ":[" << location << "-"
                            << sublocation << "]: " << residual << " " << message
                            << std::endl;
            }
        }
        return true;
    }
};

/// Report the status of the solution, and stops if the solution is incorrect
/// @param solution Could be any of the (deterministic or randomized) solutions from craam
/// @tparam S solution class, must support status, iterations, and time
template <class S> void report_solution_status(const craam::Solution<S>& solution) {
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

/// Report the status of the solution, and stops if the solution is incorrect
/// @param solution Could be any of the (deterministic or randomized) solutions from craam
/// @tparam S solution class, must support status, iterations, and time
template <class S> void report_solution_status(const craam::StaticSolution<S>& solution) {
    if (solution.status != 0) {
        if (solution.status == 1) {
            Rcpp::warning("Ran out of time or iterations. The solution may be "
                          "suboptimal. Time: " +
                          to_string(solution.time));
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
//'         and value as the objective
//[[Rcpp::export]]
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

//' Computes the maximum distribution subject to weighted L1 constraints
//'
//' @param value Random variable (objective)
//' @param reference_dst Reference distribution of the same size as value
//' @param budget Maximum L1 distance from the reference dst
//' @param w set of weights for ambiguity set
//'
//' @returns A list with dst as the worstcase distribution,
//'         and value as the objective
//[[Rcpp::export]]
Rcpp::List worstcase_l1_w(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst,
                          Rcpp::NumericVector w, double budget) {
    craam::numvec p;  // resulting probability
    double objective; // resulting objective value

    craam::numvec vz(value.begin(), value.end()),
        vq(reference_dst.begin(), reference_dst.end()), vw(w.begin(), w.end());
    std::tie(p, objective) = craam::worstcase_l1_w(vz, vq, vw, budget);

    Rcpp::List result;
    result["dst"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["value"] = objective;

    return result;
}

//' Computes the maximum distribution subject to weighted L1 constraints using Gurobi
//'
//' The function is only supported when the package is installed with Gurobi support
//'
//' @param value Random variable (objective)
//' @param reference_dst Reference distribution of the same size as value
//' @param budget Maximum L1 distance from the reference dst
//' @param w set of weights for ambiguity set
//'
//' @returns A list with dst as the worstcase distribution,
//'         and value as the objective
//[[Rcpp::export]]
Rcpp::List worstcase_l1_w_gurobi(Rcpp::NumericVector value,
                                 Rcpp::NumericVector reference_dst, Rcpp::NumericVector w,
                                 double budget) {
#ifdef GUROBI_USE
    craam::numvec p;  // resulting probability
    double objective; // resulting objective value

    craam::numvec vz(value.begin(), value.end()),
        vq(reference_dst.begin(), reference_dst.end()), vw(w.begin(), w.end());
    std::tie(p, objective) = craam::worstcase_l1_w_gurobi(vz, vq, vw, budget);

    Rcpp::List result;
    result["dst"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["value"] = objective;

    return result;
#else
    Rcpp::stop("The function is not supported because Gurobi is not installed");
#endif // GUROBI_USE
}

//' Computes the maximum distribution subject to weighted Linf constraints using Gurobi
//'
//' The function is only supported when the package is installed with Gurobi support
//'
//' @param value Random variable (objective)
//' @param reference_dst Reference distribution of the same size as value
//' @param budget Maximum Linf distance from the reference dst
//' @param w set of weights for ambiguity set
//'
//' @returns A list with dst as the worstcase distribution,
//'         and value as the objective
//[[Rcpp::export]]
Rcpp::List worstcase_linf_w_gurobi(Rcpp::NumericVector value,
                                   Rcpp::NumericVector reference_dst,
                                   Rcpp::NumericVector w, double budget) {
#ifdef GUROBI_USE
    craam::numvec p;  // resulting probability
    double objective; // resulting objective value

    craam::numvec vz(value.begin(), value.end()),
        vq(reference_dst.begin(), reference_dst.end()), vw(w.begin(), w.end());
    std::tie(p, objective) = craam::worstcase_linf_w_gurobi(vz, vq, vw, budget);

    Rcpp::List result;
    result["dst"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["value"] = objective;

    return result;
#else
    Rcpp::stop("The function is not supported because Gurobi is not installed");
#endif // GUROBI_USE
}

//' Computes average value at risk
//'
//' @param value Random variable (as a vector over realizations)
//' @param reference_dst Reference distribution of the same size as value
//' @param alpha Confidence value. 0 is worst case, 1 is average
//'
//' @returns A list with dst as the distorted distribution,
//'          and value as the avar value
//[[Rcpp::export]]
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
    craam::indvec states(policy.size(), 0);
    auto result = Rcpp::DataFrame::create(Rcpp::Named("idstate") = as_intvec(states),
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
    craam::indvec states, actions;
    craam::numvec values_c;

    for (size_t s = 0; s < values.size(); ++s) {
        for (size_t a = 0; a < values[s].size(); ++a) {
            states.push_back(s);
            actions.push_back(a);
            values_c.push_back(values[s][a]);
        }
    }
    auto result = Rcpp::DataFrame::create(Rcpp::Named("idstate") = as_intvec(states),
                                          Rcpp::Named("idaction") = as_intvec(actions),
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
 * @param value Value function for 0-based state space
 * @return Dataframe with columns idstate and value
 */
Rcpp::DataFrame output_value_fun(numvec value) {
    craam::indvec idstates(value.size(), 0);
    return Rcpp::DataFrame::create(Rcpp::_["idstate"] = as_intvec(idstates),
                                   Rcpp::_["value"] = value);
}

//' Packs MDP actions to be consequtive.
//'
//' If there is a state with actions where idaction = 0 and idaction = 2 and these
//' actions have transition probabilities associated with them, and idaction = 1 has
//' no transition probabilities, then idaction = 2 is renamed to idaction = 1.
//'
//' The result contains a mapping (action_mapping) dataframe that for
//' each state (idstate) contains an idaction_old which is the action before packing
//' and an idaction_new which is the id of the action after packing
//'
//' @param mdp Dataframe representation of the MDP, with columns
//'            idstatefrom, idaction, idstateto, probability, reward
//[[Rcpp::export]]
Rcpp::List pack_actions(Rcpp::DataFrame mdp) {
    MDP m = mdp_from_dataframe(mdp);
    return Rcpp::List::create(Rcpp::_["action_mapping"] = actionmap2df(m.pack_actions()),
                              Rcpp::_["mdp"] = mdp_to_dataframe(m));
}

//' Cleans the MDP dataframe
//'
//' Makes cosmetic changes to the MDP. It mostly aggregates transition probabilities
//' to the same state. When there are multiple rows that represent the same transition,
//' then it sums the probabilities and computes a weighted average of the rewards.
//'
//' The method achieves this by parsing and de-parsing the MDP definition.
//'
//' @param mdp Dataframe representation of the MDP, with columns
//'            idstatefrom, idaction, idstateto, probability, reward
//[[Rcpp::export]]
Rcpp::DataFrame mdp_clean(Rcpp::DataFrame mdp) {
    return mdp_to_dataframe(mdp_from_dataframe(mdp));
}

//' Solves a plain Markov decision process.
//'
//' This method supports only deterministic policies. See solve_mdp_rand for a
//' method that supports randomized policies.
//'
//' If the actions are packed then the mapping used internaly can be
//' computed by calling the function pack_actions on the dataframe passed
//' to this MDP
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param discount Discount factor in [0,1]
//' @param algorithm One of "mpi", "vi", "vi_j", "vi_g", "pi". Also supports "lp"
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
//'          in the result. The output policy is automatically remapped.
//' @param show_progress Whether to show a progress bar during the computation.
//'         0 means no progress, 1 is progress bar, and 2 is a detailed report
//' @return A list with value function policy and other values
// [[Rcpp::export]]
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm = "mpi",
                     Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                     double maxresidual = 10e-4, size_t iterations = 10000,
                     double timeout = 300,
                     Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                     bool pack_actions = false, int show_progress = 1) {
    if (policy_fixed.isNotNull() && pack_actions) {
        Rcpp::warning("Providing a policy_fixed and setting pack_actions = true is a bad "
                      "idea. When the actions are re-indexed, the provided policy may "
                      "become invalid.");
    }

    // parse MDP from the dataframe
    MDP m = mdp_from_dataframe(mdp);
    if (m.size() == 0) return Rcpp::List();

    // Construct the output (to get the output from pack actions)
    Rcpp::List result;
    std::optional<std::vector<craam::indvec>> actionmap;
    // remove actions that are not being used
    if (pack_actions) {
        actionmap = m.pack_actions();
        // do not report the value to prevent confusion
        //result["action_map"] = actionmap2df(m.pack_actions());
    }

    // Solution to be constructed and returned
    DetermSolution sol;

    // initialized policies from the parameter
    indvec policy = policy_fixed.isNotNull()
                        ? parse_s_values<long>(m.size(), policy_fixed.get(), -1,
                                               "idaction", "policy_fixed")
                        : indvec(0);

    // initialized value function from the parameters
    numvec vf_init =
        value_init.isNotNull()
            ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value", "value_init")
            : numvec(0);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    if (algorithm == "mpi") {
        // Modified policy iteration
        sol = solve_mpi(m, discount, vf_init, policy, iterations, maxresidual,
                        defaults::mpi_vi_count, 0.9, progress);
    } else if (algorithm == "vi_j" || algorithm == "vi") {
        // Jacobian value iteration
        sol = solve_mpi(m, discount, vf_init, policy, iterations, maxresidual, 1, 0.9,
                        progress);
    } else if (algorithm == "vi_g") {
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

    // check if we need to remap the actions if they were packed
    if (actionmap.has_value()) {
        craam::indvec mapped_policy(sol.policy.size(), -1);
        // policy has the id of the action for each state
        for (std::size_t istate = 0; istate < sol.policy.size(); ++istate) {
            // make sure to handle policy values -1 for terminal states!
            if (sol.policy[istate] >= 0) // not terminal
                mapped_policy[istate] = (actionmap->at(istate)).at(sol.policy[istate]);
            else
                mapped_policy[istate] = -1; // terminal
        }
        result["policy"] = output_policy(mapped_policy);
    } else {
        result["policy"] = output_policy(sol.policy);
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;
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
//' @param algorithm One of "mpi", "vi", "vi_j", "vi_g", "pi"
//' @param policy_fixed States for which the  policy should be fixed. This
//'         should be a dataframe with columns idstate, idaction, probability.
//'          The policy is optimized only for states that are missing, and the
//'          fixed policy is used otherwise
//' @param maxresidual Residual at which to terminate
//' @param iterations Maximum number of iterations
//' @param timeout Maximum number of secods for which to run the computation
//' @param value_init A  dataframe that contains the initial value function used
//'          to initialize the method. The columns should be idstate and value.
//'          Any states that are not provided are initialized to 0.
//' @param show_progress Whether to show a progress bar during the computation
//'         0 means no progress, 1 is progress bar, and 2 is a detailed report
//'
//' @return A list with value function policy and other values
// [[Rcpp::export]]
Rcpp::List solve_mdp_rand(Rcpp::DataFrame mdp, double discount,
                          Rcpp::String algorithm = "mpi",
                          Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                          double maxresidual = 10e-4, size_t iterations = 10000,
                          double timeout = 300,
                          Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                          int show_progress = 1) {
    // Note: this method only makes sense to be called with a provided
    // fixed policy and in that case, packing the actions is quite nonsensical

    // Construct the output (to get the output from pack actions)
    Rcpp::List result;

    MDP m = mdp_from_dataframe(mdp);
    if (m.size() == 0) return Rcpp::List();

    // use one of the solutions, stochastic or deterministic
    RandSolution rsol;

    // initialized policies from the parameter
    numvecvec rpolicy =
        policy_fixed.isNotNull()
            ? parse_sa_values(m, Rcpp::as<Rcpp::DataFrame>(policy_fixed.get()), 0.0,
                              "probability", "policy_fixed")
            : numvecvec(0);

    // initialized value function from the parameters
    numvec vf_init =
        value_init.isNotNull()
            ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value", "value_init")
            : numvec(0);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    if (algorithm == "mpi") {
        // Modified policy iteration
        rsol = solve_mpi_r(m, discount, vf_init, rpolicy, iterations, maxresidual,
                           defaults::mpi_vi_count, 0.9, progress);

    } else if (algorithm == "vi_j" || algorithm == "vi") {
        // Jacobian value iteration
        rsol = solve_mpi_r(m, discount, vf_init, rpolicy, iterations, maxresidual, 0, 0.9,
                           progress);
    } else if (algorithm == "vi_g") {
        // Gauss-seidel value iteration
        rsol =
            solve_vi_r(m, discount, vf_init, rpolicy, iterations, maxresidual, progress);
    } else if (algorithm == "pi") {
        // Policy iteration
        rsol =
            solve_pi_r(m, discount, vf_init, rpolicy, iterations, maxresidual, progress);
    } else
        Rcpp::stop("Unknown algorithm type.");

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
        numvecvec values = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par),
                                           0.0, "budget", "nature_par");
        return algorithms::nats::robust_l1(values);
    } else if (nature == "l1w") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),
                                       0.0, "budget", "budgets");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight", "weights");
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
        vector<numvec> values = parse_sa_values(
            mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0, "budget", "budget");
        return algorithms::nats::robust_l1w_gurobi(values);
    } else if (nature == "l1w_g") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),
                                       0.0, "budget", "budgets");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight", "weights");
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
//' @param algorithm One of "ppi", "mppi", "vppi", "mpi", "vi", "vi_j", "vi_g", "pi". MPI and PI may
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
//' @param show_progress Whether to show a progress bar during the computation.
//'         0 means no progress, 1 is progress bar, and 2 is a detailed report
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
//'                 var is inf{x in R : P[X <= x] >= alpha}, with alpha = 0 being the
//'                 worst-case.
//'         \item "evaru" a convex combination of expectation and AV@R over
//'                 transition probabilites. Uniform over states
//'                 nature_par is a list with parameters (alpha, beta). The worst-case
//'                 response is computed as:
//'                 beta * var [z] + (1-beta) * E[z], where
//'                 var is AVaR(z,alpha) = 1/alpha * ( E[X I{X <= x_a} ] + x_a (alpha - P[X <= x_a])
//'                 where I is the indicator function and
//'                 x_a = inf{x in R : P[X <= x] >= alpha} being the
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
                         int show_progress = 1) {
    Rcpp::List result;

    // make robust transitions to states with 0 probability possible
    MDP m = mdp_from_dataframe(mdp, true);
    if (m.size() == 0) return Rcpp::List();

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    indvec policy = policy_fixed.isNotNull()
                        ? parse_s_values<long>(m.size(), policy_fixed.get(), -1,
                                               "idaction", "policy_fixed")
                        : indvec(0);

    // initialized value function from the parameters
    numvec vf_init =
        value_init.isNotNull()
            ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value", "value_init")
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
    } else if (algorithm == "vi_g") {
        sol = rsolve_vi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                        maxresidual, progress);
    } else if (algorithm == "vi_j" || algorithm == "vi") {
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
//' @param algorithm One of "ppi", "mppi", "mpi", "vi", "vi_j", "vi_g", "pi". MPI may
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
//' @param show_progress Whether to show a progress bar during the computation.
//'         0 means no progress, 1 is progress bar, and 2 is a detailed report
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
//'         \item "eavaru" a convex combination of expectation and AV@R over
//'                 transition probabilites. Uniform over states
//'                 nature_par is a list with parameters (alpha, beta). The worst-case
//'                 response is computed as:
//'                 beta * avar [z] + (1-beta) * E[z], where
//'                 avar is \eqn{AVaR(z,alpha) =  1/alpha * ( E[X I{X <= x_a} ] + x_a (alpha - P[X <= x_a] )}
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
                          int show_progress = 1) {
    Rcpp::List result;

    // What would be the point of forcing to add transitions even if
    // the  probabilities are 0?
    // perhaps only if the RMDP is transformed to an MDP
    // that is why this is set to true for now .... it is also easy to remove the 0s from
    // the dataframe
    MDPO m = mdpo_from_dataframe(mdpo, true);
    if (m.size() == 0) return Rcpp::List();

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    indvec policy = policy_fixed.isNotNull()
                        ? parse_s_values<long>(m.size(), policy_fixed.get(), -1,
                                               "idaction", "policy_fixed")
                        : indvec(0);

    // initialized value function from the parameters
    numvec vf_init =
        value_init.isNotNull()
            ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value", "value_init")
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
    } else if (algorithm == "vi_g") {
        sol = rsolve_vi(m, discount, std::move(natparsed), vf_init, policy, iterations,
                        maxresidual, progress);
    } else if (algorithm == "vi_j" || algorithm == "vi") {
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

//' Solves an MDPO with static uncertainty using a non-convex global optimization method.
//'
//' The objective is:
//'  \deqn{\max_{\pi} \beta * CVaR_{P \sim f}^\alpha [return(\pi,P)] + (1-\beta) * E_{P \sim f}^\alpha [return(\pi,P)]}
//'
//' The parameters for the optimizer (such as the time to run, or the outputs to show) can be set
//' by calling `gurobi_set_param` and passing the value "nonconvex" to the `optimizer` argument.
//'
//' @param mdpo Uncertain MDP. The outcomes are assumed to represent the uncertainty over MDPs.
//'              The number of outcomes must be uniform for all states and actions
//'              (except for terminal states which have no actions).
//' @param alpha Risk level of avar (0 = worst-case). The minimum value is 1e-5, the maximum
//'              value is 1.
//' @param beta Weight on AVaR and the complement (1-beta) is the weight
//'              on the expectation term. The value must be between 0 and 1.
//' @param discount Discount factor. Clamped to be in [0,1]
//' @param init_distribution Initial distribution over states. The columns should be
//'                             are idstate, and probability.
//' @param model_distribution Distribution over the models. The default is empty, which translates
//'                   to a uniform distribution. The columns should be idstate, and probablity.
//' @param output_filename Name of the file to save the model output. Valid suffixes are
//'                          .mps, .rew, .lp, or .rlp for writing the model itself.
//'                        If it is an empty string, then it does not write the file.
//'
//' @return Returns a list with policy, objective (return), time (computation),
//'               status (whether it is optimal, directly passed from gurobi)
// [[Rcpp::export]]
Rcpp::List srsolve_mdpo(Rcpp::DataFrame mdpo, Rcpp::DataFrame init_distribution,
                        double discount, double alpha, double beta,
                        Rcpp::String algorithm = "milp",
                        Rcpp::Nullable<Rcpp::DataFrame> model_distribution = R_NilValue,
                        Rcpp::String output_filename = "") {
#ifdef GUROBI_USE
    Rcpp::List result;

    // What would be the point of forcing to add transitions even if
    // the  probabilities are 0?
    // perhaps only if the RMDP is transformed to an MDP
    // that is why this is set to true for now .... it is also easy to remove the 0s from
    // the dataframe
    MDPO m = mdpo_from_dataframe(mdpo, true);

    const ProbDst init_dst = parse_s_values(m.size(), init_distribution, 0.0,
                                            "probability", "init_distribution");
    const ProbDst model_dst =
        model_distribution.isNotNull()
            ? parse_s_values(m.size(), model_distribution.get(), 0.0, "probability",
                             "model_distribution")
            : ProbDst(0);

    auto grb = craam::get_gurobi(craam::OptimizerType::NonconvexOptimization);
    if (algorithm == "milp") {
        const craam::DetStaticSolution sol = craam::statalgs::srsolve_avar_milp(
            *grb, m, alpha, beta, discount, init_dst, model_dst, output_filename);

        result["policy"] = output_policy(sol.policy);
        result["objective"] = sol.objective;
        result["time"] = sol.time;
        result["status"] = sol.status;

        report_solution_status(sol);
    } else if (algorithm == "quadratic") {
        const craam::RandStaticSolution sol = craam::statalgs::srsolve_avar_quad(
            *grb, m, alpha, beta, discount, init_dst, model_dst, output_filename);

        result["policy_rand"] = output_policy(sol.policy);
        result["objective"] = sol.objective;
        result["time"] = sol.time;
        result["status"] = sol.status;

        report_solution_status(sol);
    } else {
        Rcpp::stop("Unknown algorithm");
    }

    return result;
#else
    Rcpp::stop("Not supported without gurobi support");
#endif
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
            mdp.size(), Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0, "budget", "budgets");
        return algorithms::nats::robust_s_l1(values);
    }
    if (nature == "l1w") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets =
            parse_s_values(mdp.size(), Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0,
                           "budget", "budgets");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight", "weights");
        return algorithms::nats::robust_s_l1w(budgets, weights);
    }
    // ----- gurobi only -----
#ifdef GUROBI_USE
    if (nature == "l1_g") {
        numvec values = parse_s_values(mdp.size(), Rcpp::as<Rcpp::DataFrame>(nature_par),
                                       0.0, "budget", "budget");
        return algorithms::nats::robust_s_l1_gurobi(values);
    }
    if (nature == "l1w_g") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets =
            parse_s_values(mdp.size(), Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0,
                           "budget", "budgets");
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]),
                                        1.0, "weight", "weights");
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
//' @param algorithm One of "ppi", "mppi", "mpi", "vi", "vi_j", "v_g", "pi". MPI may
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
//' @param show_progress Whether to show a progress bar during the computation.
//'         0 means no progress, 1 is progress bar, and 2 is a detailed report
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
                        int show_progress = 1) {
    Rcpp::List result;

    MDP m = mdp_from_dataframe(mdp, true);
    if (m.size() == 0) return Rcpp::List();

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    // rpolicy stands for randomize policy and NOT robust policy
    numvecvec rpolicy =
        policy_fixed.isNotNull()
            ? parse_sa_values(m, policy_fixed.get(), 0.0, "probability", "policy_fixed")
            : numvecvec(0);

    // initialized value function from the parameters
    numvec vf_init =
        value_init.isNotNull()
            ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value", "value_init")
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
    } else if (algorithm == "vi_g") {
        sol = rsolve_s_vi_r(m, discount, std::move(natparsed), vf_init, rpolicy,
                            iterations, maxresidual, progress);
    } else if (algorithm == "vi_j" || algorithm == "vi") {
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

/// Parses the name and the parameter of the provided nature
algorithms::SNatureOutcome parse_nature_s(const MDPO& mdpo, const string& nature,
                                          SEXP nature_par) {
#ifdef GUROBI_USE
    if (nature == "eavaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_s_avar_exp_u_gurobi(
            Rcpp::as<double>(par["alpha"]), Rcpp::as<double>(par["beta"]));
    } else {
        Rcpp::stop("unknown nature.");
    }
#else
    Rcpp::stop("unknown nature.");
#endif
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
//' @param algorithm One of "ppi", "mppi", "mpi", "vi", "vi_j", "vi_g", "pi". MPI may
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
//' @param show_progress Whether to show a progress bar during the computation.
//'         0 means no progress, 1 is progress bar, and 2 is a detailed report
//'
//' @return A list with value function policy and other values
//'
//' @details
//'
//' The options for nature and the corresponding nature_par are:
//'    \itemize{
//'         \item "exp" plain expectation over the outcomes
//'         \item "eavaru" a convex combination of expectation and AV@R over
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
Rcpp::List rsolve_mdpo_s(Rcpp::DataFrame mdpo, double discount, Rcpp::String nature,
                         SEXP nature_par, Rcpp::String algorithm = "mppi",
                         Rcpp::Nullable<Rcpp::DataFrame> policy_fixed = R_NilValue,
                         double maxresidual = 10e-4, size_t iterations = 10000,
                         double timeout = 300,
                         Rcpp::Nullable<Rcpp::DataFrame> value_init = R_NilValue,
                         bool pack_actions = false, bool output_tran = false,
                         int show_progress = 1) {
#ifdef GUROBI_USE // all srectangular MDPO methods require gurobi so far
    Rcpp::List result;

    // What would be the point of forcing to add transitions even if
    // the  probabilities are 0?
    // perhaps only if the RMDP is transformed to an MDP
    // that is why this is set to true for now .... it is also easy to remove the 0s from
    // the dataframe
    MDPO m = mdpo_from_dataframe(mdpo, true);
    if (m.size() == 0) return Rcpp::List();

    // remove actions that are not being used
    if (pack_actions) { result["action_map"] = m.pack_actions(); }

    // policy: the method can be used to compute the robust solution for a policy
    // rpolicy stands for randomized policy and NOT robust policy
    numvecvec rpolicy =
        policy_fixed.isNotNull()
            ? parse_sa_values(m, policy_fixed.get(), 0.0, "probability", "policy_fixed")
            : numvecvec(0);

    // initialized value function from the parameters
    numvec vf_init =
        value_init.isNotNull()
            ? parse_s_values<prec_t>(m.size(), value_init.get(), 0, "value", "value_init")
            : numvec(0);

    algorithms::SNatureOutcome natparsed = parse_nature_s(m, nature, nature_par);

    ComputeProgress progress(iterations, maxresidual, show_progress, timeout);

    SRobustOutcomeSolution sol;
    if (algorithm == "mppi") {
        sol = rsolve_s_mppi(m, discount, std::move(natparsed), vf_init, rpolicy,
                            iterations, maxresidual, progress);
    } else if (algorithm == "ppi") {
        sol = rsolve_s_ppi(m, discount, std::move(natparsed), vf_init, rpolicy,
                           iterations, maxresidual, progress);
    } else if (algorithm == "mpi") {
        Rcpp::warning("The robust version of the mpi method may cycle forever "
                      "without converging.");
        sol =
            rsolve_s_mpi(m, discount, std::move(natparsed), vf_init, rpolicy, iterations,
                         maxresidual, defaults::mpi_vi_count, 0.5, progress);
    } else if (algorithm == "vi_g") {
        sol = rsolve_s_vi(m, discount, std::move(natparsed), vf_init, rpolicy, iterations,
                          maxresidual, progress);
    } else if (algorithm == "vi_j" || algorithm == "vi") {
        // Jacobian value iteration, simulated using mpi
        sol = rsolve_s_mpi(m, discount, std::move(natparsed), vf_init, rpolicy,
                           iterations, maxresidual, 0, 0.5, progress);
    } else if (algorithm == "pi") {
        Rcpp::warning("The robust version of the pi method may cycle forever without "
                      "converging.");
        sol = rsolve_s_pi(m, discount, std::move(natparsed), vf_init, rpolicy, iterations,
                          maxresidual, progress);
    } else {
        Rcpp::stop("Unknown solver type.");
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;

    auto [dec_pol, nat_pol] = unzip(sol.policy);

    if (output_tran) {
        auto pb = craam::algorithms::SRobustOutcomeBellman(m, natparsed);
        auto tmat = craam::algorithms::transition_mat(pb, sol.policy);
        auto rew = craam::algorithms::rewards_vec(pb, sol.policy);
        result["transitions"] = as_matrix(tmat);
        result["rewards"] = rew;
    }

    result["policy_rand"] = output_policy(dec_pol);
    result["nature"] = output_snature(m, nat_pol);
    result["valuefunction"] = output_value_fun(move(sol.valuefunction));
    result["status"] = sol.status;
    report_solution_status(sol);

    return result;
#else
    Rcpp::stop("Not supported without Gurobi.");
#endif
}

//' Evaluates a randomized policy or computes the optimal policy for many Bayesian samples (MDPO)
//'
//' @param mdpo Dataframe with `idstatefrom`, `idaction`, `idstateto`, `idoutcome`, `probability`, `reward`.
//'             Each `idoutcome` represents a sample. The outcomes must be sorted increasingly.
//' @param discount Discount rate in [0,1) (or = 1 at the risk of divergence)
//' @param policy_rand Randomized policy with columns `idstate`, `idaction`, `probability`.
//' @param initial Initial distribution with columns `idstate` and `probability`. If null
//'                 then the function returns the value functions instead of the returns
//' @param show_progress Whether to show a progress bar
//'
//' @return List of return values / or solutions for all outcomes
// [[Rcpp::export]]
Rcpp::DataFrame
revaluate_mdpo_rnd(Rcpp::DataFrame mdpo, double discount,
                   Rcpp::Nullable<Rcpp::DataFrame> policy_rnd = R_NilValue,
                   Rcpp::Nullable<Rcpp::DataFrame> initial = R_NilValue,
                   bool show_progress = true) {

    if (mdpo.nrow() == 0) return Rcpp::List();

    craam::indvec idstatefrom = mdpo["idstatefrom"], idaction = mdpo["idaction"],
                  idstateto = mdpo["idstateto"], idoutcome = mdpo["idoutcome"];
    craam::numvec probability = mdpo["probability"], reward = mdpo["reward"];

    // parse the data for the first outcome
    if (!std::is_sorted(idoutcome.cbegin(), idoutcome.cend())) {
        Rcpp::stop("The function requires that the outcomes are sorted increasingly.");
    }

    // get the unique outcomes
    craam::indvec outcome_uniq = idoutcome;
    {
        auto unique_end = std::unique(outcome_uniq.begin(), outcome_uniq.end());
        outcome_uniq.erase(unique_end, outcome_uniq.end());
    }

    // parse the first MDP to get the number of states (assumed be the same for each outcome!)
    auto mdp_init = mdp_from_mdpo_dataframe(idstatefrom, idaction, idoutcome, idstateto,
                                            probability, reward, outcome_uniq[0], false);

    // policy: the method can be used to compute the robust solution for a policy
    // rpolicy stands for randomized policy and NOT robust policy
    const craam::numvecvec rpolicy =
        policy_rnd.isNotNull()
            ? parse_sa_values(mdp_init, policy_rnd, 0.0, "probability", "policy_fixed")
            : craam::numvecvec(0);

    // initial distribution
    const craam::Transition initial_tran =
        initial.isNotNull() ? craam::Transition(parse_s_values(mdp_init.size(), initial,
                                                               0.0, "probability"))
                            : craam::Transition();

    //  whether to compute value functions
    std::variant<craam::numvec, craam::numvecvec> mdp_returns;
    if (initial.isNotNull()) {
        mdp_returns = craam::numvec(outcome_uniq.size(),
                                    numeric_limits<craam::prec_t>::quiet_NaN());
    } else {
        mdp_returns = craam::numvecvec(outcome_uniq.size());
    }

    // create a progress bar and use to interrupt the computation
    RcppProg::Progress progress(outcome_uniq.size(), show_progress);

    // check that all states and actions have the same number of outcomes
    // skip the first element, because that one is already parsed
#pragma omp parallel for
    for (size_t iout = 0; iout < outcome_uniq.size(); ++iout) {

        // check if an abort was called; do not stop or bad things happen because of openMP
        if (!progress.check_abort()) {
            // parse the MDP for the given outcome
            auto mdp =
                mdp_from_mdpo_dataframe(idstatefrom, idaction, idoutcome, idstateto,
                                        probability, reward, outcome_uniq[iout], false);
            // solve the MDP
            { // make sure sol cannot be used elsewhe since we modev the value function
                auto sol = solve_mpi_r(mdp, discount, craam::numvec(0), rpolicy);
                if (std::holds_alternative<craam::numvec>(mdp_returns)) {
                    std::get<craam::numvec>(mdp_returns)[iout] =
                        sol.total_return(initial_tran);
                } else {
                    std::get<craam::numvecvec>(mdp_returns)[iout] =
                        move(sol.valuefunction);
                }
            }

            // RcppProg is safe with OpenMP
            progress.increment(1);
        }
    }
    if (progress.check_abort()) { Rcpp::stop("Computation aborted."); }

    if (std::holds_alternative<craam::numvec>(mdp_returns)) {
        return Rcpp::DataFrame::create(Rcpp::_["idoutcome"] = as_intvec(outcome_uniq),
                                       Rcpp::_["return"] =
                                           std::get<craam::numvec>(mdp_returns));
    } else {
        const craam::numvecvec& valuefunctions = std::get<craam::numvecvec>(mdp_returns);
        // returns an empty dataframe: the code below assumes non-empty results
        if (valuefunctions.empty()) { return Rcpp::DataFrame(); }

        // aggregate all values into a single array
        // reserve, just in case that all mdps are not of the same size
        const size_t state_count = valuefunctions.front().size();

        craam::indvec idoutcome;
        idoutcome.reserve(outcome_uniq.size() * state_count);
        craam::indvec idstate;
        idstate.reserve(outcome_uniq.size() * state_count);
        craam::numvec value;
        value.reserve(outcome_uniq.size() * state_count);
        for (std::size_t iout = 0; iout < valuefunctions.size(); ++iout) {
            for (size_t is = 0; is < valuefunctions[iout].size(); ++is) {
                idoutcome.push_back(iout);
                idstate.push_back(is);
                value.push_back(valuefunctions[iout][is]);
            }
        }
        return Rcpp::DataFrame::create(Rcpp::_["idoutcome"] = as_intvec(idoutcome),
                                       Rcpp::_["idstate"] = as_intvec(idstate),
                                       Rcpp::_["value"] = value);
    }
}

//'
//' Sets the number of threads for parallelization.
//' @param n Number of threads
// [[Rcpp::export]]
void rcraam_set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    Rcpp::stop("Compiled without OPENMP support, cannot set the number of threads.");
#endif
}

//' Sets a gurobi parameter. Even numeric values may be provided as strings
//'
//' See https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/refman.pdf
//' for examples
//'
//' For example, to enable logging call:
//' gurobi_set_param("OutputFlag", "1")
//'
//' @examples
//' gurobi_set_param("TimeLimit", "100.0")
//'
//' @param optimizer Which internal gurobi environment should be used. Should be one of
//'                  c('nature', 'lp', 'nonconvex', 'other'). Any other string leads to an error.
//' @param param The name (string) of the parameter to set
//' @param value String value of the parameter (see examples)
// [[Rcpp::export]]
void gurobi_set_param(Rcpp::String optimizer, Rcpp::String param, Rcpp::String value) {
#ifdef GUROBI_USE
    craam::OptimizerType type = [](const Rcpp::String& optimizer_) {
        if (optimizer_ == "nature") {
            return OptimizerType::NatureUpdate;
        } else if (optimizer_ == "lp") {
            return OptimizerType::LinearProgramMDP;
        } else if (optimizer_ == "nonconvex") {
            return OptimizerType::NonconvexOptimization;
        } else if (optimizer_ == "other") {
            return OptimizerType::Other;
        } else {
            Rcpp::stop("Unknown optimizer type");
        }
    }(optimizer);

    get_gurobi(type)->set(param, value);
#else
    Rcpp::stop("Gurobi not supported.");
#endif
}

//' Builds an MDP from samples
//'
//' The samples can be weighted using a column weights. If the column is not provided
//' then value 1.0 is used instead
//'
//' @param samples_frame Dataframe with columns idstatefrom, idaction, idstateto, reward, weight.
//'        The column weight is optional
//'
//'
// [[Rcpp::export]]
Rcpp::DataFrame mdp_from_samples(Rcpp::DataFrame samples_frame) {
    craam::indvec idstatefrom = samples_frame["idstatefrom"],
                  idaction = samples_frame["idaction"],
                  idstateto = samples_frame["idstateto"];
    craam::numvec reward = samples_frame["reward"];

    craam::numvec weight(0); // it is length 0 by default (not used)
    if (samples_frame.containsElementNamed("weight"))
        weight = craam::numvec(samples_frame["weight"]);

    craam::msen::DiscreteSamples samples;

    for (int i = 0; i < samples_frame.nrows(); ++i) {
        samples.add_sample(idstatefrom[i], idaction[i], idstateto[i], reward[i],
                           weight.empty() ? 1.0 : weight[i], i, 0);
    }

    craam::msen::SampledMDP smdp;
    smdp.add_samples(samples);

    return mdp_to_dataframe(*smdp.get_mdp());
}

//' Constructs the linear programming matrix for the MDP
//'
//' The method can construct the LP matrix for the MDP, which is defined as
//' follows:
//' A = [I - gamma P_1; I - gamma P_2; ...] where P_a is the transition
//' probability for action_a. It also constructs the corresponding vector of rewards
//'
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param discount Discount factor in [0,1]
//'
//' @return A list with entries A, b, and idstateaction which is a dataframe with
//'         row_index (1-based), stateid (0-based), actionid (0-based) that
//'         identifies the state and action for each row of the output
// [[Rcpp::export]]
Rcpp::List matrix_mdp_lp(Rcpp::DataFrame mdp, double discount) {

    MDP m = mdp_from_dataframe(mdp);
    if (m.size() == 0) return Rcpp::List();

    auto [A, b, idstateaction] = craam::algorithms::lp_matrix(m, discount);

    Rcpp::List result;
    result["A"] = as_matrix(A);
    result["b"] = as_vector(b);

    auto [idstate, idaction] = unzip(idstateaction);
    indvec row_index(idstate.size(), -1);
    std::iota(row_index.begin(), row_index.end(), 1);

    Rcpp::DataFrame df_idsa =
        Rcpp::DataFrame::create(Rcpp::_["row_index"] = as_intvec(row_index),
                                Rcpp::_["idstate"] = as_intvec(idstate),
                                Rcpp::_["idaction"] = as_intvec(idaction));
    result["idstateaction"] = df_idsa;

    return result;
}

//' Constructs transition probability matrix the MDP
//'
//' The method constructs the transition probability matrix (stochastic matrix)  P_pi
//' and rewards r_pi for the MDP
//'
//'
//' @param mdp A dataframe representation of the MDP. Each row
//'            represents a single transition from one state to another
//'            after taking an action a. The columns are:
//'            idstatefrom, idaction, idstateto, probability, reward
//' @param policy The policy used to construct the transition probabilities and rewards.
//'            It can be a deterministic policy, in which case it should be a dataframe
//'            with columns idstate and idaction. Both indices are 0-based.
//'            It can also be a randomized policy, in which case it should be a dataframe with
//'            columns idstate, idaction, probability.
//' @return A list with P and r, the transition matrix and the reward vector
// [[Rcpp::export]]
Rcpp::List matrix_mdp_transition(Rcpp::DataFrame mdp, Rcpp::DataFrame policy) {

    MDP m = mdp_from_dataframe(mdp);
    if (m.size() == 0) return Rcpp::List();

    Rcpp::List result;

    if (policy.containsElementNamed("probability")) {
        // randomized policy
        numvecvec policy_rand = parse_sa_values(m, policy, 0.0, "probability", "policy");

        auto pb = craam::algorithms::PlainBellmanRand(m);
        auto tmat = craam::algorithms::transition_mat(pb, policy_rand);
        auto rew = craam::algorithms::rewards_vec(pb, policy_rand);
        result["P"] = as_matrix(tmat);
        result["r"] = rew;

    } else {
        // deterministic policy
        indvec policy_det =
            parse_s_values<long>(m.size(), policy, -1, "idaction", "policy");

        auto pb = craam::algorithms::PlainBellman(m);
        auto tmat = craam::algorithms::transition_mat(pb, policy_det);
        auto rew = craam::algorithms::rewards_vec(pb, policy_det);
        result["P"] = as_matrix(tmat);
        result["r"] = rew;
    }
    return result;
}

//' Whether Gurobi LP and MILP is installed
//'
//' This function can be used when determining which functionality
//' is available in the package
// [[Rcpp::export]]
bool rcraam_supports_gurobi() {
#ifdef GUROBI_USE
    return true;
#else
    return false;
#endif
}
