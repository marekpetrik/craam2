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

/** Computes the maximum distribution subject to L1 constraints */
// [[Rcpp::export]]
Rcpp::List worstcase_l1(Rcpp::NumericVector z, Rcpp::NumericVector q, double t) {
    // resulting probability
    craam::numvec p;
    // resulting objective value
    double objective;

    craam::numvec vz(z.begin(), z.end()), vq(q.begin(), q.end());
    std::tie(p, objective) = craam::worstcase_l1(vz, vq, t);

    Rcpp::List result;
    result["p"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["obj"] = objective;

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
 * @returns A vector over states with the included values
 */
template <class T>
vector<T> parse_s_values(const MDP& mdp, const Rcpp::DataFrame& frame, T def_value = 0,
                         const string& value_column = "value") {
    vector<T> result(mdp.size());
    Rcpp::IntegerVector idstates = frame["idstate"];
    Rcpp::NumericVector values = frame[value_column];
    for (long i = 0; i < idstates.size(); i++) {
        long idstate = idstates[i];

        if (idstate < 0) Rcpp::stop("idstate must be non-negative");
        if (idstate > mdp.size())
            Rcpp::stop("idstate must be smaller than the number of MDP states");

        T value = values[i];
        result[idstate] = value;
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
*
* @returns A vector over states with an inner vector of actions
*/
numvecvec parse_sa_values(const MDP& mdp, const Rcpp::DataFrame& frame,
                          double def_value = 0) {

    vector<numvec> result(mdp.size());
    for (long i = 0; i < mdp.size(); i++) {
        result[i] = numvec(mdp[i].size(), def_value);
    }

    Rcpp::IntegerVector idstates = frame["idstate"], idactions = frame["idaction"];
    Rcpp::NumericVector values = frame["value"];

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
 * ans taget states.
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
vector<vector<numvec>> parse_sas_values(const MDP& mdp, const Rcpp::DataFrame& frame,
                                        double def_value = 0) {

    vector<vector<numvec>> result(mdp.size());
    for (long i = 0; i < mdp.size(); i++) {
        result[i] = numvecvec(mdp[i].size());
        for (long j = 0; j < mdp[i].size(); j++) {
            // this is the number of non-zero transition probabilities
            result[i][j] = numvec(mdp[i][j].size(), def_value);
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
                          to_string(idstatefrom) + ", idaction = " + to_string(idaction));
        } else {
            result[idstatefrom][idaction][indexto] = values[i];
        }
    }
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
    indvec states(policy.size());
    std::iota(states.begin(), states.end(), 0);
    auto states_v = Rcpp::IntegerVector(states.cbegin(), states.cend());
    auto policy_v = Rcpp::IntegerVector(policy.cbegin(), policy.cend());
    auto result = Rcpp::DataFrame::create(Rcpp::Named("idstate") = states_v,
                                          Rcpp::Named("idaction") = policy_v);
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

/**
 * Note of actions are packed, the policy may be recoverd
 * from the output parameter action_map
 *
 * @param options
 *          algorithm: "mpi", "vi", "vi_j", "pi"
 *          pack_actions: bool
 *          iterations: int
 *          precision: double
 *          policy : evaluate the deterministic policy
 *          policy_rand: evalute the randomized policy
 */
// [[Rcpp::export]]
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::List options) {
    Rcpp::List result;

    MDP m = mdp_from_dataframe(mdp);

    if (options.containsElementNamed("pack_actions") &&
        Rcpp::as<bool>(options["pack_actions"])) {
        result["action_map"] = m.pack_actions();
    }

    long iterations = options.containsElementNamed("iterations")
                          ? Rcpp::as<long>(options["iterations"])
                          : defaults::iterations;
    double precision = options.containsElementNamed("precision")
                           ? Rcpp::as<double>(options["precision"])
                           : defaults::maxresidual;

    // outputs the matrix of transition probabilities and the
    // vector of rewards
    bool output_mat = options.containsElementNamed("output_tran")
                          ? Rcpp::as<bool>(options["output_tran"])
                          : false;

    double timeout = options.containsElementNamed("timeout")
                         ? Rcpp::as<double>(options["timeout"])
                         : defaults::timeout;

    bool show_progress = options.containsElementNamed("progress")
                             ? Rcpp::as<bool>(options["progress"])
                             : defaults::show_progress;

    if (options.containsElementNamed("policy") &&
        options.containsElementNamed("policy_rand")) {
        Rcpp::stop("Cannot provide both a deterministic and a randomized policy.");
    }

    bool is_randomized = options.containsElementNamed("policy_rand");

    // use one of the solutions, stochastic or deterministic
    DetermSolution sol;
    RandSolution rsol;

    // initialized policies from the parameter
    indvec policy =
        options.containsElementNamed("policy")
            ? parse_s_values<long>(m, Rcpp::as<Rcpp::DataFrame>(options["policy"]), -1,
                                   "idaction")
            : indvec(0);
    numvecvec rpolicy =
        options.containsElementNamed("policy_rand")
            ? parse_sa_values(m, Rcpp::as<Rcpp::DataFrame>(options["policy_rand"]), 0.0)
            : numvecvec(0);

    ComputeProgress progress(iterations, precision, show_progress, timeout);

    if (!options.containsElementNamed("algorithm") ||
        Rcpp::as<string>(options["algorithm"]) == "mpi") {
        // Modified policy iteration
        if (is_randomized) {
            rsol =
                solve_mpi_r(m, discount, numvec(0), rpolicy, sqrt(iterations), precision,
                            std::min(sqrt(iterations), 1000.0), 0.9, progress);
        } else {
            sol = solve_mpi(m, discount, numvec(0), policy, sqrt(iterations), precision,
                            std::min(sqrt(iterations), 1000.0), 0.9, progress);
        }

    } else if (Rcpp::as<string>(options["algorithm"]) == "vi_j") {
        // Jacobian value iteration
        if (is_randomized) {
            rsol = solve_mpi_r(m, discount, numvec(0), rpolicy, iterations, precision, 1,
                               0.9, progress);
        } else {
            sol = solve_mpi(m, discount, numvec(0), policy, iterations, precision, 1, 0.9,
                            progress);
        }

    } else if (Rcpp::as<string>(options["algorithm"]) == "vi") {
        // Gauss-seidel value iteration
        if (is_randomized) {
            rsol = solve_vi_r(m, discount, numvec(0), rpolicy, iterations, precision,
                              progress);
        } else {
            sol =
                solve_vi(m, discount, numvec(0), policy, iterations, precision, progress);
        }

    } else if (Rcpp::as<string>(options["algorithm"]) == "pi") {
        // Gauss-seidel value iteration
        if (is_randomized) {
            rsol = solve_pi_r(m, discount, numvec(0), rpolicy, iterations, precision,
                              progress);
        } else {
            sol =
                solve_pi(m, discount, numvec(0), policy, iterations, precision, progress);
        }
    }
#ifdef GUROBI_USE
    else if (Rcpp::as<string>(options["algorithm"]) == "lp") {
        // Gauss-seidel value iteration
        if (is_randomized) {
            Rcpp::stop("LP with randomized policies not supported.");
        } else {
            sol = solve_lp(m, discount, policy);
        }

    }
#endif // GUROBI_USE
    else {
        Rcpp::stop("Unknown algorithm type.");
    }

    if (output_mat) {
        if (is_randomized) {
            auto pb = craam::algorithms::PlainBellmanRand(m);
            auto tmat = craam::algorithms::transition_mat(pb, rsol.policy);
            auto rew = craam::algorithms::rewards_vec(pb, rsol.policy);
            result["mat"] = as_matrix(tmat);
            result["rew"] = rew;
        } else {
            auto pb = craam::algorithms::PlainBellman(m);
            auto tmat = craam::algorithms::transition_mat(pb, sol.policy);
            auto rew = craam::algorithms::rewards_vec(pb, sol.policy);
            result["mat"] = as_matrix(tmat);
            result["rew"] = rew;
        }
    }

    if (is_randomized) {
        result["iters"] = rsol.iterations;
        result["residual"] = rsol.residual;
        result["time"] = rsol.time;
        result["policy"] = output_policy(rsol.policy);
        result["valuefunction"] = move(rsol.valuefunction);
        result["status"] = rsol.status;
        if (rsol.status != 0)
            Rcpp::warning(
                "Ran out of time or iterations. The solution may be suboptimal.");
    } else {
        result["iters"] = sol.iterations;
        result["residual"] = sol.residual;
        result["time"] = sol.time;
        result["policy"] = output_policy(sol.policy);
        result["valuefunction"] = move(sol.valuefunction);
        result["status"] = sol.status;
        if (sol.status != 0)
            Rcpp::warning(
                "Ran out of time or iterations. The solution may be suboptimal.");
    }

    return result;
}

/**
 * Computes the function for the MDP for the given value function and discount factor
 */
// [[Rcpp::export]]
Rcpp::DataFrame compute_qvalues(Rcpp::DataFrame mdp, Rcpp::NumericVector valuefunction,
                                double discount) {

    MDP m = mdp_from_dataframe(mdp);
    if (m.size() != size_t(valuefunction.size())) {
        Rcpp::stop("value function must have the same size as the MDP.");
    }

    vector<numvec> qvalue = craam::algorithms::compute_qfunction(
        m, Rcpp::as<numvec>(valuefunction), discount);

    return output_sa_values(qvalue, "qvalue");
}
/**
 * Parses the name and the parameter of the provided nature
 */
algorithms::SANature parse_nature_sa(const MDP& mdp, const string& nature,
                                     SEXP nature_par) {
    if (nature == "l1u") {
        return algorithms::nats::robust_l1u(Rcpp::as<double>(nature_par));
    }
    if (nature == "l1") {
        numvecvec values =
            parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_l1(values);
    }
    if (nature == "l1w") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets =
            parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0);
        auto weights =
            parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_l1w(budgets, weights);
    }
    if (nature == "evaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_var_exp_u(Rcpp::as<double>(par["alpha"]),
                                                  Rcpp::as<double>(par["beta"]));
    }
    if (nature == "eavaru") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        return algorithms::nats::robust_avar_exp_u(Rcpp::as<double>(par["alpha"]),
                                                   Rcpp::as<double>(par["beta"]));
    }
#ifdef GUROBI_USE
    if (nature == "l1_g") {
        vector<numvec> values =
            parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_l1w_gurobi(values);
    }
    if (nature == "l1w_g") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets =
            parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0);
        auto weights =
            parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_l1w_gurobi(budgets, weights);
    }
#endif // end gurobi
    else {
        Rcpp::stop("unknown nature");
    }
}
/**
 * Solves a robust MDP version of the problem with sa-rectangular ambiguity
 *
 * The algorithms: pi, mpi may cycle infinitely without converging to a solution,
 * when solving a robust MDP.
 * The lagorithm ppi is guaranteed to converge to an optimal solition.
 */
// [[Rcpp::export]]
Rcpp::List rsolve_mdp_sa(Rcpp::DataFrame mdp, double discount, Rcpp::String nature,
                         SEXP nature_par, Rcpp::List options) {
    Rcpp::List result;
    try {

        MDP m = mdp_from_dataframe(mdp);

        if (options.containsElementNamed("pack_actions") &&
            Rcpp::as<bool>(options["pack_actions"])) {
            result["action_map"] = m.pack_actions();
        }

        long iterations = options.containsElementNamed("iterations")
                              ? Rcpp::as<long>(options["iterations"])
                              : defaults::iterations;
        double precision = options.containsElementNamed("precision")
                               ? Rcpp::as<double>(options["precision"])
                               : defaults::maxresidual;

        double timeout = options.containsElementNamed("timeout")
                             ? Rcpp::as<double>(options["timeout"])
                             : defaults::timeout;

        bool show_progress = options.containsElementNamed("progress")
                                 ? Rcpp::as<bool>(options["progress"])
                                 : defaults::show_progress;

        SARobustSolution sol;
        algorithms::SANature natparsed = parse_nature_sa(m, nature, nature_par);

        ComputeProgress progress(iterations, precision, show_progress, timeout);

        // the default method is to use ppa
        if (!options.containsElementNamed("algorithm") ||
            Rcpp::as<string>(options["algorithm"]) == "ppi") {
            sol = rsolve_ppi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                             iterations, precision, progress);
        } else if (Rcpp::as<string>(options["algorithm"]) == "mpi") {
            Rcpp::warning("The robust version of the mpi method may cycle forever "
                          "without converging.");
            sol =
                rsolve_mpi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                           sqrt(iterations), precision, sqrt(iterations), 0.5, progress);
        } else if (Rcpp::as<string>(options["algorithm"]) == "vi") {
            sol = rsolve_vi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                            iterations, precision, progress);

        } else if (Rcpp::as<string>(options["algorithm"]) == "vi_j") {
            // Jacobian value iteration, simulated using mpi
            sol = rsolve_mpi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                             iterations, precision, 1, 0.5, progress);
        } else if (Rcpp::as<string>(options["algorithm"]) == "pi") {
            Rcpp::warning("The robust version of the pi method may cycle forever without "
                          "converging.");
            sol = rsolve_pi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                            iterations, precision, progress);
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

        result["policy"] = output_policy(dec_pol);
        result["policy.nature"] = move(nat_pol);
        result["valuefunction"] = move(sol.valuefunction);
        result["status"] = sol.status;
        if (sol.status != 0)
            Rcpp::warning(
                "Ran out of time or iterations. The solution may be suboptimal.");

    } catch (std::exception& ex) { forward_exception_to_r(ex); } catch (...) {
        ::Rf_error("c++ exception (unknown reason)");
    }
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
        numvec values =
            parse_s_values<prec_t>(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_s_l1(values);
    }
    if (nature == "l1w") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets =
            parse_s_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0);
        auto weights =
            parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_s_l1w(budgets, weights);
    }
    // ----- gurobi only -----
#ifdef GUROBI_USE
    if (nature == "l1_g") {
        numvec values = parse_s_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_s_l1_gurobi(values);
    }
    if (nature == "l1w_g") {
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets =
            parse_s_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]), 0.0);
        auto weights =
            parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_s_l1w_gurobi(budgets, weights);
    }
#endif
    // ---- end gurobi -----
    else {
        Rcpp::stop("unknown nature");
    }
}
/**
 * Solves a robust MDP version of the problem with s-rectangular ambiguity
 */
// [[Rcpp::export]]
Rcpp::List rsolve_mdp_s(Rcpp::DataFrame mdp, double discount, Rcpp::String nature,
                        SEXP nature_par, Rcpp::List options) {
    MDP m = mdp_from_dataframe(mdp);
    Rcpp::List result;
    if (options.containsElementNamed("pack_actions") &&
        Rcpp::as<bool>(options["pack_actions"])) {
        result["action_map"] = m.pack_actions();
    }
    long iterations = options.containsElementNamed("iterations")
                          ? Rcpp::as<long>(options["iterations"])
                          : defaults::iterations;
    double precision = options.containsElementNamed("precision")
                           ? Rcpp::as<double>(options["precision"])
                           : defaults::maxresidual;

    double timeout = options.containsElementNamed("timeout")
                         ? Rcpp::as<double>(options["timeout"])
                         : defaults::timeout;

    bool show_progress = options.containsElementNamed("progress")
                             ? Rcpp::as<bool>(options["progress"])
                             : defaults::show_progress;
    SRobustSolution sol;
    algorithms::SNature natparsed = parse_nature_s(m, nature, nature_par);

    ComputeProgress progress(iterations, precision, show_progress, timeout);

    if (!options.containsElementNamed("algorithm") ||
        Rcpp::as<string>(options["algorithm"]) == "ppi") {
        sol = rsolve_s_ppi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                           iterations, precision, progress);
    } else if (Rcpp::as<string>(options["algorithm"]) == "mpi") {
        sol = rsolve_s_mpi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                           sqrt(iterations), precision, sqrt(iterations), 0.5, progress);
    } else if (Rcpp::as<string>(options["algorithm"]) == "vi") {
        sol = rsolve_s_vi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                          iterations, precision, progress);
    } else if (Rcpp::as<string>(options["algorithm"]) == "vi_j") {
        // Jacobian value iteration, simulated using mpi
        sol = rsolve_s_mpi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                           iterations, precision, 1, 0.5, progress);
    } else if (Rcpp::as<string>(options["algorithm"]) == "pi") {
        sol = rsolve_s_pi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                          iterations, precision, progress);
    } else {
        Rcpp::stop("Unknown algorithm type: " + Rcpp::as<string>(options["algorithm"]));
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
    result["policy"] = output_policy(dec_pol);
    result["policy.nature"] = move(nat_pol);
    result["valuefunction"] = move(sol.valuefunction);
    result["status"] = sol.status;
    if (sol.status != 0)
        Rcpp::warning("Ran out of time or iterations. The solution may be suboptimal.");

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

/**
Builds MDP from samples
*/
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
