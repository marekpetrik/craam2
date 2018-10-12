#include "utils.hpp"

#include "craam/algorithms/nature_declarations.hpp"
#include "craam/algorithms/nature_response.hpp"
#include "craam/definitions.hpp"
#include "craam/optimization/optimization.hpp"
#include "craam/solvers.hpp"


#include <iostream>
#include <stdexcept>
#include <tuple>

using namespace craam;
using namespace std;

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
 * Parses a data frame  to an MDP
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idstateto, reward, probability.
 *              Multiple state-action-state rows have summed probabilities and averaged rewards.
 *
 * @returns Corresponding MDP definition
 */
craam::MDP mdp_from_dataframe(const Rcpp::DataFrame& data) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    MDP m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idstateto[i],
                              probability[i], reward[i]);
    }

    return m;
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
 * @param def_value The default value for when frame does not specify anything for the state action pair
 *
 * @returns A vector over states with the included values
 */
numvec parse_s_values(const MDP& mdp, const Rcpp::DataFrame& frame,
                      double def_value = 0) {

    numvec result(mdp.size());

    Rcpp::IntegerVector idstates = frame["idstate"];
    Rcpp::NumericVector values = frame["value"];

    for (long i = 0; i < idstates.size(); i++) {
        long idstate = idstates[i];

        if (idstate < 0) Rcpp::stop("idstate must be non-negative");
        if (idstate > mdp.size())
            Rcpp::stop("idstate must be smaller than the number of MDP states");

        double value = values[i];
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
vector<numvec> parse_sa_values(const MDP& mdp, const Rcpp::DataFrame& frame,
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
        for (long j = 0; j < mdp[i].size(); j++) {
            result[i][j] = numvec(mdp[i][j].size(), def_value);
        }
    }

    Rcpp::IntegerVector idstatesfrom = frame["idstatefrom"],
                        idactions = frame["idaction"], idstatesto = frame["idstateto"];
    Rcpp::NumericVector values = frame["value"];

    for (long i = 0; i < idstatesfrom.size(); i++) {
        long idstatefrom = idstatesfrom[i], idstateto = idstatesto[i],
             idaction = idactions[i];

        if (idstatefrom < 0) Rcpp::stop("idstatefrom must be non-negative");
        if (idstatefrom > mdp.size())
            Rcpp::stop("idstatefrom must be smaller than the number of MDP states");
        if (idaction < 0) Rcpp::stop("idaction must be non-negative");
        if (idaction > mdp[idstatefrom].size())
            Rcpp::stop("idaction must be smaller than the number of actions for the "
                       "corresponding state");
        if (idstateto < 0) Rcpp::stop("idstateto must be non-negative");
        if (idstateto > mdp[idstatefrom][idaction].size())
            Rcpp::stop("idstateto must be smaller than the number of positive transition "
                       "probabilites");

        double value = values[i];
        result[idstatefrom][idaction][idstateto] = value;
    }

    return result;
}

/** Packs MDP actions to be consequitive */
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
 */
// [[Rcpp::export]]
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::List options) {
    MDP m = mdp_from_dataframe(mdp);
    Rcpp::List result;

    if (options.containsElementNamed("pack_actions") &&
        Rcpp::as<bool>(options["pack_actions"])) {
        result["action_map"] = m.pack_actions();
    }

    long iterations = options.containsElementNamed("iterations")
                          ? Rcpp::as<long>(options["iterations"])
                          : 1000000;
    double precision = options.containsElementNamed("precision")
                           ? Rcpp::as<long>(options["precision"])
                           : 0.0001;

    DetermSolution sol;

    if (!options.containsElementNamed("algorithm") ||
        Rcpp::as<string>(options["algorithm"]) == "mpi") {
        // Modified policy iteration
        sol = solve_mpi(m, discount, numvec(0), indvec(0), sqrt(iterations), precision,
                        std::min(sqrt(iterations), 1000.0), 0.9);
    } else if (Rcpp::as<string>(options["algorithm"]) == "vi_j") {
        // Jacobian value iteration
        sol = solve_mpi(m, discount, numvec(0), indvec(0), iterations, precision, 1, 0.9);
    } else if (Rcpp::as<string>(options["algorithm"]) == "vi") {
        // Gauss-seidel value iteration
        sol = solve_vi(m, discount, numvec(0), indvec(0), iterations, precision);
    } else if (Rcpp::as<string>(options["algorithm"]) == "pi") {
        // Gauss-seidel value iteration
        sol = solve_pi(m, discount, numvec(0), indvec(0), iterations, precision);
#ifdef GUROBI_USE
    } else if (Rcpp::as<string>(options["algorithm"]) == "lp") {
        // Gauss-seidel value iteration
        sol = solve_lp(m, discount);
#endif // GUROBI_USE
    } else {
        Rcpp::stop("Unknown algorithm type.");
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;
    result["policy"] = move(sol.policy);
    result["valuefunction"] = move(sol.valuefunction);
    return result;
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
        vector<numvec> values =
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
 */
// [[Rcpp::export]]
Rcpp::List rsolve_mdp_sa(Rcpp::DataFrame mdp, double discount, Rcpp::String nature,
                         SEXP nature_par, Rcpp::List options) {

    MDP m = mdp_from_dataframe(mdp);
    Rcpp::List result;

    if (options.containsElementNamed("pack_actions") &&
        Rcpp::as<bool>(options["pack_actions"])) {
        result["action_map"] = m.pack_actions();
    }

    long iterations = options.containsElementNamed("iterations")
                          ? Rcpp::as<long>(options["iterations"])
                          : 1000000;
    double precision = options.containsElementNamed("precision")
                           ? Rcpp::as<long>(options["precision"])
                           : 0.0001;

    SARobustSolution sol;
    algorithms::SANature natparsed = parse_nature_sa(m, nature, nature_par);
    if (!options.containsElementNamed("algorithm") ||
        Rcpp::as<string>(options["algorithm"]) == "mpi") {
        sol = rsolve_mpi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                         sqrt(iterations), precision, sqrt(iterations), 0.5);
    } else if (Rcpp::as<string>(options["algorithm"]) == "vi") {
        sol = rsolve_vi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                        iterations, precision);

    } else if (Rcpp::as<string>(options["algorithm"]) == "pi") {
        sol = rsolve_pi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                        iterations, precision);

    } else {
        Rcpp::stop("Unknown solver type.");
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;

    auto [dec_pol, nat_pol] = unzip(sol.policy);
    result["policy"] = move(dec_pol);
    result["policy.nature"] = move(nat_pol);
    result["valuefunction"] = move(sol.valuefunction);
    return result;
}

/**
 * Parses the name and the parameter of the provided nature
 */
algorithms::SNature parse_nature_s(const MDP& mdp, const string& nature,
                                   SEXP nature_par) {
    /*if(nature == "l1u"){
        return algorithms::nats::robust_l1u(Rcpp::as<double>(nature_par));
    }*/
    if (nature == "l1") {
        numvec values = parse_s_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_s_l1(values);
    }
    /*if(nature == "l1w"){
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),0.0);
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_l1w(budgets, weights);
    }*/
    // ----- gurobi only -----
#ifdef GUROBI_USE
    if (nature == "l1_g") {
        numvec values = parse_s_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_s_l1_gurobi(values);
    }
    /*if(nature == "l1w_g"){
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),0.0);
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_l1w_gurobi(budgets, weights);
    }*/
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
                          : 1000000;
    double precision = options.containsElementNamed("precision")
                           ? Rcpp::as<long>(options["precision"])
                           : 0.0001;

    SRobustSolution sol;
    algorithms::SNature natparsed = parse_nature_s(m, nature, nature_par);
    if (!options.containsElementNamed("algorithm") ||
        Rcpp::as<string>(options["algorithm"]) == "mpi") {
        sol = rsolve_s_mpi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                           sqrt(iterations), precision, sqrt(iterations), 0.5);
    } else if (Rcpp::as<string>(options["algorithm"]) == "vi") {
        sol = rsolve_s_vi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                          iterations, precision);
    } else if (Rcpp::as<string>(options["algorithm"]) == "pi") {
        sol = rsolve_s_pi(m, discount, std::move(natparsed), numvec(0), indvec(0),
                          iterations, precision);
    } else {
        Rcpp::stop("Unknown solver type.");
    }

    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;

    auto [dec_pol, nat_pol] = unzip(sol.policy);
    result["policy"] = move(dec_pol);
    result["policy.nature"] = move(nat_pol);
    result["valuefunction"] = move(sol.valuefunction);
    return result;
}
