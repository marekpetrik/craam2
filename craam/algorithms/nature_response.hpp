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

#include "craam/Transition.hpp"
#include "craam/definitions.hpp"
#include "craam/optimization/bisection.hpp"
#include "craam/optimization/gurobi.hpp"
#include "craam/optimization/optimization.hpp"
#include "craam/optimization/srect_gurobi.hpp"
#include <functional>

namespace craam { namespace algorithms { namespace nats {

// *******************************************************
// SA Nature definitions
// *******************************************************

/**
 * L1 robust response. Implements the SANature concept.
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1 {
protected:
    /// Budget for each state and action
    vector<numvec> budgets;

public:
    robust_l1(numvecvec budgets) : budgets(move(budgets)) {}

    /**
    * Implements SANature interface
    */
    pair<numvec, prec_t> operator()(long stateid, long actionid,
                                    const numvec& nominalprob,
                                    const numvec& zfunction) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(actionid >= 0 && actionid < long(budgets[stateid].size()));

        return worstcase_l1(zfunction, nominalprob, budgets[stateid][actionid]);
    }
};

/**
 * L1 robust response with weights for states
 * The length of the state vector depends on the MDP type. For the plain
 * MDP it is the number of non-zero probabilities, a for an MDPO it is the
 * number of outcomes.
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1w {
protected:
    vector<numvec> budgets;
    /// The weights are optional, if empty then uniform weights are used.
    /// The elements are over states, actions, and then next state values
    vector<vector<numvec>> weights;

public:
    /**
   * @param budgets One value for each state and action
   * @param weights State weights used in the L1 norm. One set of vectors for
   * each state and action. Use and empty vector to specify uniform weights.
   */
    robust_l1w(numvecvec budgets, vector<vector<numvec>> weights)
        : budgets(move(budgets)), weights(weights) {}

    /**
   * @brief Implements SANature interface
   */
    pair<numvec, prec_t> operator()(long stateid, long actionid,
                                    const numvec& nominalprob,
                                    const numvec& zfunction) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(actionid >= 0 && actionid < long(budgets[stateid].size()));
        assert(zfunction.size() == weights[stateid][actionid].size());

        return worstcase_l1_w(zfunction, nominalprob, weights[stateid][actionid],
                              budgets[stateid][actionid]);
    }
};

/**
 * L1 robust response with a untiform budget/threshold
 *
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1u {
protected:
    prec_t budget;

public:
    robust_l1u(prec_t budget) : budget(budget) {}

    /**
     * Implements SANature interface
     */
    pair<numvec, prec_t> operator()(long, long, const numvec& nominalprob,
                                    const numvec& zfunction) const {
        return worstcase_l1(zfunction, nominalprob, budget);
    }
};

/**
 * Response that just computes the expectation
 */
class robust_exp {

public:
    robust_exp() {}

    /// Implements SANature interface
    pair<numvec, prec_t> operator()(long, long, const numvec& nominalprob,
                                    const numvec& zfunction) const {
        auto mean_value = std::inner_product(zfunction.cbegin(), zfunction.cend(),
                                             nominalprob.cbegin(), 0.0);

        return {nominalprob, mean_value};
    }
};

/**
 * Response that is a convex combination of expectation and value at risk:
 *
 *  beta * var_alpha[X] + (1-beta) * E[x]
 *
 */
class robust_var_exp_u {
protected:
    prec_t alpha; // risk level for value at risk
    prec_t beta;  // weight on the value at risk (1-beta is the weight of the expectation

public:
    robust_var_exp_u(prec_t alpha, prec_t beta) : alpha(alpha), beta(beta) {}

    /**
     * Implements SANature interface
     */
    pair<numvec, prec_t> operator()(long, long, const numvec& nominalprob,
                                    const numvec& zfunction) const {
        return var_exp(zfunction, nominalprob, alpha, beta);
    }
};

/**
 * Response that is a convex combination of expectation and average value at risk:
 *
 *  beta * avar_alpha[X] + (1-beta) * E[x]
 */
class robust_avar_exp_u {
protected:
    prec_t alpha; // risk level for value at risk
    prec_t beta;  // weight on the value at risk (1-beta) is the weight of the expectation

public:
    robust_avar_exp_u(prec_t alpha, prec_t beta) : alpha(alpha), beta(beta) {}

    /**
     * Implements SANature interface
     */
    pair<numvec, prec_t> operator()(long, long, const numvec& nominalprob,
                                    const numvec& zfunction) const {
        return avar_exp(zfunction, nominalprob, alpha, beta);
    }
};

/**
 * L1 robust response
 *
 * @see rsolve_mpi, rsolve_vi
 */
class optimistic_l1 {
protected:
    vector<numvec> budgets;

public:
    optimistic_l1(vector<numvec> budgets) : budgets(move(budgets)) {}

    /**
   * @brief Implements SANature interface
   */
    pair<numvec, prec_t> operator()(long stateid, long actionid,
                                    const numvec& nominalprob,
                                    const numvec& zfunction) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(actionid >= 0 && actionid < long(budgets[stateid].size()));
        assert(nominalprob.size() == zfunction.size());

        numvec minusv(zfunction.size());
        transform(begin(zfunction), end(zfunction), begin(minusv), negate<prec_t>());
        auto result = worstcase_l1(minusv, nominalprob, budgets[stateid][actionid]);
        return make_pair(result.first, -result.second);
    }
};

/**
 * L1 robust response with a untiform budget/threshold
 *
 * @see rsolve_mpi, rsolve_vi
 */
class optimistic_l1u {
protected:
    prec_t budget;

public:
    optimistic_l1u(prec_t budget) : budget(move(budget)){};

    /**
   * @brief Implements SANature interface
   */
    pair<numvec, prec_t> operator()(long, long, const numvec& nominalprob,
                                    const numvec& zfunction) const {
        assert(nominalprob.size() == zfunction.size());

        numvec minusv(zfunction.size());
        transform(begin(zfunction), end(zfunction), begin(minusv), negate<prec_t>());
        auto result = worstcase_l1(minusv, nominalprob, budget);
        return make_pair(result.first, -result.second);
    }
};

/// Absolutely worst outcome
struct robust_unbounded {
    /**
   * @brief Implements SANature interface
   */
    pair<numvec, prec_t> operator()(long, long, const numvec&,
                                    const numvec& zfunction) const {
        // assert(v.size() == p.size());
        numvec dist(zfunction.size(), 0.0);
        size_t index =
            size_t(min_element(begin(zfunction), end(zfunction)) - begin(zfunction));
        dist[index] = 1;
        return make_pair(dist, zfunction[index]);
    }
};

/// Absolutely best outcome
struct optimistic_unbounded {
    /**
   * @brief Implements SANature interface
   */
    pair<numvec, prec_t> operator()(long, long, const numvec&,
                                    const numvec& zfunction) const {
        // assert(v.size() == p.size());
        numvec dist(zfunction.size(), 0.0);
        size_t index =
            size_t(max_element(begin(zfunction), end(zfunction)) - begin(zfunction));
        dist[index] = 1;
        return make_pair(dist, zfunction[index]);
    }
};

// --------------- GUROBI START ----------------------------------------------------
#ifdef GUROBI_USE

/**
 * L1 robust response using gurobi (slower!). Allows for weighted L1 norm.
 *
 * min_p p^T z
 * s.t. sum_i w_i |p_i - hat{p}_i| <= xi
 *
 */
class robust_l1w_gurobi {
protected:
    vector<numvec> budgets;
    /// The weights are optional, if empty then uniform weights are used.
    /// The elements are over states, actions, and then next state values
    vector<vector<numvec>> weights;
    shared_ptr<GRBEnv> env;

public:
    /**
   * Automatically constructs a gurobi environment object. Weights are uniform
   * when not provided
   * @param budgets Budgets, with a single value for each MDP state and action
   */
    robust_l1w_gurobi(vector<numvec> budgets) : budgets(move(budgets)), weights(0) {
        env = get_gurobi();
        // possibly? make sure it is run in a single thread so it can be parallelized
        // but this interferes with other use of the environment
        // env->set(GRB_IntParam_Threads, 1);
    };

    /**
   * Automatically constructs a gurobi environment object. Weights are
   * considered to be uniform.
   * @param budgets Budgets, with a single value for each MDP state and action
   * @param weights State weights used in the L1 norm. One set of vectors for
   * each state and action. Use and empty vector to specify uniform weights.
   */
    robust_l1w_gurobi(vector<numvec> budgets, vector<vector<numvec>> weights)
        : budgets(move(budgets)), weights(move(weights)) {
        env = get_gurobi();
        // make sure it is run in a single thread so it can be parallelized
        // env->set(GRB_IntParam_Threads, 1);
    };

    /**
   * @param budgets Budgets, with a single value for each MDP state and action
   * @param grbenv Gurobi environment that will be used. Should be
   * single-threaded and probably disable printout. This environment is NOT
   *                thread-safe.
   */
    robust_l1w_gurobi(vector<numvec> budgets, vector<vector<numvec>> weights,
                      const shared_ptr<GRBEnv>& grbenv)
        : budgets(move(budgets)), weights(move(weights)), env(grbenv){};

    /**
   * Implements the SANature interface
   */
    pair<numvec, prec_t> operator()(long stateid, long actionid,
                                    const numvec& nominalprob,
                                    const numvec& zfunction) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(actionid >= 0 && actionid < long(budgets[stateid].size()));

        // check for whether weights are being used
        if (weights.empty()) {
            return worstcase_l1_w_gurobi(*env, zfunction, nominalprob, numvec(0),
                                         budgets[stateid][actionid]);
        } else {
            return worstcase_l1_w_gurobi(*env, zfunction, nominalprob,
                                         weights[stateid][actionid],
                                         budgets[stateid][actionid]);
        }
    }
};

#endif
// --------------- GUROBI END ----------------------------

// *******************************************************
// S Nature definitions
// *******************************************************

/**
 * L1 robust response with a untiform budget/threshold
 *
 * @see rsolve_s_mpi, rsolve_s_vi. rsolve_s_pi
 */
class robust_s_l1u {
protected:
    prec_t budget;

public:
    robust_s_l1u(prec_t budget) : budget(budget) {}

    /**
     * Implements SNature interface
     */
    tuple<numvec, vector<numvec>, prec_t>
    operator()(long stateid, const numvec& policy, const vector<numvec>& nominalprobs,
               const vector<numvec>& zvalues) const {

        // TODO: refactor this and the robust_s_l1 to the same method

        assert(stateid >= 0);
        assert(nominalprobs.size() == zvalues.size());

        prec_t outcome;
        numvec actiondist;
        vector<numvec> new_probability;

        // no decision maker's policy provided
        if (policy.empty()) {
            numvec sa_budgets;
            // compute the distribution of actions and the optimal budgets
            tie(outcome, actiondist, sa_budgets) =
                solve_srect_bisection(zvalues, nominalprobs, budget);

            assert(actiondist.size() == zvalues.size());
            assert(sa_budgets.size() == actiondist.size());

            // compute actual worst-case responses for all actions
            // and aggregate them in a sparse transition probability
            new_probability.reserve(actiondist.size());
            for (size_t a = 0; a < nominalprobs.size(); a++) {
                // skip the ones that have not transition probability
                if (actiondist[a] > EPSILON) {
                    new_probability.push_back(
                        worstcase_l1(zvalues[a], nominalprobs[a], sa_budgets[a]).first);
                } else {
                    new_probability.push_back(numvec(nominalprobs[a].size(), 0.0));
                }
            }
        }
        // a policy is provided
        else {
            std::tie(outcome, new_probability) =
                evaluate_srect_bisection_l1(zvalues, nominalprobs, budget, policy);
            actiondist = policy;
        }
        return {move(actiondist), move(new_probability), outcome};
    }
};

/**
 * S-rectangular L1 constraint with a single budget for every state
 * and optional weights for each action for each state.
 *
 * This class does not support using weighted L1 norms
 */
class robust_s_l1 {
protected:
    numvec budgets;

public:
    robust_s_l1(numvec budgets) : budgets(move(budgets)) {}

    /**
     * Implements SNature interface
     */
    tuple<numvec, vector<numvec>, prec_t>
    operator()(long stateid, const numvec& policy, const vector<numvec>& nominalprobs,
               const vector<numvec>& zvalues) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        prec_t outcome;
        numvec actiondist;
        vector<numvec> new_probability;

        // no decision maker's policy provided
        if (policy.empty()) {
            // compute the distribution of actions and the optimal budgets

            numvec sa_budgets;
            tie(outcome, actiondist, sa_budgets) =
                solve_srect_bisection(zvalues, nominalprobs, budgets[stateid]);

            assert(actiondist.size() == zvalues.size());
            assert(sa_budgets.size() == actiondist.size());

            // compute actual worst-case responses for all actions
            // and aggregate them in a sparse transition probability
            new_probability.reserve(actiondist.size());
            for (size_t a = 0; a < nominalprobs.size(); a++) {
                // skip the ones that have not transition probability
                if (actiondist[a] > EPSILON) {
                    new_probability.push_back(
                        worstcase_l1(zvalues[a], nominalprobs[a], sa_budgets[a]).first);
                } else {
                    new_probability.push_back(numvec(nominalprobs[a].size(), 0.0));
                }
            }
        } else {
            std::tie(outcome, new_probability) = evaluate_srect_bisection_l1(
                zvalues, nominalprobs, budgets[stateid], policy);
            actiondist = policy;
        }
        // make sure that the states and nature have the same number of elements
        assert(actiondist.size() == new_probability.size());
        return {move(actiondist), move(new_probability), outcome};
    }
};

/**
 * S-rectangular L1 constraint with a single budget for every state
 * and support for weighted L1 norms.
 */
class robust_s_l1w {

protected:
    /// one budget value per state
    numvec budgets;
    /// one vector of weights per each state and action
    vector<numvecvec> weights;

public:
    /**
     * Initialize with weights and budgets
     * @param budgets Must have one budget for each state
     * @param weights Must have one vector of weights for each state and action. The number of
     * weights must match the number of state transition with positive probabilities.
     */
    robust_s_l1w(numvec budgets, vector<numvecvec> weights)
        : budgets(move(budgets)), weights(move(weights)) {
        if (this->weights.size() != this->budgets.size()) {
            throw invalid_argument(
                "There must be one weight and one budget for each state.");
        }
    }

    /**
     * Implements SNature interface
     */
    tuple<numvec, vector<numvec>, prec_t>
    operator()(long stateid, const numvec& policy, const vector<numvec>& nominalprobs,
               const vector<numvec>& zvalues) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        prec_t outcome;
        numvec actiondist;
        vector<numvec> new_probability;

        //std::cout << stateid << "," << policy << std::endl;

        // no decision maker's policy provided
        if (policy.empty()) {
            numvec sa_budgets;

            // compute the distribution of actions and the optimal budgets

            tie(outcome, actiondist, sa_budgets) = solve_srect_bisection(
                zvalues, nominalprobs, budgets[stateid], numvec(0), weights[stateid]);

            assert(actiondist.size() == zvalues.size());
            assert(sa_budgets.size() == actiondist.size());

            // compute actual worst-case responses for all actions
            // and aggregate them in a sparse transition probability
            new_probability.reserve(actiondist.size());
            for (size_t a = 0; a < nominalprobs.size(); a++) {
                // skip the ones that have not transition probability
                if (actiondist[a] > EPSILON) {
                    new_probability.push_back(
                        worstcase_l1(zvalues[a], nominalprobs[a], sa_budgets[a]).first);
                } else {
                    new_probability.push_back(numvec(nominalprobs[a].size(), 0.0));
                }
            }
        }
        // a policy is provided
        else {
            std::tie(outcome, new_probability) = evaluate_srect_bisection_l1(
                zvalues, nominalprobs, budgets[stateid], policy, weights[stateid]);
            actiondist = policy;
        }
        // make sure that the states and nature have the same number of elements
        assert(actiondist.size() == new_probability.size());
        return {move(actiondist), move(new_probability), outcome};
    }
};

// --------------- GUROBI BEGIN ----------------------------
#ifdef GUROBI_USE

/**
 * S-rectangular L1 constraint with a single budget for every state.
 */
class robust_s_l1_gurobi {
protected:
    // a budget for every state
    numvec budgets;
    shared_ptr<GRBEnv> env;

public:
    /**
   * Automatically constructs a gurobi environment object. Weights are uniform
   * when not provided
   * @param budgets Budgets, with a single value for each MDP state
   */
    robust_s_l1_gurobi(numvec budgets) : budgets(move(budgets)) {
        env = get_gurobi();
        // make sure it is run in a single thread so it can be parallelized
        //env->set(GRB_IntParam_Threads, 1);
    };

    /**
   * @param env Gurobi environment to use
   * @param budgets Budgets, with a single value for each MDP state
   */
    robust_s_l1_gurobi(const shared_ptr<GRBEnv>& env, numvec budgets)
        : budgets(move(budgets)), env(env){};

    /**
   * Implements SNature interface
   */
    tuple<numvec, vector<numvec>, prec_t>
    operator()(long stateid, const numvec& policy, const vector<numvec>& nominalprobs,
               const vector<numvec>& zvalues) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        prec_t outcome;
        numvec actiondist, sa_budgets;

        // compute the distribution of actions and the optimal budgets

        tie(outcome, actiondist, sa_budgets) = srect_l1_solve_gurobi(
            *env, zvalues, nominalprobs, budgets[stateid], numvecvec(0), policy);

        assert(actiondist.size() == zvalues.size());

        vector<numvec> new_probability;
        new_probability.reserve(actiondist.size());
        for (size_t a = 0; a < nominalprobs.size(); a++) {
            // skip the ones that have not transition probability
            if (actiondist[a] > EPSILON) {
                new_probability.push_back(
                    worstcase_l1(zvalues[a], nominalprobs[a], sa_budgets[a]).first);
            } else {
                new_probability.push_back(numvec(nominalprobs[a].size(), 0.0));
            }
        }

        // make sure that the states and nature have the same number of elements
        assert(actiondist.size() == new_probability.size());
        return {move(actiondist), move(new_probability), outcome};
    }
};

/**
 * S-rectangular L1 constraint with a single budget for every state
 * and optional weights for each action for each state.
 */
class robust_s_l1w_gurobi {

protected:
    /// each budget is for each individual state
    numvec budgets;
    /// The weights are optional, if empty then uniform weights are used.
    /// The elements are over states, actions, and then next state values
    vector<numvecvec> weights;
    shared_ptr<GRBEnv> env;

public:
    /**
   * Automatically constructs a gurobi environment object. Weights are
   * considered to be uniform.
   * @param budgets Budgets, with a single value for each MDP state and action
   * @param weights State weights used in the L1 norm. One set of vectors for
   * each state and action. Use and empty vector to specify uniform weights.
   */
    robust_s_l1w_gurobi(numvec budgets, vector<vector<numvec>> weights)
        : budgets(move(budgets)), weights(move(weights)) {

        assert(this->weights.size() == this->budgets.size());

        env = get_gurobi();
        // make sure it is run in a single thread so it can be parallelized
        //env->set(GRB_IntParam_Threads, 1);
    };

    /**
    * @param budgets Budgets, with a single value for each MDP state and action
    * @param grbenv Gurobi environment that will be used. Should be
    * single-threaded and probably disable printout. This environment is NOT
    *                thread-safe.
    */
    robust_s_l1w_gurobi(numvec budgets, vector<vector<numvec>> weights,
                        const shared_ptr<GRBEnv>& grbenv)
        : budgets(move(budgets)), weights(move(weights)), env(grbenv) {
        assert(this->weights.size() == this->budgets.size());
    };

    /**
     * Implements the SNature interface
     */
    tuple<numvec, vector<numvec>, prec_t> operator()(long stateid, const numvec& policy,
                                                     const numvecvec& nominalprobs,
                                                     const numvecvec& zvalues) const {
        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        //if (!policy.empty()) {
        //    throw invalid_argument("Does not support decision maker's policies.");
        // }

        prec_t outcome;
        numvec actiondist, sa_budgets;

        // compute the distribution of actions and the optimal budgets

        tie(outcome, actiondist, sa_budgets) = srect_l1_solve_gurobi(
            *env, zvalues, nominalprobs, budgets[stateid], weights[stateid], policy);

        assert(actiondist.size() == zvalues.size());

        // use the fast method once the budgets have been calculated (avoids
        // having to solve the gurobi dual)
        vector<numvec> new_probability;
        new_probability.reserve(actiondist.size());
        for (size_t a = 0; a < nominalprobs.size(); a++) {
            // skip the ones that have not transition probability
            if (actiondist[a] > EPSILON) {
                new_probability.push_back(
                    worstcase_l1(zvalues[a], nominalprobs[a], sa_budgets[a]).first);
            } else {
                new_probability.push_back(numvec(nominalprobs[a].size(), 0.0));
            }
        }

        return make_tuple(move(actiondist), move(new_probability), outcome);
    };
};

class robust_s_avar_exp_u_gurobi {
protected:
    /// a single risk-level for all states, must be in [0,1)
    prec_t alpha;
    /// a single beta for all states, must be in [0,1]
    prec_t beta;

    shared_ptr<GRBEnv> env;

public:
    /**
     * Constructs the Gurobi enviroment and initializes variables.
     *
     * The objective is beta * AVaR_alpha [Z] + (1 - beta) Exp [Z]
     *
     * @param alpha Risk level of avar (0 = worst-case)
     * @param beta Weight on AVaR and the complement (1-beta) is the weight
     *                  on the expectation term
     */
    robust_s_avar_exp_u_gurobi(prec_t alpha, prec_t beta) : alpha(alpha), beta(beta) {
        env = get_gurobi();
        // make sure it is run in a single thread so it can be parallelized
        //env->set(GRB_IntParam_Threads, 1);
    }

    /**
     * Implements SNatureOutcome interface
     */
    tuple<numvec, numvec, prec_t> operator()(long stateid, const numvec& policy,
                                             const numvec& nominalprobs,
                                             const numvecvec& zvalues) const {
        assert(zvalues.size() > 0);
        assert(zvalues[0].size() == nominalprobs.size());
        //std::cout << zvalues.size() << std::endl << policy.size() << std::endl;
        assert(policy.empty() || zvalues.size() == policy.size());

        auto [objective, opt_policy, opt_nature] =
            srect_avar_exp(*env, zvalues, nominalprobs, alpha, beta, policy);

        return {move(opt_policy), move(opt_nature), objective};
    }
};

class robust_s_linf_gurobi {
protected:
    // a budget for every state
    numvec budgets;
    shared_ptr<GRBEnv> env;

public:
    /**
     * Automatically constructs a gurobi environment object. Weights are uniform
     * when not provided
     * @param budgets Budgets, with a single value for each MDP state
     */
    robust_s_linf_gurobi(numvec budgets) : budgets(move(budgets)) {
        env = get_gurobi();
        // make sure it is run in a single thread so it can be parallelized
        //env->set(GRB_IntParam_Threads, 1);
    };

    /**
       * @param env Gurobi environment to use
       * @param budgets Budgets, with a single value for each MDP state
       */
    robust_s_linf_gurobi(const shared_ptr<GRBEnv>& env, numvec budgets)
        : budgets(move(budgets)), env(env){};

    /**
       * Implements SNature interface
       */
    tuple<numvec, vector<numvec>, prec_t>
    operator()(long stateid, numvec policy, const vector<numvec>& nominalprobs,
               const vector<numvec>& zvalues) const {
        //assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        if (!policy.empty())
            throw std::logic_error(
                "PPI with Linf is not implemented yet. Need to support "
                "policy evaluation.");

        prec_t outcome;
        numvec actiondist, sa_budgets;

        // compute the distribution of actions and the optimal budgets
        tie(outcome, actiondist, sa_budgets) =
            srect_linf_solve_gurobi(*env, zvalues, nominalprobs, budgets[stateid]);

        assert(actiondist.size() == zvalues.size());

        // compute actual worst-case transition probability (deviated by sa_budget amount from the nominal transition)
        // for all actions and aggregate them in a sparse transition probability
        vector<numvec> new_probability;
        new_probability.reserve(actiondist.size());
        for (size_t a = 0; a < nominalprobs.size(); a++) {
            // skip the ones that have not transition probability
            if (actiondist[a] > EPSILON) {
                new_probability.push_back(
                    worstcase_linf_w_gurobi(*env, zvalues[a], nominalprobs[a], numvec(0),
                                            sa_budgets[a])
                        .first);
            } else {
                new_probability.push_back(numvec(nominalprobs[a].size(), 0.0));
            }
        }

        return make_tuple(move(actiondist), move(new_probability), outcome);
    }
};
#endif // GUROBI_USE
// --------------- GUROBI END ----------------------------

}}} // namespace craam::algorithms::nats
