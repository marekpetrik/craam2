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

#include "craam/definitions.hpp"

#ifdef GUROBI_USE
#include "craam/MDP.hpp"
#include "craam/Solution.hpp"

#include <chrono>
#include <gurobi/gurobi_c++.h>
#include <memory>

namespace craam { namespace algorithms {

/**
 * Solves the MDP using the primal formulation (using value functions)
 *
 * Solves the linear program
 * min_v  1^T v
 * s.t.   P_a v >= r_a  for all a
 *
 * Note:
 * The construction could be slow for large MDPs and could be sped up by
 * by using GRBLinExpr::addTerms
 *
 * @param mdp Markov decision process
 * @param discount Discount factor
 * @return Solution that includes the policy, value function
 */
inline DetermSolution solve_lp_primal(const GRBEnv& env, const MDP& mdp,
                                      prec_t discount) {

    // general constants values
    const double inf = std::numeric_limits<prec_t>::infinity();
    const auto nstates = mdp.size();

    // just quit when there are no states
    if (nstates == 0) { return DetermSolution(0, 0); }
    // time the computation
    auto start = chrono::steady_clock::now();

    // construct the LP model
    GRBModel model(env);

    // values v
    auto v = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstates, -inf).data(), nullptr, nullptr,
        std::vector<char>(nstates, GRB_CONTINUOUS).data(), nullptr, int(nstates)));

    // create the objective value
    GRBLinExpr objective;

    // vector of constraints to determine the dual solution and thus the policy
    // one constraint per each state and action
    vector<vector<GRBConstr>> constraints(nstates);

    // add objectives and constraints for each state
    for (size_t si = 0; si < nstates; ++si) {
        // objective
        objective += v[si];
        // constraints for each action
        const State& s = mdp[si];
        constraints[si].resize(s.size());
        for (size_t ai = 0; ai < s.size(); ++ai) {
            // TODO: could be faster using GRBLinExpr::addTerms
            GRBLinExpr constraint;

            // v[s] - discount * sum_{s'} p[s'] v[s'] >= r[s,a]
            constraint += v[si];

            // now go over all the sparse transitions
            const Action& a = s[ai];
            for (size_t ti = 0; ti < a.size(); ++ti) {
                assert(size_t(a.get_indices()[ti]) < nstates);
                constraint -=
                    discount * a.get_probabilities()[ti] * v[a.get_indices()[ti]];
            }
            // adds the constraint
            constraints[si][ai] = model.addConstr(constraint >= a.mean_reward());
        }
    }
    model.setObjective(objective, GRB_MINIMIZE);

    // run optimization
    model.optimize();

    int status = model.get(GRB_IntAttr_Status);
    if ((status == GRB_INF_OR_UNBD) || (status == GRB_INFEASIBLE) ||
        (status == GRB_UNBOUNDED)) {
        return DetermSolution{0, 2};
        //throw runtime_error("Failed to solve the LP.");
    }

    // retrieve policy and action values
    numvec valuefunction(nstates);
    indvec policy(nstates);
    for (size_t si = 0; si < nstates; si++) {
        valuefunction[si] = v[si].get(GRB_DoubleAttr_X);
        //policy[i] = d[i].get(GRB_DoubleAttr_X);
        // allocate the duals
        const State& s = mdp[si];
        numvec tmp_duals(s.size());
        assert(s.size() == constraints[si].size());
        // retrieve the dual values for each constraint
        for (size_t i = 0; i < s.size(); ++i) {
            tmp_duals[i] = constraints[si][i].get(GRB_DoubleAttr_Pi);
        }
        // the action is just the arg max of the dual
        policy[si] =
            max_element(tmp_duals.cbegin(), tmp_duals.cend()) - tmp_duals.cbegin();
    }

    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    return DetermSolution(move(valuefunction), move(policy), 0.0, -1, duration.count());
}

}} // namespace craam::algorithms

#endif // GUROBI_USE
