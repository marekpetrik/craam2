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

// The file includes methods to solve MDPO with a soft-robust objective

#pragma once

#include "craam/definitions.hpp"

#ifdef GUROBI_USE

#include "craam/MDPO.hpp"
#include "craam/Solution.hpp"

#include <memory>

#include <gurobi/gurobi_c++.h>

namespace craam::algorithms {

/**
 * Solves a MDPO (uncertain MDP) with a AVaR soft-robust objective, assuming
 * static uncertainty.
 *
 *
 *
 * The problem is formulated as a *non-convex* quadratic program and solved
 * using Gurobi. The objective to solve is:
 * max_pi lambda * CVaR_{P ~ f}^alpha [return(pi,P)] +
 *        (1-lambda) * E__{P ~ f}^alpha [return(pi,P)]
 * where pi is a randomized policy. The formulation allows for uncertain rewards
 * jointly with uncertain transition probabilities.
 *
 * The outcomes in the MDPO are represented by the value omega below (Omega
 * is the set of all possible outcomes.) The states are s, and actions are a.
 *
 * The actual quadratic program formulation is as follows:
 * max_{pi, d} max_{z,y} z + sum_{omega} (
 *              (1-beta) * sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) +
 *                 beta  * y(omega) )
 * subject to:
 *      y(omega) >= z - sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) for each omega
 *      d(s,omega) = gamma * sum_{s',a'} d(s',omega) * pi(s',a') * pi^omega(s',a',s) +
 *                      f(omega) p_0(omega), for each omega and s
 *      sum_a pi(s,a) = 1  for each s
 *      pi(s,a) >= 0 for each s and a
 *      d(s,omega) >= 0 for each s and omega
 *
 *
 * @param mdpo Uncertain MDP. The outcomes are assumed to represent the uncertainty over MDPs.
 *              The number of outcomes must be uniform for all states and actions
 *              (except for terminal states which have no actions).
 * @param alpha Risk level of avar (0 = worst-case)
 * @param beta Weight on AVaR and the complement (1-beta) is the weight
 *              on the expectation term
 * @return Solution that includes the policy and the value function
 */
inline SARobustSolution srsolve_avar_static_mdpo_quad(const GRBEn& env, const MDPO& mdpo,
                                                      prec_t alpha, prec_t beta) {
    // general constants values
    const double inf = std::numeric_limits<prec_t>::infinity();
    const size_t nstates = mdpo.size();

    if (nstates == 0) return SARobustSolution(0, 0);
    // TODO: move this functionality to the MDPO definition?
    // count the number of outcomes
    const size_t noutcomes = [&] {
        // (find a non-terminal state first)
        auto ps = std::find_if_not(mdpo.begin(), mdpo.end(),
                                   [&](const StateO& s) { s.is_terminal(); });
        // not found: -1
        return ps == mdpo.end() ? -1ul : ps->get_action(0).outcome_count();
    }();

    // all states are terminal, just return an empty solution
    if (noutcomes < 0) return SARobustSolution(numvec(0.0, nstates), {}, -1, -1, 0, 0);

    // find a state with outcomes that do not match the expected number
    for (size_t is = 0; is < mdpo.size(); ++is)
        for (size_t ia = 0; ia < mdpo[ia].size(); ++ia)
            if (mdpo[is][ia].size() != noutcomes)
                throw ModelError(
                    "Number of outcomes is not uniform across all states and actions", is,
                    ia);

    // time the computation
    auto start = chrono::steady_clock::now();

    // --- proceed with creating the model -----
    GRBModel model(env);

    // build a vector of vectors with each element referencing the index in the policy
    // cannot use auto [..,..] because the reference does not work with the lambda later
    vector<indvec> array_index_sa;
    size_t nstateactions;
    std::tie(array_index_sa, nstateactions) = [&]() {
        vector<indvec> index_sa(nstates);
        size_t count_sa = 0; // current count of state-action pairs
        for (size_t i = 0; i < nstates; ++i) {
            index_sa[i].resize(mdpo[i].size());
            std::iota(index_sa[i].begin(), index_sa[i].end(), count_sa);
            count_sa += mdpo[i].size();
        }
        return std::make_pair(move(index_sa), count_sa);
    }();
    // used to index pi
    const auto index_sa = [&](size_t s, size_t a) { return array_index_sa[s][a]; };
    // used to index d, w is omega (first loop over s and then omega/w)
    const auto index_sw = [&](size_t s, size_t w) { return s * noutcomes + w; };
    const size_t nstateoutcomes = nstates * noutcomes;

    auto pi = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstateactions, 0).data(), nullptr, nullptr,
        std::vector<char>(nstateactions, GRB_CONTINUOUS).data(), nullptr, nstateactions));

    auto d = std::unique_ptr<GRBVar[]>(
        model.addVars(numvec(nstateoutcomes, 0).data(), nullptr, nullptr,
                      std::vector<char>(nstateactions, GRB_CONTINUOUS).data(), nullptr,
                      nstateoutcomes));

    auto y = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(noutcomes, 0).data(), nullptr, nullptr,
        std::vector<char>(noutcomes, GRB_CONTINUOUS).data(), nullptr, noutcomes));

    auto z = model.addVar(-inf, +inf, 0, GRB_CONTINUOUS, "");

    // constraint:
    // y(omega) >= z - sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) for each omega
    for (size_t iw = 0; iw < noutcomes; ++iw) {}

} // namespace craam::algorithms

} // namespace craam::algorithms

#endif
