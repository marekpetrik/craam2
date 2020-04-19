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

#include <algorithm>
#include <deque>
#include <numeric>
#include <tuple>
// if available, use gurobi
#ifdef GUROBI_USE
#include "gurobi/gurobi_c++.h"
#include <cmath>  // pow in gurobi
#include <memory> // unique_pointer for gurobi
#endif

// The file includes methods for fast solutions of various optimization problems

namespace craam {

/**
@brief Worstcase distribution with a bounded deviation.

Efficiently computes the solution of:
min_p   p^T * z
s.t.    ||p - pbar|| <= xi
        1^T p = 1
        p >= 0

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the provide probability distribution sums
to 1.

@see worstcase_l1_penalty
@param z Reward values
@param pbar Nominal probability distribution
@param t Bound on the L1 norm deviation
@return Optimal solution p and the objective value
*/
std::pair<numvec, prec_t> inline worstcase_l1(numvec const& z, numvec const& pbar,
                                              prec_t xi) {
    assert(*min_element(pbar.cbegin(), pbar.cend()) >= -THRESHOLD);
    assert(*max_element(pbar.cbegin(), pbar.cend()) <= 1 + THRESHOLD);
    assert(xi >= 0.0);
    assert(z.size() > 0 && z.size() == pbar.size());

    // run craam::clamp when std is not available
    xi = clamp(xi, 0.0, 2.0);

    const size_t sz = z.size();
    // sort z values
    const sizvec sorted_ind = sort_indexes<prec_t>(z);
    // initialize output probability distribution; copy the values because most
    // may be unchanged
    numvec o(pbar);
    // pointer to the smallest (worst case) element
    size_t k = sorted_ind[0];
    // determine how much deviation is actually possible given the provided
    // distribution
    prec_t epsilon = std::min(xi / 2, 1 - pbar[k]);
    // add all the possible weight to the smallest element (structure of the
    // optimal solution)
    o[k] += epsilon;
    // start from the last element
    size_t i = sz - 1;
    // find the upper quantile that corresponds to the epsilon
    while (epsilon > 0) {
        k = sorted_ind[i];
        // compute how much of epsilon remains and can be addressed by the current
        // element
        auto diff = std::min(epsilon, o[k]);
        // adjust the output and epsilon accordingly
        o[k] -= diff;
        epsilon -= diff;
        i--;
    }
    prec_t r = inner_product(o.cbegin(), o.cend(), z.cbegin(), prec_t(0.0));
    return {move(o), r};
}

/**
@brief Worstcase deviation given a linear constraint. Used to compute
s-rectangular solutions

Efficiently computes the solution of:
min_{p,t} ||p - q||
s.t.    z^T p <= b
        1^T p = 1
        p >= 0

When the problem is infeasible, then the returned objective value is infinite
and the solution is a vector of length 0.

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the provided probability distribution sums
to 1.

@see worstcase_l1
@param z Reward values
@param p Distribution
@param b Constant in the linear inequality
@return Optimal solution p and the objective value (t).
        Important: returns the objective value and not the dot product
*/
std::pair<numvec, prec_t> inline worstcase_l1_deviation(numvec const& z, numvec const& p,
                                                        prec_t b) {
    assert(*min_element(p.cbegin(), p.cend()) >= -THRESHOLD);
    assert(*max_element(p.cbegin(), p.cend()) <= 1 + THRESHOLD);
    assert(b >= 0.0);
    assert(z.size() > 0 && z.size() == p.size());

    const size_t sz = z.size();
    // sort z values (increasing order)
    const std::vector<size_t> sorted_ind = sort_indexes<prec_t>(z);
    // initialize output probability distribution; copy the values because most
    // may be unchanged
    numvec o(p);
    // initialize the difference t for the output (this is 1/2 of the output
    // value)
    prec_t t = 0;
    // start with t = 0 and increase it progressively until the constraint is
    // satisifed epsilon is the remainder of the constraint that needs to be
    // satisfied
    prec_t epsilon = inner_product(z.cbegin(), z.cend(), p.cbegin(), 0.0) - b;
    // now, simply add violation until the constraint is tight
    // start with the largest element and move towards the beginning
    size_t i = sz - 1;
    // cache the smallest element
    const prec_t smallest_z = z[sorted_ind[0]];
    while (epsilon > 0 && i > 0) {
        size_t k = sorted_ind[i];
        // adjustment size
        prec_t derivative = z[k] - smallest_z;
        // compute how much of epsilon remains and can be addressed by the current
        // element
        prec_t diff = std::min(epsilon / derivative, o[k]);
        // adjust the output and epsilon accordingly
        o[k] -= diff;
        t += diff;
        epsilon -= derivative * diff;
        i--;
    }
    // if there is still some value epsilon, then the solution is not feasible
    if (epsilon > 0) {
        return make_pair(numvec(0), std::numeric_limits<prec_t>::infinity());
    } else {
        // adjust the smallest element
        o[sorted_ind[0]] += t;
        // the l1 norm is twice the difference for the smallest element
        return make_pair(move(o), 2 * t);
    }
}

/**
Identifies knots of the piecewise linear function of the worstcase
l1-constrained response.

Consider the function:
q^-1(b) = min_{p,t} ||p - q||_1
s.t.    z^T p <= b
        1^T p = 1
        p >= 0

The function returns the points of nonlinearity of q^{-1}(b).

The function is convex. It is infty as b -> -infty and constant as b -> infty.

@param z Reward values
@param p Nominal distribution.
@param presorted_ind (optional) Presorted indices for d. If length is 0 then the
indexes are computed.

@return A pair: (knots = b values, function values = xi values)
        knots: Set of values xi for which the function is nonlinear. It is
linear everywhere in between (where defined). The values are generated in a
decreasing order. value: Corresponding values of t for the values above.
*/
std::pair<numvec, numvec> inline worstcase_l1_knots(
    const numvec& z, const numvec& p, const sizvec& presorted_ind = sizvec(0)) {
    assert(z.size() == p.size());

    // sorts indexes if they are not provided
    sizvec sorted_cache(0);
    if (presorted_ind.empty()) { sorted_cache = sort_indexes(z); }
    // sort z values (increasing order)
    const sizvec& sorted_ind = presorted_ind.empty() ? sorted_cache : presorted_ind;

    // cache the smallest element
    const prec_t smallest_z = z[sorted_ind.front()];

    // knots = values of b
    numvec knots;
    knots.reserve(z.size());
    // objective values = values of t
    numvec values;
    values.reserve(z.size());

    prec_t knot = inner_product(z.cbegin(), z.cend(), p.cbegin(), 0.0);
    prec_t value = 0; // start with no value
    knots.push_back(knot);
    values.push_back(value);

    for (long k = long(z.size()) - 1; k > 0; k--) {
        knot -= (z[sorted_ind[size_t(k)]] - smallest_z) * p[sorted_ind[size_t(k)]];
        value += 2 * p[sorted_ind[size_t(k)]];
        knots.push_back(knot);
        values.push_back(value);
    }
    return {move(knots), move(values)};
}

/**
 * Holds and computes the gradients for the homotopy methods. This is used by
 *          worstcase_l1_w and worstcase_l1_w_knots
 *
 * The function computes and sorts the possible basic feasible solutions of:
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{1,w} <= xi
 *
 */
class GradientsL1_w {
protected:
    numvec derivatives; // derivative for each potential basic solution
    indvec donors;      // the index of the donor for each potential basic solution
    indvec receivers;   // the index of the receiver for each potential basic solution
    std::vector<bool> donor_greater; // whether the donor is greater than the nominal
                                     // solution for each potential basic solution
    std::vector<size_t> sorted;      // order of elements after sorted increasingly
                                     // according to the derivatives

public:
    /**
     * Constructs an empty structure
     */
    GradientsL1_w(){};

    /**
     * Computes the possible gradients and sorts them increasingly
     * @param z Objective function
     * @param w Weights in the definition of the L1 norm
     */
    GradientsL1_w(const numvec& z, const numvec& w) {
        constexpr prec_t epsilon = 1e-10;
        size_t element_count = z.size();

        assert(z.size() == element_count);
        assert(w.size() == element_count);

        derivatives.reserve(element_count);
        donors.reserve(element_count + 1);
        receivers.reserve(element_count + 1);
        // whether the donor p is greater than the corresponding pbar
        donor_greater.reserve(element_count + 1);

        // identify possible receivers (must be less weight than all the smaller
        // elements)
        std::vector<std::size_t> possible_receivers;
        { // limit the visibility of these variables
            std::vector<std::size_t> z_increasing = sort_indexes(z);
            prec_t smallest_w = std::numeric_limits<prec_t>::infinity();

            for (size_t iz : z_increasing) {
                assert(w[iz] > epsilon);
                if (w[iz] < smallest_w) {
                    possible_receivers.push_back(iz);
                    smallest_w = w[iz];
                }
            }
        }

        // ** compute derivatives for possible donor-receiver pairs

        // case a: donor is less or equal to pbar
        // donor
        for (size_t i = 0; i < element_count; i++) {
            // receiver
            for (size_t j : possible_receivers) {
                // cannot donate from a smaller value to a larger one; just skip it
                if (z[i] <= z[j]) continue;

                // case a: donor is less or equal to pbar
                derivatives.push_back((-z[i] + z[j]) / (w[i] + w[j]));
                donors.push_back(long(i));
                receivers.push_back(long(j));
                donor_greater.push_back(false);
            }
        }

        // case b: current donor value is greater than pbar and the weight change is
        // non-negative (otherwise a contradiction with optimality) donor (only
        // possible receiver can be a donor here)
        for (size_t i : possible_receivers) {
            // receiver
            for (size_t j : possible_receivers) {
                // cannot donate from a smaller value to a larger one; just skip it
                if (z[i] <= z[j]) continue;

                if (std::abs(w[i] - w[j]) > epsilon && w[i] < w[j]) {
                    // HACK!: adding the epsilon here makes sure that these basic
                    // solutions are preferred in case of ties. This is to prevent
                    // skipping over this kind of basis when it is tied with type a
                    derivatives.push_back(epsilon + (-z[i] + z[j]) / (-w[i] + w[j]));
                    donors.push_back(long(i));
                    receivers.push_back(long(j));
                    donor_greater.push_back(true);
                }
            }
        }

        assert(donors.size() == receivers.size());
        assert(donor_greater.size() == receivers.size());

        sorted = sort_indexes(derivatives);
    }

    /** Returns the number of potential basic solutions generated */
    size_t size() const { return derivatives.size(); }

    /**
     * Returns parameters for the basic solution with gradients that
     * increase with increasing indexes
     * @param index Position, 0 is the smallest derivative, size() -1 is the
     * largest one
     * @return (gradient, donor index, receiver index, does donor probability must
     * be greater than nominal?)
     */
    std::tuple<prec_t, size_t, size_t, bool> steepest_solution(size_t index) const {
        size_t e = sorted[index];
        return {derivatives[e], donors[e], receivers[e], donor_greater[e]};
    }

    /// Genrates a string description of the gradients
    std::string to_string() const {
        std::stringstream sstream;

        for (auto i : sorted) {
            sstream << "(" << (donor_greater[i] ? "+" : "") << donors[i] << "=>"
                    << receivers[i] << "," << derivatives[i] << ")";
        }
        return sstream.str();
    }
};

/**
 * Solve the worst case response problem using a homotopy method.
 *
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{1,w} <= xi
 *
 * @param gradients Pre-computed greadients (n^2 worst-case complexity)
 * @param z Objective
 * @param pbar Nominal distribution
 * @param w Weights in the norm
 * @return the optimal solution and the objective value
 */
std::pair<numvec, prec_t> inline worstcase_l1_w(const GradientsL1_w& gradients,
                                                const numvec& z, const numvec& pbar,
                                                const numvec& w, prec_t xi) {

    assert(pbar.size() == z.size());
    assert(*min_element(pbar.cbegin(), pbar.cend()) >= 0);
    assert(std::abs(accumulate(pbar.cbegin(), pbar.cend(), 0.0) - 1.0) < 1e-6);

    constexpr prec_t epsilon = 1e-10;

    // the working value of the new probability distribution
    numvec p = pbar;
    // remaining value of xi that needs to be allocated
    prec_t xi_rest = xi;

    // keep a queue of the last few gradients in order to
    // minimize numerical issues due to ties and near-ties
    // which can cause nasty inversions in the order in which the bases should be
    // processed
    // this is the working set of gradients
    std::deque<std::tuple<prec_t, size_t, size_t, bool>> grad_que;
    constexpr prec_t grad_epsilon = 1e-5;

    for (size_t k = 0; k < gradients.size(); k++) {

        // push the steepest solution to the queue
        grad_que.push_back(gradients.steepest_solution(k));

        // check if the span of the gradients in the que if greater than and grad_epsilon
        // and keep popping from the head (while there are at least two elements)
        while (grad_que.size() > 1 && std::get<0>(grad_que.front()) <
                                          std::get<0>(grad_que.back()) - grad_epsilon)
            grad_que.pop_front();

        // examine the feasibility all gradients and apply them if needed
        for (size_t l = 0; l < grad_que.size(); l++) {
#ifdef __cpp_structured_bindings
            // edge index
            auto [ignore, donor, receiver, donor_greater] = grad_que[l];
#else
            size_t donor, receiver;
            bool donor_greater;
            tie(std::ignore, donor, receiver, donor_greater) = grad_que[l];
#endif

            // Type C2 basis is not feasible; skip it
            if (donor_greater && p[donor] <= pbar[donor] + epsilon) continue;

            // Type C1 basis is not feasible; skip it
            if (!donor_greater && p[donor] > pbar[donor] + epsilon) continue;

            // make sure that the donor can give
            if (p[donor] < epsilon) continue;

            prec_t weight_change =
                donor_greater ? (-w[donor] + w[receiver]) : (w[donor] + w[receiver]);
            assert(weight_change > 0);

            prec_t donor_step = std::min(
                xi_rest / weight_change,
                (p[donor] > pbar[donor] + epsilon) ? (p[donor] - pbar[donor]) : p[donor]);
            p[donor] -= donor_step;
            p[receiver] += donor_step;
            xi_rest -= donor_step * weight_change;

            // stop if there is nothing left
            if (xi_rest < epsilon) break;
        }
        if (xi_rest < epsilon) break;
    }

    prec_t objective = inner_product(p.cbegin(), p.cend(), z.cbegin(), 0.0);
    return make_pair(move(p), objective);
}

/**
 * See the documentation for the overloaded function
 */
std::pair<numvec, prec_t> inline worstcase_l1_w(const numvec& z, const numvec& pbar,
                                                const numvec& w, prec_t xi) {
    return worstcase_l1_w(GradientsL1_w(z, w), z, pbar, w, xi);
}

/**
Identifies knots of the piecewise linear function of the weighted worstcase
l1-constrained response.

Consider the function:
q^-1(u) = min_{p,t} ||p - pbar||_{1,w}
s.t.    z^T p <= u
        1^T p = 1
        p >= 0

The function returns the points of nonlinearity of q^{-1}(u). It probably works
even when p sums to less than 1.

The function is convex and non-increasing. It is infty as u -> -infty and
constant as u -> infty.

@param gradients Pre-computed greadients (n^2 worst-case complexity)
@param z Reward values
@param pbar Nominal probability distribution.
@param w Weights used in the L1 norm

@return A pair: (knots = b values, function values = xi values)
        knots: Set of values xi for which the function is nonlinear. It is
linear everywhere in between (where defined). The values are generated in a
decreasing order. value: Corresponding values of t for the values above.
*/
std::pair<numvec, numvec> inline worstcase_l1_w_knots(const GradientsL1_w& gradients,
                                                      const numvec& z, const numvec& pbar,
                                                      const numvec& w) {

    constexpr prec_t epsilon = 1e-10;

    // the working value of the new probability distribution
    numvec p = pbar;

    // will hold the values and knots for the solution path
    numvec knots;
    numvec values;

    // initial value
    knots.push_back(inner_product(pbar.cbegin(), pbar.cend(), z.cbegin(), 0.0)); // u
    values.push_back(0.0); // || ||_{1,w}

    // keep a queue of the last few gradients in order to
    // minimize numerical issues due to ties and near-ties
    // which can cause nasty inversions in the order in which the bases should be
    // processed
    // this is the working set of gradients
    std::deque<std::tuple<prec_t, size_t, size_t, bool>> grad_que;
    constexpr prec_t grad_epsilon = 1e-5;

    // trace the value of the norm and update the norm difference as well as the
    // value of the return (u)
    for (size_t k = 0; k < gradients.size(); k++) {

        // push the steepest solution to the queue
        grad_que.push_back(gradients.steepest_solution(k));

        // check if the span of the gradients in the que if greater than and grad_epsilon
        // and keep popping from the head (while there are at least two elements)
        while (grad_que.size() > 1 && std::get<0>(grad_que.front()) <
                                          std::get<0>(grad_que.back()) - grad_epsilon)
            grad_que.pop_front();

        // examine the feasibility all gradients and apply them if needed
        for (size_t l = 0; l < grad_que.size(); l++) {
#ifdef __cpp_structured_bindings
            // edge index
            auto [ignore, donor, receiver, donor_greater] = grad_que[l];
#else
            size_t donor, receiver;
            bool donor_greater;
            tie(std::ignore, donor, receiver, donor_greater) = grad_que[l];
#endif
            // Type C2 basis is not feasible here, just skip it
            if (donor_greater && p[donor] <= pbar[donor] + epsilon) continue;

            // Type C1 basis is not feasible here, just skip it
            if (!donor_greater && p[donor] > pbar[donor] + epsilon) continue;

            // make sure that the donor can give
            if (p[donor] < epsilon) continue;

            prec_t weight_change =
                donor_greater ? (-w[donor] + w[receiver]) : (w[donor] + w[receiver]);
            assert(weight_change > 0);

            prec_t donor_step = donor_greater ? (p[donor] - pbar[donor]) : p[donor];
            p[donor] -= donor_step;
            p[receiver] += donor_step;

            knots.push_back(knots.back() + donor_step * (z[receiver] - z[donor]));
            values.push_back(values.back() + donor_step * weight_change);
        }
    }

    return make_pair(move(knots), move(values));
}

/// See the overloaded method
std::pair<numvec, numvec> inline worstcase_l1_w_knots(const numvec& z, const numvec& pbar,
                                                      const numvec& w) {
    return worstcase_l1_w_knots(GradientsL1_w(z, w), z, pbar, w);
}

#ifdef GUROBI_USE
/**
 * Uses gurobi to solve for the worst case response subject to a weighted L1
 * constraint
 *
 * min_p  p^T z
 * s.t.   1^T p = 1^T pbar
 *        p >= 0
 *        ||p - pbar||_{1,w} <= xi
 *
 * The linear program formulation is as follows:
 *
 * min_{p,l} p^T z
 * s.t.   1^T p = 1^T pbar
 *        p >= 0
 *        p - pbar <= l
 *        pbar - p <= l
 *        w^T l <= xi
 *        l >= 0
 *
 *  @param wi Weights. Optional, all 1 if not provided.
 * @return Objective value and the optimal solution
 */
std::pair<numvec, prec_t> inline worstcase_l1_w_gurobi(const GRBEnv& env, const numvec& z,
                                                       const numvec& pbar,
                                                       const numvec& wi, prec_t xi) {
    const size_t nstates = z.size();
    assert(nstates == pbar.size());
    assert(wi.empty() || nstates == wi.size());

    numvec ws;
    if (wi.empty()) ws = numvec(nstates, 1.0);
    const numvec& w = wi.empty() ? ws : wi;

    prec_t pbar_sum = accumulate(pbar.cbegin(), pbar.cend(), 0.0);

    GRBModel model = GRBModel(env);

    // Probabilities
    auto p = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstates, 0.0).data(), nullptr, nullptr,
        std::vector<char>(nstates, GRB_CONTINUOUS).data(), nullptr, nstates));
    // Element-wise errors
    auto l = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstates, 0.0).data(), nullptr, nullptr,
        std::vector<char>(nstates, GRB_CONTINUOUS).data(), nullptr, nstates));

    // constraint: 1^T p = 1
    GRBLinExpr ones;
    ones.addTerms(numvec(nstates, 1.0).data(), p.get(), nstates);
    model.addConstr(ones, GRB_EQUAL, pbar_sum);

    // constraint: w^T l <= xi
    GRBLinExpr weights;
    weights.addTerms(w.data(), l.get(), nstates);
    model.addConstr(weights, GRB_LESS_EQUAL, xi);

    // constraints: p - pbar <= l (p - l <= pbar) and
    //              pbar - p <= l (l - p <= -pbar)
    for (size_t idstate = 0; idstate < nstates; idstate++) {
        model.addConstr(p[idstate] - l[idstate] <= pbar[idstate]);
        model.addConstr(-l[idstate] - p[idstate] <= -pbar[idstate]);
    }

    // objective p^T z
    GRBLinExpr objective;
    objective.addTerms(z.data(), p.get(), nstates);
    model.setObjective(objective, GRB_MINIMIZE);

    // solve
    model.optimize();

    // retrieve probability values
    numvec p_result(nstates);
    for (size_t i = 0; i < nstates; i++) {
        p_result[i] = p[i].get(GRB_DoubleAttr_X);
    }

    // get optimal objective value
    prec_t objective_value = model.get(GRB_DoubleAttr_ObjVal);

    return make_pair(move(p_result), objective_value);
}

/**
 * @brief worstcase_l_inf_w_gurobi Uses gurobi to solve for the worst case
 *  response subject to a weighted L_inf constraint
 *
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{inf,w} <= xi
 *
 * The linear program formulation is as follows:
 *
 * min_{p} p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        p - pbar <= w .* xi
 *        pbar - p <= w .* xi
 *
 * @param wi Weights. Optional, all 1 if not provided.
 * @return Objective value and the optimal solution
 */
std::pair<numvec, double> worstcase_linf_w_gurobi(const GRBEnv& env, const numvec& z,
                                                  const numvec& pbar, const numvec& wi,
                                                  double xi) {
    const size_t nstates = z.size();
    assert(nstates == pbar.size());
    assert(wi.empty() || nstates == wi.size());

    numvec ws;
    if (wi.empty()) ws = numvec(nstates, 1.0);
    const numvec& w = wi.empty() ? ws : wi;

    GRBModel model = GRBModel(env);

    // Probabilities
    auto p = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstates, 0.0).data(), nullptr, nullptr,
        std::vector<char>(nstates, GRB_CONTINUOUS).data(), nullptr, nstates));

    // constraint: 1^T p = 1
    GRBLinExpr ones;
    ones.addTerms(numvec(nstates, 1.0).data(), p.get(), nstates);
    model.addConstr(ones, GRB_EQUAL, 1.0);

    //TODO: Check if the weighted linf derivation is correct. unweighted linf looks good though.
    // constraints: p - pbar <= w .* xi (p - (w .* xi) <= pbar) and
    //              pbar - p <= w .* xi ((w .* xi) - p <= -pbar)
    for (size_t idstate = 0; idstate < nstates; idstate++) {
        model.addConstr(p[idstate] - (w[idstate] * xi) <= pbar[idstate]);
        model.addConstr((-w[idstate] * xi) - p[idstate] <= -pbar[idstate]);
    }

    // objective p^T z
    GRBLinExpr objective;
    objective.addTerms(z.data(), p.get(), nstates);
    model.setObjective(objective, GRB_MINIMIZE);

    // solve
    model.optimize();

    // retrieve probability values
    numvec p_result(nstates);
    for (size_t i = 0; i < nstates; i++) {
        p_result[i] = p[i].get(GRB_DoubleAttr_X);
    }

    // get optimal objective value
    double objective_value = model.get(GRB_DoubleAttr_ObjVal);

    return make_pair(move(p_result), objective_value);
}

/**
 * @brief Computes the worst case probability distribution subject to a
 * wasserstein constraint
 *
 * min_{p, lambda} p^T z
 * s.t. 1^T p = 1
 *      p >= 0
 *      sum_j lambda_ij = pbar_i
 *      sum_i lambda_ij = p_j
 *      lambda_ij >= 0
 *
 * @param env Linear program envorinment to prevent repetitive initialization
 * @param z Objective value
 * @param pbar Reference probability distribution
 * @param dst Matrix of distances (or costs) when moving the distribution
 * weights
 * @param xi Size of the ambiguity set
 * @return Worst-case distribution and the objective value
 */
std::pair<numvec, prec_t> inline worstcase_wasserstein_gurobi(const GRBEnv& env,
                                                              const numvec& z,
                                                              const numvec& pbar,
                                                              const numvecvec& dst,
                                                              prec_t xi) {
    GRBModel model = GRBModel(env);

    size_t nstates = z.size();
    assert(nstates == z.size());
    assert(pbar.size() == nstates);
    assert(dst.size() == nstates);

    auto p = std::unique_ptr<GRBVar[]>(model.addVars(
        nullptr, nullptr, nullptr, std::vector<char>(nstates, GRB_CONTINUOUS).data(),
        nullptr, int(nstates)));

    std::vector<std::vector<GRBVar>> lambda(nstates, std::vector<GRBVar>(nstates));

    {
        GRBLinExpr wSum;
        for (size_t i = 0; i < nstates; ++i) {
            model.addConstr(p[i], GRB_GREATER_EQUAL, 0.0,
                            "W_" + std::to_string(i) + "_nonNegative");
            p[i].set(GRB_StringAttr_VarName, "W_" + std::to_string(i));
            wSum += p[i];
        }
        model.addConstr(wSum, GRB_EQUAL, 1.0);
    }

    // create lambda variables
    for (size_t i = 0; i < nstates; ++i) {
        for (size_t j = 0; j < nstates; ++j) {
            lambda[i][j] =
                model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                             "lambda_w_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    for (size_t i = 0; i < nstates; ++i) {
        GRBLinExpr psum;
        for (size_t j = 0; j < nstates; ++j) {
            psum += lambda[i][j];
        }
        model.addConstr(psum, GRB_EQUAL, p[i]);
    }

    for (size_t j = 0; j < nstates; ++j) {
        GRBLinExpr pbarsum;
        for (size_t i = 0; i < nstates; ++i) {
            pbarsum += lambda[i][j];
        }
        model.addConstr(pbarsum, GRB_EQUAL, pbar[j]);
    }

    {
        GRBLinExpr distance;
        for (size_t i = 0; i < nstates; ++i) {
            for (size_t j = 0; j < nstates; ++j) {
                distance += lambda[i][j] * dst[i][j];
            }
        }
        model.addConstr(distance, GRB_LESS_EQUAL, xi);
    }

    // objective value
    GRBLinExpr obj_w;
    obj_w.addTerms(z.data(), p.get(), int(nstates));
    model.setObjective(obj_w, GRB_MINIMIZE);

    model.optimize();
    // model_w.write("./debug_wass_w.lp");

    prec_t objective_w = obj_w.getValue();
    numvec w_result(nstates);

    for (size_t i = 0; i < pbar.size(); i++) {
        w_result[i] = p[i].get(GRB_DoubleAttr_X);
        std::cout << "W[" << i << "]: " << w_result[i] << "\t\t";
    }

    return {move(w_result), objective_w};
}

/**
 * @brief worstcase_l1_w_gurobi Uses gurobi to solve for the worst case response
 * subject to a weighted L1 constraint
 *
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{2,w} <= xi
 *
 * The linear program formulation is as follows:
 *
 * min_{p,l} p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        p - pbar <= l
 *        pbar - p <= l
 *        l^T diaq(w^2) l <= xi^2
 *        l >= 0
 *
 *  @param wi Weights. Optional, all 1 if not provided.
 * @return Objective value and the optimal solution
 */
std::pair<numvec, prec_t> inline worstcase_l2_w_gurobi(const GRBEnv& env, const numvec& z,
                                                       const numvec& pbar,
                                                       const numvec& wi, prec_t xi) {
    const size_t nstates = z.size();
    assert(nstates == pbar.size());
    assert(wi.empty() || nstates == wi.size());

    numvec ws;
    if (wi.empty()) ws = numvec(nstates, 1.0);
    const numvec& w = wi.empty() ? ws : wi;

    GRBModel model = GRBModel(env);

    // Probabilities
    auto p = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstates, 0.0).data(), nullptr, nullptr,
        std::vector<char>(nstates, GRB_CONTINUOUS).data(), nullptr, nstates));
    // Element-wise errors
    auto l = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstates, 0.0).data(), nullptr, nullptr,
        std::vector<char>(nstates, GRB_CONTINUOUS).data(), nullptr, nstates));

    // constraint: 1^T p = 1
    GRBLinExpr ones;
    ones.addTerms(numvec(nstates, 1.0).data(), p.get(), nstates);
    model.addConstr(ones, GRB_EQUAL, 1.0);

    // constraint: l^T W l <= xi^2
    numvec wsquared(w.size());
    transform(w.cbegin(), w.cend(), wsquared.begin(),
              [](prec_t iw) { return pow(iw, 2); });
    GRBQuadExpr weights;
    weights.addTerms(wsquared.data(), l.get(), l.get(), nstates);
    model.addQConstr(weights, GRB_LESS_EQUAL, pow(xi, 2));

    // constraints: p - pbar <= l (p - l <= pbar) and
    //              pbar - p <= l (l - p <= -pbar)
    for (size_t idstate = 0; idstate < nstates; idstate++) {
        model.addConstr(p[idstate] - l[idstate] <= pbar[idstate]);
        model.addConstr(-l[idstate] - p[idstate] <= -pbar[idstate]);
    }

    // objective p^T z
    GRBLinExpr objective;
    objective.addTerms(z.data(), p.get(), nstates);
    model.setObjective(objective, GRB_MINIMIZE);

    // solve
    model.optimize();

    // retrieve probability values
    numvec p_result(nstates);
    for (size_t i = 0; i < nstates; i++) {
        p_result[i] = p[i].get(GRB_DoubleAttr_X);
    }

    // get optimal objective value
    prec_t objective_value = model.get(GRB_DoubleAttr_ObjVal);

    return make_pair(move(p_result), objective_value);
}

#endif

/**
 * Computes the average value at risk and the probability distribution that corresponds to it.
 * The assumption is that this is a maximization problem, and it remains a maximization
 * problem when the measure is applied.
 *
 * The formulation that is used is (see https://en.wikipedia.org/wiki/Expected_shortfall):
 * AVaR(z,alpha) =  1/alpha * ( E[X I{X <= x_a} ] + x_a (alpha - P[X <= x_a] )
 * where I is the indicator function and
 * x_a = inf{x \in R : P[X <= x] >= alpha}
 *
 * The function works by solving the robust representation  / dual form
 * of the risk measure.
 * The dual form is:
 * min_{q in Q} q^T z
 * where
 * Q = {q in Delta : q_i / pbar_i <= 1/alpha }
 * and Delta is the probability simplex. The solution is to assign as much probability weight
 * to elements with small z values as possible.
 *
 * The value x_a is the same as the Value at Risk at level alpha
 *
 * Note that the definition is the negative of the definition on wikipedia.
 *
 * This function is concave.
 *
 * @param z  Objective / random variable
 * @param pbar Nominal distribution of the random variable
 * @param alpha Risk level in [0,1]. alpha = 0 is the worst case
 *              (minimum of z, alpha = 0 is treated as alpha -> 0),
 *              while alpha = 1 is the mean.
 *
 * @return A distribution p and The average value at risk as well as the distribution such that
 * p^T z = avar. The p is the optimal solution to the robust representation of the risk measure.
 */
std::pair<numvec, prec_t> inline avar(const numvec& z, const numvec& pbar, prec_t alpha) {

    // this method sorts the values, which is not as efficient as it may be,
    // the problem could be solved using a variant of quick select, but it is
    // probably not worth the effort.
    if (z.size() == 0 || pbar.size() == 0) return {numvec(0), std::nan("")};

    assert(pbar.size() == z.size());
    assert(alpha >= 0.0 && alpha <= 1.0);
    assert(is_probability_dist(pbar.cbegin(), pbar.cend()));

    const sizvec sortedi = sort_indexes(z);

    // this is the new distribution
    numvec distribution(pbar.size(), 0.0);
    prec_t value = 0; // this the return value, updated while computing the distribution

    // for really small alpha, just return the worst case outright
    if (alpha <= EPSILON) {
        const auto min_pos = std::min_element(z.cbegin(), z.cend());
        value = *min_pos;
        distribution[std::distance(z.cbegin(), min_pos)] = 1.0;
        return {distribution, value};
    }

    // here, we can assume that 1/alpha is not too large
    prec_t probs_accum = 0.0; // accumulated sum of probabilities in q

    size_t pos = 0; // make sure we can use the position later
    // iterate from the smallest element
    for (; pos < sortedi.size() && probs_accum < 1.0; ++pos) {
        // the original index of the current element
        const auto element = sortedi[pos];
        // new probability value of the element, but at most what is left to acheive 1.0
        const prec_t increment = std::min(pbar[element] / alpha, 1.0 - probs_accum);
        // update the distribution, assumulation and the value
        distribution[element] = increment;
        value += increment * z[element];
        probs_accum += increment;
    }
    assert(probs_accum <= 1.0 + EPSILON);
    // make sure that the results are consistent
    assert(std::abs(std::inner_product(distribution.cbegin(), distribution.cend(),
                                       z.cbegin(), 0.0) -
                    value) < EPSILON);
    return {move(distribution), value};
}

/**
 * Computes the value at risk and the probability distribution that corresponds to it.
 * The assumption is that this is a maximization problem, and it remains a maximization
 * problem when the measure is applied.
 *
 * The formulation that is used is (see https://en.wikipedia.org/wiki/Value_at_risk):
 * inf{x \in R : P[X <= x] >= alpha}
 *
 * In general, this function is neither convex nor concave.
 *
 * @param z  Objective / random variable
 * @param pbar Nominal distribution of the random variable
 * @param alpha Risk level in [0,1]. alpha = 0 is the worst case
 *              (minimum of z, alpha = 0 is treated as alpha -> 0)
 *              alpha = 0.5 is the median, and alpha
 *
 * @return A distribution p and the value at risk as well as the distribution such that
 * p^T z = var. Note that there may not be an equivalent robust representation for var like there
 * is for coherent/convex risk measures.
 */
std::pair<numvec, prec_t> inline var(const numvec& z, const numvec& pbar, prec_t alpha) {

    // this method sorts the values, which is not as efficient as it may be,
    // the problem could be solved using a variant of quick select, but it is
    // probably not worth the effort.
    if (z.size() == 0 || pbar.size() == 0) return {numvec(0), std::nan("")};

    assert(pbar.size() == z.size());
    assert(alpha >= 0.0 && alpha <= 1.0);
    assert(is_probability_dist(pbar.cbegin(), pbar.cend()));

    const sizvec sortedi = sort_indexes(z);

    // find the index such that the sum of the probabilities is greater than alpha
    // terminate outright if alpha <= 0
    long pos = 0;             // make sure we can use the position later
    prec_t probs_accum = 0.0; // accumulated sum of probabilities
    for (; pos < long(sortedi.size()); ++pos) {
        probs_accum += pbar[sortedi[pos]];
        if (probs_accum >= alpha) break;
    }
    assert(probs_accum >= alpha - 1e-5);

    // the loop may run all the way to the end .. need to step back then
    const auto element = sortedi[std::min(long(sortedi.size()) - 1, pos)];
    numvec distribution(pbar.size(), 0.0);
    distribution[element] = 1.0;
    // make sure that the results are consistent
    const auto value = z[element];
    assert(std::abs(std::inner_product(distribution.cbegin(), distribution.cend(),
                                       z.cbegin(), 0.0) -
                    value) < EPSILON);
    return {move(distribution), value};
}

/**
 * Computes a convex combination of var and expectation:
 *
 * beta * var_pbar [z] + (1-beta) * E_pbar[z]
 *
 * See also var for the details on how the variance is computed. alpha = 0 is the
 * worst case.
 *
 */
std::pair<numvec, prec_t> inline var_exp(const numvec& z, const numvec& pbar,
                                         prec_t alpha, prec_t beta) {

#ifdef __cpp_structured_bindings
    auto [var_dist, var_value] = var(z, pbar, alpha);
#else
    numvec var_dist;
    prec_t var_value;
    std::tie(var_dist, var_value) = var(z, pbar, alpha);
#endif

    assert(var_dist.size() == pbar.size());
    const auto mean_value = std::inner_product(z.cbegin(), z.cend(), pbar.cbegin(), 0.0);

    numvec result_dist(z.size(), 0.0);
    std::transform(
        var_dist.cbegin(), var_dist.cend(), pbar.cbegin(), result_dist.begin(),
        [beta](prec_t var, prec_t exp) { return beta * var + (1 - beta) * exp; });
    const auto result_val = beta * var_value + (1 - beta) * mean_value;
    return {move(result_dist), result_val};
}

/**
 * Computes a convex combination of avar and expectation:
 *
 * beta * avar_pbar [z] + (1-beta) * E_pbar[z]
 *
 * See also avar for the details on how the variance is computed. alpha = 0 is the
 * worst case.
 *
 */
std::pair<numvec, prec_t> inline avar_exp(const numvec& z, const numvec& pbar,
                                          prec_t alpha, prec_t beta) {

#ifdef __cpp_structured_bindings
    auto [avar_dist, avar_value] = avar(z, pbar, alpha);
#else
    numvec avar_dist;
    prec_t avar_value;
    std::tie(avar_dist, avar_value) = avar(z, pbar, alpha);
#endif
    assert(avar_dist.size() == pbar.size());
    const auto mean_value = std::inner_product(z.cbegin(), z.cend(), pbar.cbegin(), 0.0);

    numvec result_dist(z.size(), 0.0);
    std::transform(
        avar_dist.cbegin(), avar_dist.cend(), pbar.cbegin(), result_dist.begin(),
        [beta](prec_t avar, prec_t exp) { return beta * avar + (1 - beta) * exp; });
    const auto result_val = beta * avar_value + (1 - beta) * mean_value;
    return {move(result_dist), result_val};
}

} // namespace craam
