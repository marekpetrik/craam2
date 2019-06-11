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
#include "optimization.hpp"

#include <tuple>

namespace craam {

using namespace std;

/**
 *  Computes a value of a piecewise linear function h(x)
 *
 * The lower bound of the range is closed and a smaller parameter values than
 * the lower limit is not allowed.
 *
 * The upper bound of the range is open and the function is assumed to be
 * constant going to the infinity.
 *
 * @param knots Knots of the function (in parameter x). The array must be
 *             sorted increasingly.
 * @param values Values in the knots (h(k) for knot k)
 * @param x The parameter value
 * @param break_ties_last Whether to choose the last element of a tied set.
 *              (multiple knots equal x)
 *              Otherwise the first element is chosen.
 * @return Value of the piecewise linear function and the index of the knot.
 *       The value is between knots[index-1] and knots[index]. If the parameter is
 *       past the largest knot, then index points beyond the end of the array
 */
inline std::pair<prec_t, size_t> piecewise_linear(const numvec& knots,
                                                  const numvec& values, prec_t x,
                                                  bool break_ties_last = false) {

    const prec_t epsilon = 1e-10;

    assert(knots.size() == values.size());
    assert(is_sorted(knots.cbegin(), knots.cend()));

    size_t index;
    if (break_ties_last) {
        // call upper_bound to get the last index (smallest xi) in case of a tie
        index = size_t(
            distance(knots.cbegin(), std::upper_bound(knots.cbegin(), knots.cend(), x)));
        // move the index back if x is identical to the last value
        if (index > 0 && x == knots[index - 1]) --index;
    } else {
        index = size_t(
            distance(knots.cbegin(), std::lower_bound(knots.cbegin(), knots.cend(), x)));
    }

    // check for function boundaries
    if (index <= 0) {
        if (x < knots.front() - epsilon)
            throw std::invalid_argument("Parameter x is smaller than the valid range.");
        else
            return {values.front(), index};
    }
    // if all elements are smaller than the last element is returned, so just
    // return the last value
    if (index >= knots.size() - 1) {
        // decide the index based on whether x is really larger than the last knot
        if (x > knots.back()) {
            return {values.back(), index + 1};
        } else if (x == knots.back()) {
            return {values.back(), index};
        } // otherwise just continue!
    }

    // the linear segment is between (index - 1) and (index),
    // so we need to average them
    prec_t x0 = knots[index - 1];
    prec_t x1 = knots[index];
    // x = alpha * x0 + (1 - alpha) * x1
    // alpha = (x - x1) / (x0 - x1)
    prec_t alpha = abs(x1 - x0) > EPSILON ? (x1 - x) / (x1 - x0) : 0.5;
    assert(alpha >= 0 && alpha <= 1);

    prec_t value = alpha * values[index - 1] + (1 - alpha) * values[index];
    return {value, size_t(index)};
}

/**
 * Computes the right derivatives of a piecewise linear function h(x).
 * The function is not differentiable, but it has right derivatives.
 *
 * The right derivative is defined as:
 * partial_+ f(a) = lim_{x -> a+} (f(x) - f(a)) / (x-a)
 *
 * @param knots Knots of the function (in parameter x). The array must be
 *              sorted increasingly.
 * @param values Values in the knots (h(k) for knot k)
 * @param last_derivative The last derivetive (assuming that the function is convex).
 *                          Optional.
 *
 * @return The right derivative at each knot, except for the last one.
 */
inline numvec piecewise_derivatives(const numvec& knots, const numvec& values,
                                    prec_t last_derivative = 0.0) {

    assert(knots.size() == values.size());

    // preallocate the derivates
    numvec derivatives;
    derivatives.reserve(knots.size());

    // compute each right derivative
    for (long i = 0; i < long(knots.size()) - 1; ++i) {
        derivatives.push_back((values[i + 1] - values[i]) / (knots[i + 1] - knots[i]));
    }
    derivatives.push_back(last_derivative);
    return derivatives;
}

/**
 * Minimizes a piecewise linear **convex** function defined by a slope and a collection
 * of knots that define piecewise linear segments. The function is assumed to
 * be defined on x >= x_0.
 *
 * The values di must be non-decreasing
 *
 * f(x) = sum_i^{n-1} (d_i ([x - x_i]_+ - [x - x_{i+1}]_+) ) +
 *                  + d_n [x - x_n]_+ + lambda x
 *
 * where x_i are the x-values of knots, d_i are the derivatives, and
 * lambda is an additional linear term. The number of knots is n
 *
 * The optimal solution of this piecewise linear function is the knot i such
 * that -d_{i-1} >= lambda and -d_i < lambda,
 * == d_{i-1} <= -lambda and d_i > -lambda
 *
 * The function always returns a knot value. It returns the extreme knots
 * even when the optimal solution may be outside.
 *
 * @param knots Values x_i that define the piecewise linear function. Assumed to be
 *                  defined only for x >= x_0
 * @param derivatives The right derivatives in all knots
 * @param lambda An additive linear term
 *
 * @return The optimal knot index. Returns the last knot even if the optimal
 *          solution would be infinity
 */
inline long minimize_piecewise(const numvec& knots, const numvec& derivatives,
                               prec_t lambda) {

    assert(knots.size() == derivatives.size());
    // make sure that the derivatives are non-decreasing
    assert(std::is_sorted(cbegin(derivatives), cend(derivatives)));

    auto begin = cbegin(derivatives);
    long pos = std::distance(begin, std::lower_bound(begin, cend(derivatives), -lambda));
    // make sure to return the last knot even if the value infinite
    pos = std::min(pos, long(derivatives.size()) - 1);
    return pos;
}

/**
 * Computes the optimal objective value of the s-rectangular problem
 *
 * Solves the optimization problem:
 *
 * max_d min_{xi,p} sum_a d(a) p_a^T z_a
 * s.t.    1^T pi = 1, pi >= 0
 *         sum_a xi(a) wa(a) <= psi
 *         || p_a - pbar_a ||_{1,ws_a} <= xi_a
 *
 * The algorithm works by reformulating the problem to:
 *
 * min_u {u : sum_a xi(a) wa(a) <= psi, q_a^{-1}(xi_a) <= u}, where
 * q_a^{-1}(u_a) = min_{p,t} || p - pbar ||_{1,ws_a}
 * s.t.    z^T e <= b
 *        1^T e = 1
 *         p >= 0
 *
 * The function q_a^{-1} is represented by a piecewise linear function.
 *
 * @note Note that the returned xi values may sum to less than psi. This happens
 * when an an action is not active and xi for the particular action is already at
 * its maximal value.
 *
 *
 * @param z Rewards (or values) for all actions
 * @param p Nominal distributions for all actions
 * @param psi Bound on the sum of L1 deviations
 * @param wa Optional set of weights on action errors
 * @param ws Optional set of weights on state errors (using these values can
 * significantly slow the computation)
 * @param gradients Optional structure that holds pre-computed gradients to speed
 * up the computation of the weighted L1 response. Only used with weighted L1
 * computation; the unweighted L1 is too fast to make this useful.
 *
 * @return Objective value, policy (d),
 *         nature's deviation from nominal probability distribution (xi)
 */
inline tuple<prec_t, numvec, numvec>
solve_srect_bisection(const numvecvec& z, const numvecvec& pbar, const prec_t psi,
                      const numvec& wa = numvec(0), const numvecvec ws = numvecvec(0),
                      const vector<GradientsL1_w> gradients = vector<GradientsL1_w>(0)) {

    // make sure that the inputs make sense
    if (z.size() != pbar.size())
        throw invalid_argument("pbar and z must have the same size.");
    if (psi < 0.0) throw invalid_argument("psi must be non-negative");
    if (!wa.empty() && wa.size() != z.size())
        throw invalid_argument("wa must be the same size as pbar and z.");
    if (!ws.empty() && ws.size() != z.size())
        throw invalid_argument("ws must be the same size as pbar and z.");
    if (!gradients.empty() && gradients.size() != z.size())
        throw invalid_argument("gradients must be the same length as pbar and z.");

    // define the number of actions
    const size_t nactions = z.size();

    if (nactions == 0) throw invalid_argument("cannot be called with 0 actions");

    for (size_t a = 0; a < nactions; a++) {
        assert(abs(1.0 - accumulate(pbar[a].cbegin(), pbar[a].cend(), 0.0)) < EPSILON);
        assert(*min_element(pbar[a].cbegin(), pbar[a].cend()) >= 0.0);
    }

    // define the knots and the corresponding values for the piecewise linear
    // q_a^{-1}(xi)
    vector<numvec> knots(
        nactions),        // knots are the possible values of q_a^{-1} (values of u)
        values(nactions); // corresponding values of xi_a for the corresponsing
                          // value of q_a

    // minimal and maximal possible values of u
    prec_t min_u = -numeric_limits<prec_t>::infinity(),
           max_u = -numeric_limits<prec_t>::infinity();

    for (size_t a = 0; a < nactions; a++) {
        // compute the piecewise linear approximation
        assert(z[a].size() == pbar[a].size());

        // check whether state weights are being used,
        // this determines which knots function would be called
        if (ws.empty()) {
            tie(knots[a], values[a]) = worstcase_l1_knots(z[a], pbar[a]);
        } else {
            if (gradients.empty())
                tie(knots[a], values[a]) = worstcase_l1_w_knots(z[a], pbar[a], ws[a]);
            else
                tie(knots[a], values[a]) =
                    worstcase_l1_w_knots(gradients[a], z[a], pbar[a], ws[a]);
        }

        // knots are in the reverse order than what we want here
        // This step could be eliminated by changing the function worstcase_l1_knots
        // to generate the values in a reverse order
        reverse(knots[a].begin(), knots[a].end());
        reverse(values[a].begin(), values[a].end());

        // IMPORTANT: The xi values in values[a] always decrese even when there
        // are ties between the the z values for the particular action
        // IMPORTANT: Values[a] could be tied when there is zero
        // probability of a transition: p[a][i] == 0 for some element i

        // make sure that the largest knot has xi = 0
        assert(abs(values[a].back()) <= 1e-6);

        // update the lower and upper limits on u
#ifdef __cpp_structured_bindings
        auto [minval, maxval] = minmax_element(knots[a].cbegin(), knots[a].cend());
#else
        auto minmaxval = minmax_element(knots[a].cbegin(), knots[a].cend());
        auto minval = minmaxval.first;
        auto maxval = minmaxval.second;
#endif
        // cout << "minval " << *minval << "  maxval " << *maxval << endl;
        // the function is infinite for values smaller than the minumum for any
        // action
        min_u = max(*minval, min_u);
        max_u = max(*maxval, max_u);
    }

    // *** run a bisection search on the value of u. Treats u as a continuous
    // variable, but identifies when the function becomes linear and then
    // terminates with the precise solution
    // lower and upper bounds on the value of u.
    //  => u_lower: largest known value for which the problem is infeasible
    //  => u_upper: smallest known value for which the problem is feasible
    //  => u_pivot: the next value of u that should be examined
    prec_t u_lower = min_u, // feasible for this value of u, but that why we set
                            // the pivot there
        u_upper = max_u, u_pivot = (max_u + min_u) / 2;

    // cout << "min_u " << min_u << endl;
    // cout << "max_u " << max_u << endl;

    // indexes of the largest lower bound knots for u_lower and u_upper, and the
    // pivot.
    // these are used to determine when the lower and upper bounds are on a single
    // line
    sizvec indices_upper(nactions), indices_lower(nactions);

    // define vectors with the problem solutions
    numvec pi(nactions, 0);
    numvec xi(nactions);

    // sum of xi, to compute the final optimal value which is a linear combination
    // of both
    prec_t xisum_lower = 0, xisum_upper = 0;

    // compute xisum_upper and indices,
    for (size_t a = 0; a < nactions; a++) {
#ifdef __cpp_structured_bindings
        auto [xia, index] =
            piecewise_linear(knots[a], values[a], u_upper, true); // choose the min xi
#else
        double xia;
        size_t index;
        tie(xia, index) =
            piecewise_linear(knots[a], values[a], u_upper, true); // choose the min xi
#endif
        xisum_upper += xia * (wa.empty() ? 1.0 : wa[a]);
        indices_upper[a] = index;
    }

    // compute xisum_lower
    for (size_t a = 0; a < nactions; a++) {
#ifdef __cpp_structured_bindings
        // this function should compute the **smallest** xi that can achieve
        // the desired u.
        auto [xia, index] =
            piecewise_linear(knots[a], values[a], u_lower, false); // choose the max xi
#else
        double xia;
        size_t index;
        tie(xia, index) =
            piecewise_linear(knots[a], values[a], u_lower, false); // choose the max xi
#endif
        assert((wa.empty() ? 1.0 : wa[a]) > 0.0);
        xisum_lower += xia * (wa.empty() ? 1.0 : wa[a]);
        indices_lower[a] = index;
        xi[a] = xia; // cache the solution in case the value is feasible
    }

    // need to handle the case when u_lower is feasible. Because the rest of the
    // code assumes that u_lower is infeasible.
    // This situation happens when psi is not constraining (very large)
    if (xisum_lower <= psi || u_lower == u_upper) {
        // index of the state which the index is 0
        size_t zero_index =
            size_t(distance(indices_lower.cbegin(),
                            min_element(indices_lower.cbegin(), indices_lower.cend())));
        assert(indices_lower[zero_index] == 0);
        // the policy will be to take the action in which the index is 0
        // because the other actions will be worse; the derivative of
        // this action is infty
        pi[zero_index] = 1;

        // just return the solution value
        return make_tuple(u_lower, move(pi), move(xi));
    }
    // The upper bound solution should be always feasible
    // this could happen when the minimum xi is not returned for the
    // value of u in case of ties
    assert(xisum_upper <= psi);

    // run the iteration until the upper bounds and lower bounds are close
    while (u_upper - u_lower >= 1e-10) {
        assert(u_lower <= u_upper);
        assert(u_pivot >= u_lower && u_pivot <= u_upper);

        // add up the values of xi and indices
        prec_t xisum = 0;
        sizvec indices_pivot(nactions);
        for (size_t a = 0; a < nactions; a++) {
#ifdef __cpp_structured_bindings
            auto [xia, index] =
                piecewise_linear(knots[a], values[a], u_pivot, true); // choose the min xi
#else
            double xia;
            size_t index;
            tie(xia, index) =
                piecewise_linear(knots[a], values[a], u_pivot, true); //choose the min xi
#endif
            xisum += xia * (wa.empty() ? 1.0 : wa[a]);
            indices_pivot[a] = index;
        }

        // set lower an upper bound depending on whether the solution is feasible
        if (xisum <= psi) {
            // solution is feasible
            u_upper = u_pivot;
            xisum_upper = xisum;
            indices_upper = move(indices_pivot);
        } else {
            // solution is infeasible
            u_lower = u_pivot;
            xisum_lower = xisum;
            indices_lower = move(indices_pivot);
        }
        // update the new pivot in the middle
        u_pivot = (u_lower + u_upper) / 2;

        // xisums decrease with u, make sure that this is indeed the case
        assert(xisum_lower >= xisum_upper);
        // make sure that xsisum_upper remains feasible
        assert(xisum_upper <= psi);

        // if this is the same linear segment, then we can terminate and compute the
        // final solutions as a linear combination
        if (indices_lower == indices_upper) { break; }
    }
    // cout << "xisum lower " << xisum_lower << "  xisum_upper " << xisum_upper <<
    // endl; cout << "u_lower " << u_lower << "  u upper " << u_upper << endl;

    // compute the value as a linear combination of the individual values
    // alpha xisum_lower + (1-alpha) xisum_upper = psi
    // alpha = (psi - xisum_upper) / (xisum_lower - xisum_upper)
    prec_t alpha = (psi - xisum_upper) / (xisum_lower - xisum_upper);

    assert(alpha >= 0 && alpha <= 1);

    // the result is then: alpha * u_lower + (1-alpha) * u_upper
    // Note: numerical issues possible with alpha * u_lower + (1-alpha) * u_upper
    prec_t u_result = alpha * (u_lower - u_upper) + u_upper;

    // yes, I have seen this being violated here
    assert(u_result >= u_lower && u_result <= u_upper);

    // ***** NOW compute the primal solution (pi) ***************
    // this is based on computing pi such that the subderivative of
    // d/dxi ( sum_a pi_a f_a(xi_a) - lambda (sum_a xi_a f(a) ) ) = 0 for xi^*
    // and ignore the inactive actions (for which u is higher than the last
    // segment)

    for (size_t a = 0; a < nactions; a++) {
#ifdef __cpp_structured_bindings
        auto [xia, index] =
            piecewise_linear(knots[a], values[a], u_result, true); // choose min xi
#else
        double xia;
        size_t index;
        tie(xia, index) =
            piecewise_linear(knots[a], values[a], u_result, true); // choose min xi
#endif
        xi[a] = xia;

        // cout << " index " << index << "/" << knots[a].size() << endl;
        assert(knots[a].size() == values[a].size());

        // This means that the smallest knot was returned for the primal solution
        // probably because of a numerical issues -> just change it to index 1
        if (index == 0) {

            assert(abs(u_result - knots[a][0]) < EPSILON);
            index = 1;
            //std::cout << "z[a] = " << z[a] << std::endl;
            //std::cout << "pbar[a] = " << pbar[a] << std::endl;
            //std::cout << "knots = " << knots[a] << std::endl;
            //std::cout << "values = " << values[a] << std::endl;

            // TODO: Can this ever happen?
            //throw std::runtime_error(
            //    "This should not happen (can happen when z's are all "
            //    "the same); index = 0 should be handled by the "
            //    "special case with u_lower feasible. u_lower = " +
            //    to_string(u_lower) + ", u_upper = " + to_string(u_upper) +
            //    ", xisum_lower = " + to_string(xisum_lower) +
            //    ", xisum_upper = " + to_string(xisum_upper) +
            //    ", psi = " + to_string(psi) + ", knots = " + to_string(knots[a].size()));
        }

        // the value u lies between index - 1 and index
        // when it is outside of the largest knot, the derivative is 0 and policy
        // will be 0 (this is when f_a(xi_a^*) < u^*)
        //      that case can be ignored because pi is already initialized
        if (index < knots[a].size()) {
            // we are not outside of the largest knot or the smallest knot

            // compute the derivative of f (1/derivative of g)
            // prec_t derivative = (knots[a][index] - knots[a][index-1]) /
            // (values[a][index] - values[a][index-1]); pi[a] = 1/derivative;
            if (knots[a][index] - knots[a][index - 1] > EPSILON) {
                pi[a] = -(values[a][index] - values[a][index - 1]) /
                        (knots[a][index] - knots[a][index - 1]);
            } else {
                // if the derivative is very close to zero, just assign a small value
                // that will turn to 1 if other actions are not being taking
                pi[a] = EPSILON;
            }
        }

        // cout << "pi[a] " << pi[a] << endl;
        assert(pi[a] >= 0);
    }
    // normalize the probabilities
    prec_t pisum = accumulate(pi.cbegin(), pi.cend(), 0.0);
    // u_upper is chosen to be the upper bound, and thus at least one index should
    // be within the range
    assert(pisum >= EPSILON);
    // divide by the sum to normalize
    transform(pi.cbegin(), pi.cend(), pi.begin(),
              [pisum](prec_t t) { return t / pisum; });

    return {u_result, move(pi), move(xi)};
}

/**
 * Computes the optimal response of the nature for s-rectangular ambiguity
 * for a given randomized decision maker's policy.
 *
 * The goal is to solve the following optimization problem:
 * min_xi sum_a  d_a * (min_p p^T z s.t. ||p - pbar|| <= xi_a)
 * such that sum_a xi_a <= psi
 * The solution is by bisection on the dual of the problem:
 * max_lambda sum_a (min_xi_a d_a Q(min_p p^T z s.t. ||p - pbar|| <= xi_a) +
 *              + lambda xi_a - lambda psi)
 * The method bisects on lambda, but considers only the knots of the
 * piecewise linear functions.
 *
 * When using weighted L1 norms, providing the L1 gradients is likely to
 * singnificantly speed up the execution.
 *
 * @param z Rewards (or values) for all actions
 * @param p Nominal distributions for all actions
 * @param psi Bound on the sum of L1 deviations
 * @param d Stochastic policy: a probability distribution over actions

 * @param ws Optional set of weights on state errors (using weights can
 * significantly slow down the computation)
 * @param gradients Optional structure that holds pre-computed gradients to speed
 * up the computation of the weighted L1 response. Only used with weighted L1
 * computation; the unweighted L1 is too fast to make this useful.
 *
 * @return Objective value, policy (d),
 *         nature's deviation from nominal probability distribution (xi)
 */
inline pair<prec_t, numvecvec> evaluate_srect_bisection_l1(
    const vector<numvec>& z, const vector<numvec>& pbar, const prec_t psi,
    const numvec& d, const vector<numvec> ws = vector<numvec>(0),
    const vector<GradientsL1_w> gradients = vector<GradientsL1_w>(0)) {

    // make sure that the inputs make sense
    if (z.size() != pbar.size())
        throw invalid_argument("pbar and z must have the same size.");
    if (psi < 0.0) throw invalid_argument("psi must be non-negative");
    if (!ws.empty() && ws.size() != z.size())
        throw invalid_argument("ws must be the same size as pbar and z.");
    if (!gradients.empty() && gradients.size() != z.size())
        throw invalid_argument("gradients must be the same length as pbar and z.");
    if (d.size() != z.size()) throw invalid_argument("d and z must have the same size.");
    assert(is_probability_dist(d.cbegin(), d.cend()));

    // takes advantage of the piecewise linear representations
    // of (min_xi_a d_a Q(min_p p^T z s.t. ||p - pbar|| <= xi_a)
    numvecvec knots; // collection of knots for each action
    knots.reserve(z.size());
    numvecvec values; // collection of function values for each action
    values.reserve(z.size());
    numvecvec derivatives; // collection of derivatives for each action
    derivatives.reserve(z.size());

    // the count of all actions
    auto actioncount = z.size();

    // count all knots to preallocate space
    size_t count_allknots = 0;

    // TODO: ignore actions that have 0 or close to 0 transition
    // probabilities

    // compute the knots and values
    if (ws.empty()) {
        // no weights
        for (long ai = 0; ai < long(actioncount); ++ai) {
#ifdef __cpp_structured_bindings
            // NOTE: values and knots are intentionally different from the names in the
            // function. We need the function q and not q^{-1} here.
            auto [values_a, knots_a] = worstcase_l1_knots(z[ai], pbar[ai]);
#else
            numvec knots_a, values_a;
            std::tie(values_a, knots_a) = worstcase_l1_knots(z[ai], pbar[ai]);
#endif

            // multiply the derivatives by the probability
            derivatives.push_back(
                multiply(piecewise_derivatives(knots_a, values_a, 0.0), d[ai]));
            count_allknots += knots_a.size();
            knots.push_back(move(knots_a));
            values.push_back(move(values_a));
        }
    } else {
        // using l1 weights
        for (long ai = 0; ai < long(actioncount); ++ai) {
#ifdef __cpp_structured_bindings
            // NOTE: values and knots are intentionally different from the names in the
            // function. We need the function q and not q^{-1} here.
            auto [values_a, knots_a] =
                gradients.empty()
                    ? worstcase_l1_w_knots(z[ai], pbar[ai], ws[ai])
                    : worstcase_l1_w_knots(gradients[ai], z[ai], pbar[ai], ws[ai]);
#else
            numvec knots_a, values_a;
            std::tie(values_a, knots_a) =
                gradients.empty()
                    ? worstcase_l1_w_knots(z[ai], pbar[ai], ws[ai])
                    : worstcase_l1_w_knots(gradients[ai], z[ai], pbar[ai], ws[ai]);
#endif

            derivatives.push_back(
                multiply(piecewise_derivatives(knots_a, values_a, 0.0), d[ai]));
            count_allknots += knots_a.size();
            knots.push_back(move(knots_a));
            values.push_back(move(values_a));
        }
    }

    // construct the set of all knots (to know which lamdas to consider)
    numvec allderivatives;
    allderivatives.reserve(count_allknots);
    for (long ai = 0; ai < long(actioncount); ++ai) {
        allderivatives.insert(allderivatives.end(), derivatives[ai].cbegin(),
                              derivatives[ai].cend());
        // this is more efficient, but too much hassle
        // merge the knots and make sure that they are ordered
        //std::inplace_merge(allderivatives.begin(),
        //                   allderivatives.end() - derivatives[ai].size(),
        //                   allderivatives.end());
    }
    // lambdas will be equal to the negative of the derivatives
    multiply_inplace(allderivatives, -1);
    // allow a negative value so lambda_upper can be optimal when lambda = 0 is optimal
    allderivatives.push_back(-1);
    std::sort(allderivatives.begin(), allderivatives.end());
    assert(is_sorted(allderivatives.cbegin(), allderivatives.cend()));

    // modifies the array to remove duplicate elements (like 0 for example)
    auto last = std::unique(allderivatives.begin(), allderivatives.end());
    allderivatives.erase(last, allderivatives.end());
    // add one value that is strictly greater than all other elements (to assure a 0-solution)
    allderivatives.push_back(allderivatives.back() + 1);

    auto lambda_begin = allderivatives.cbegin();
    // the actual work starts now. Look for the optimal value of lambda to use
    // The optimal solution is a lambda such that the 0 is in the supergradient
    auto lambda_lower = lambda_begin;              // sum xi_a > psi
    auto lambda_upper = allderivatives.cend() - 1; // sum xi_a < psi

    // compute the xi sums for lower and upper bounds
    // need to compute the actual lambda that is between the last two elements
    prec_t xisum_lower = 0;
    for (long ai = 0; ai < long(actioncount); ++ai) {
        auto knot_index = minimize_piecewise(knots[ai], derivatives[ai], *lambda_lower);
        xisum_lower += knots[ai][knot_index];
    }
    // assert(xisum_lower >= psi); <=== this may not be true when psi is really large
    prec_t xisum_upper = 0;

    for (long ai = 0; ai < long(actioncount); ++ai) {
        auto knot_index = minimize_piecewise(knots[ai], derivatives[ai], *lambda_upper);
        xisum_upper += knots[ai][knot_index];
    }
    assert(xisum_lower >= xisum_upper);
    assert(xisum_upper <= psi);

    // search over lambdas in the array of all derivatives
    // the derivatives are increasing
    // lambda is that:
    // 1) too large: the sum of xi_a < psi
    // 2) too small: the sum of xi_a > psi

    // iterate until the lower an upper bound are just 1 step away;
    // then there is no way to split them
    while (std::distance(lambda_lower, lambda_upper) > 1) {

        // pick a middle point
        auto stepsize = std::distance(lambda_lower, lambda_upper) / 2;
        auto lambda_mean_it = lambda_lower;
        std::advance(lambda_mean_it, stepsize);
        // compute the optimal x_a for the middle lambda
        prec_t xi_sum = 0;
        for (long ai = 0; ai < long(actioncount); ++ai) {
            auto knot_index =
                minimize_piecewise(knots[ai], derivatives[ai], *lambda_mean_it);
            xi_sum += knots[ai][knot_index];
        }

        // compute xi_a for the lambda and check whether it should be the lower of the upper bound
        if (xi_sum <= psi) {
            lambda_upper = lambda_mean_it;
            xisum_upper = xi_sum;
        } else {
            lambda_lower = lambda_mean_it;
            xisum_lower = xi_sum;
        }
        assert(xisum_lower >= xisum_upper);
        assert(xisum_upper <= psi);
    }

    assert(xisum_lower >= xisum_upper);
    // the solution is between the lambda_lower and lambda_upper
    // compute alpha * xisum_lower + (1-alpha) * xisum_upper = psi
    // then lambda = alpha * lambda_lower + (1-alpha) * lambda_upper
    // return 1/2 if they are very close or the same
    //auto alpha = (xisum_lower - xisum_upper) > EPSILON
    //                 ? (psi - xisum_upper) / (xisum_lower - xisum_upper)
    //                 : 0.5;
    //assert(alpha >= 0.0 && alpha <= 1.0);
    // alpha = 0.0;
    //auto lambda = alpha * (*lambda_lower) + (1 - alpha) * (*lambda_upper);
    auto lambda = *lambda_upper;
    // *** compute the optimal x_a for the lambda
    // return objective value
    prec_t objective_value = 0;
    // return probabilities
    numvecvec probabilities_sol(actioncount);

    // compute xi values
    // need to allocate any remaining psi values (not applicable when lambda = 0)
    auto psi_remainder = psi - xisum_upper;
    assert(psi_remainder >= 0);
    numvec xi_values;
    xi_values.reserve(actioncount);
    for (long ai = 0; ai < long(actioncount); ++ai) {
        auto knot_index = minimize_piecewise(knots[ai], derivatives[ai], lambda);
        auto xi = knots[ai][knot_index];
        // use psi-reminder if this appears to be
        if (lambda > EPSILON &&
            std::abs(derivatives[ai][knot_index] + lambda) < EPSILON) {
            xi += psi_remainder;
            psi_remainder = 0;
        }
        xi_values.push_back(xi);
    }

    for (long ai = 0; ai < long(actioncount); ++ai) {
        auto xi = xi_values[ai];
        if (ws.empty()) {
#ifdef __cpp_structured_bindings
            auto [prob, value] = worstcase_l1(z[ai], pbar[ai], xi);
#else
            numvec prob;
            prec_t value;
            std::tie(prob, value) = worstcase_l1(z[ai], pbar[ai], xi);
#endif
            //objective_value += d[ai] * value + xi * lambda;  <=== if psi_remainder were not allocated before
            objective_value += d[ai] * value;
            probabilities_sol[ai] = prob;

        } else {
#ifdef __cpp_structured_bindings
            auto [prob, value] =
                gradients.empty()
                    ? worstcase_l1_w(z[ai], pbar[ai], ws[ai], xi)
                    : worstcase_l1_w(gradients[ai], z[ai], pbar[ai], ws[ai], xi);
#else
            numvec prob;
            prec_t value;
            std::tie(prob, value) =
                gradients.empty()
                    ? worstcase_l1_w(z[ai], pbar[ai], ws[ai], xi)
                    : worstcase_l1_w(gradients[ai], z[ai], pbar[ai], ws[ai], xi);
#endif
            //objective_value += d[ai] * value + xi * lambda; <=== if psi_remainder were not allocated before
            objective_value += d[ai] * value;
            probabilities_sol[ai] = prob;
        }
    }
    //objective_value -= lambda * psi; <=== if psi_remainder were not allocated before

    return {objective_value, probabilities_sol};
}

} // namespace craam
