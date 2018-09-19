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

/**
Robust MDP methods for computing value functions.
*/
#pragma once

#include "craam/MDPO.hpp"
#include "craam/algorithms/nature_declarations.hpp"

namespace craam { namespace algorithms {

using namespace std;

// *******************************************************
// ActionO computation methods
// *******************************************************

/**
        Computes the maximal outcome distribution constraints on the nature's
        distribution. Does not work when the number of outcomes is zero.

        @param action Action for which the value is computed
        @param valuefunction Value function reference
        @param discount Discount factor
        @param nature Method used to compute the response of nature.

        \return Outcome distribution and the mean value for the choice of the nature
        */
inline vec_scal_t value_action(const ActionO& action, const numvec& valuefunction,
                               prec_t discount, long stateid, long actionid,
                               const SANature& nature) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if (action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");

    numvec outcomevalues(action.size());
    for (size_t i = 0; i < action.size(); i++)
        outcomevalues[i] = action[i].value(valuefunction, discount);

    return nature(stateid, actionid, action.get_distribution(), outcomevalues);
}

// *******************************************************
// Robust computation methods
// *******************************************************

/**
Computes the average outcome using the provided distribution.

@param action Action for which the value is computed
@param valuefunction Updated value function
@param discount Discount factor
@return Mean value of the action
 */
inline prec_t value_action(const ActionO& action, numvec const& valuefunction,
                           prec_t discount) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if (action.get_outcomes().empty()) throw invalid_argument("ActionO with no outcomes");

    prec_t averagevalue = 0.0;
    const numvec& distribution = action.get_distribution();
    for (size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += distribution[i] * action[i].value(valuefunction, discount);
    return averagevalue;
}

/**
Computes the action value for a fixed index outcome.

@param action Action for which the value is computed
@param valuefunction Updated value function
@param discount Discount factor
@param distribution Custom distribution that is selected by nature.
@return Value of the action
 */
inline prec_t value_action(const ActionO& action, numvec const& valuefunction,
                           prec_t discount, const numvec& distribution) {

    assert(distribution.size() == action.get_outcomes().size());
    if (action.get_outcomes().empty()) throw invalid_argument("ActionO with no outcomes");

    prec_t averagevalue = 0.0;
    // TODO: simd?
    for (size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += distribution[i] * action[i].value(valuefunction, discount);
    return averagevalue;
}

}} // namespace craam::algorithms
