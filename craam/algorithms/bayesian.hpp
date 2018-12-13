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

#include "craam/MDP.hpp"
#include "craam/MDPO.hpp"

namespace craam { namespace bayesian {

enum class Norm { L1, L2, Linf };

/**
 * Computes the size of confidence intervals for sa-rectangular ambiguity sets.
 *
 * Remember that this is a confidence level for each of the state and actions
 * independently. To get the confidence for the entire solution, the simplest
 * method is to rely on the union bound. That is to get a 1-delta confidence in
 * the solution, then use:
 *
 * level = 1 - 1/(1-delta)/states/actions
 *
 * @param mdpo MDP with outcomes
 * @param level Confidence level between 0 and 1
 * @param norm The type of the norm to use for the confidence interval
 *
 * @return An MDP with the nominal points and the appropriate size of the confidence intervals
 *          for each state and action
 */
pair<MDP, numvecvec> confidence_intervals_sa(const MDPO& mdpo, prec_t level, Norm norm) {
    assert(level >= 0.0 && level <= 1.0);

    MDP nominal;
    numvecvec nature;
}

}} // namespace craam::bayesian
