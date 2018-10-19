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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace craam {

using namespace std;

// **************************************************************************************
// *** Regular action
// **************************************************************************************

/**
Action in a regular MDP. There is no uncertainty and
the action contains only a single outcome.
*/
class Action : public Transition {
public:
    /** Creates an empty action. */
    Action() : Transition() {}

    /** Initializes outcomes to the provided transition vector */
    Action(const Transition& outcome) : Transition(outcome) {}

    /** Appends a string representation to the argument */
    string to_string() const { return "1(reg)"; };

    /** Whether the action has some transitions */
    bool is_valid() const { return size() > 0; }

    /** Returns the mean transition probabilities. Ignore rewards. */
    Transition mean_transition() const { return *this; }

    /** Returns the mean transition probabilities. Ignore rewards.
      @param natpolicy Nature can choose a non-zero state to go to
      */
    Transition mean_transition(const numvec& natpolicy) const {
        assert(natpolicy.size() == get_indices().size());
        return Transition(get_indices(), natpolicy, numvec(size(), 0.0));
    }

    /** Returns a json representation of the action
        @param actionid Whether to include action id*/
    string to_json(long actionid = -1) const {
        stringstream result;
        result << "{";
        result << "\"actionid\" : ";
        result << std::to_string(actionid);
        result << ",\"transition\" : ";
        result << Transition::to_json(-1);
        result << "}";
        return result.str();
    }
};

} // namespace craam
