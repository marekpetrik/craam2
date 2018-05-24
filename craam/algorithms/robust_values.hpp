// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/**
Robust MDP methods for computing value functions.
*/
#pragma once

#include "craam/RMDP.hpp"
#include "craam/algorithms/nature_declarations.hpp"

#include <rm/range.hpp>

#include "values.hpp"
#include <functional>
#include <type_traits>

namespace craam::algorithms{

using namespace std;
using namespace util::lang;


// *******************************************************
// RegularAction computation methods
// *******************************************************

/**
Computes an ambiguous value (e.g. robust) of the action, depending on the type
of nature that is provided.

\param action Action for which to compute the value
\param valuefunction State value function to use
\param discount Discount factor
\param nature Method used to compute the response of nature.
*/
template<class T>
inline vec_scal_t value_action(const RegularAction& action, const numvec& valuefunction,
                        prec_t discount, long stateid, long actionid, const SANature& nature){

    const numvec& rewards = action.get_outcome().get_rewards();
    const indvec& nonzero_indices = action.get_outcome().get_indices();

    numvec zvalues(rewards.size()); // values for individual states - used by nature.

    #pragma omp simd
    for(size_t i = 0; i < rewards.size(); i++){
        zvalues[i] = rewards[i] + discount * valuefunction[nonzero_indices[i]];
    }

    return nature(stateid, actionid, action.get_outcome().get_probabilities(), zvalues);
}


// *******************************************************
// WeightedOutcomeAction computation methods
// *******************************************************

/**
Computes the maximal outcome distribution constraints on the nature's distribution.
Does not work when the number of outcomes is zero.

\param action Action for which the value is computed
\param valuefunction Value function reference
\param discount Discount factor
\param nature Method used to compute the response of nature.

\return Outcome distribution and the mean value for the choice of the nature
 */
template<class T>
inline vec_scal_t value_action(const WeightedOutcomeAction& action, numvec const& valuefunction,
                                prec_t discount, long stateid, long actionid, const SANature& nature) {

    assert(action.get_distribution().size() == action.get_outcomes().size());

    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");

    numvec outcomevalues(action.size());
    for(size_t i = 0; i < action.size(); i++)
        outcomevalues[i] = action[i].value(valuefunction, discount);

    return nature(stateid, actionid, action.get_distribution(), outcomevalues);
}


// *******************************************************
// State computation methods
// *******************************************************


/**
Computes the value of a fixed action and any response of nature.

\param state State to compute the value for
\param valuefunction Value function to use in computing value of states.
\param discount Discount factor
\param nature Instance of a nature optimizer

\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType, class T>
inline vec_scal_t
value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              long actionid, const SANature& nature) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return make_pair(numvec(0),0);

    assert(actionid >= 0 && actionid < long(state.size()));

    if(actionid < 0 || actionid >= long(state.size())) throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if(!state.is_valid(actionid)) throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount, nature);
}

/**
Finds the greedy action and its value for the given value function.
This function assumes a robust or optimistic response by nature depending on the provided
ambiguity.

When there are no actions, the state is assumed to be terminal and the return is 0.

\param state State to compute the value for
\param valuefunction Value function to use in computing value of states.
\param discount Discount factor
\param stateid Number of the state in the MDP
\param natures Method used to compute the response of nature; one for each action available in the state.

\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<typename AType, typename T>
inline ind_vec_scal_t
value_max_state(const SAState<AType>& state, const numvec& valuefunction,
                prec_t discount, long stateid, const SANature& nature) {

    // can finish immediately when the state is terminal
    if(state.is_terminal())
        return make_tuple(-1,numvec(),0);

    // make sure that the number of natures is the same as the number of actions
    assert(natures.size() == state.size());

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();

    long result = -1;
    numvec result_outcome;

    for(size_t i = 0; i < state.get_actions().size(); i++){
        const auto& action = state[i];

        // skip invalid state.get_actions()
        if(!state.is_valid(i)) continue;

        auto value = value_action(action, valuefunction, discount, stateid, i, nature);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = long(i);
            result_outcome = move(value.first);
        }
    }

    // if the result has not been changed, that means that all actions are invalid
    if(result == -1)
        throw invalid_argument("all actions are invalid.");

    return make_tuple(result,result_outcome,maxvalue);
}


// **************************************************************************
// Helper classes to handle computing of the best response
// **************************************************************************


/**
The class abstracts some operations of value / policy iteration in order to generalize to
various types of robust MDPs. It can be used in place of response in mpi_jac or vi_gs to 
solve robust MDP objectives.

@see PlainBellman for a plain implementation
*/
template<class SType>
class SARobustBellman  {
protected:

    /// Reference to the function that is used to call the nature
    SANature nature;

public:
    // action of the decision maker, distribution of nature
    using policy_type = pair<long,numvec>;

    /**
    Constructs the object from a policy and a specification of nature. Action are optimized
    only in states in which policy is -1 (or < 0)
    @param policy Index of the action to take for each state
    @param nature Function that describes nature's response
    */
    SARobustBellman(SANature nature, indvec policy): nature(move(nature)),
                    initial_policy(move(policy)) {};

    /**
    Constructs the object from a specification of nature. No decision maker's
    policy is provided.
    @param nature Function that describes nature's response
    */
    SARobustBellman(SANature nature) : nature(move(nature)),
                    initial_policy(0) {};

    /**
    Computes the Bellman update and updates the action in the solution to the best response
    It does not update the value function in the solution.
    @param solution Solution to update
    @param state State for which to compute the Bellman update
    @param stateid  Index of the state
    @param valuefunction Value function
    @param discount Discount factor
    @returns New value for the state
     */
    pair<prec_t, policy_type> policy_update(const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{
        prec_t newvalue = 0;
        policy_type action;

        // check whether this state should only be evaluated or also optimized
        // optimizing action
        if(initial_policy.empty() || initial_policy[stateid] < 0){
            tie(action, action, newvalue) =
                    value_max_state(state, valuefunction, discount, stateid, nature);
        }
        // fixed-action, do not copy
        else{
            prec_t newvalue;
            const long actionid = initial_policy[stateid];
            tie(action, newvalue) = value_fix_state(state, valuefunction, discount, actionid, nature);
        }
        return {newvalue, action};
    }

    /**
    Computes value function using the provided policy. Used in policy evaluation.
    @param solution Solution used to infer the current policy
    @param state State for which to compute the Bellman update
    @param stateid Index of the state
    @param valuefunction Value function
    @param discount Discount factor
    @returns New value for the state
    */
    prec_t compute_value(const policy_type& action, const SType& state, const numvec& valuefunction,
                            prec_t discount) const{
        return value_fix_state(state, valuefunction, discount, action.first,
                action.second);
    }

protected:
    /// Partial policy specification (action -1 is ignored and optimized)
    const indvec initial_policy;
};


// **************************************************************************
// Wrapper methods
// **************************************************************************

/** 
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the last state to the first one, the states should be ordered
in the temporal order.

This is a simplified method interface. Use vi_gs with PolicyNature for full functionality.

\param mdp      The MDP to solve
\param discount Discount factor.
\param nature   Response of nature, the function is the same for all states and actions.
\param thresholds Parameters passed to nature response functions.
                    One value per state and then one value per action.
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
\param policy    Partial policy specification. Optimize only actions that are  policy[state] = -1. Use
                 policy length 0 to optimize all actions.
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.


\returns Solution that can be used to compute the total return, or the optimal policy.
*/
template<class SType>
inline auto rsolve_vi(const GRMDP<SType>& mdp, prec_t discount,
                        const SANature& nature,
                        numvec valuefunction=numvec(0), const indvec& policy = indvec(0),
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC){

    return vi_gs<SType, SARobustBellman<SType>>(mdp, discount, move(valuefunction), nature,
            SARobustBellman<SType>(policy), iterations, maxresidual);
}


/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

This is a simplified method interface. Use mpi_jac with PolicyNature for full functionality.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param nature Response of nature; the same value is used for all states and actions.
\param valuefunction Initial value function
\param policy Partial policy specification. Optimize only actions that are  policy[state] = -1
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param print_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType>
inline auto rsolve_mpi(const GRMDP<SType>& mdp, prec_t discount,
                const SANature& nature,
                const numvec& valuefunction=numvec(0), const indvec& policy = indvec(0),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC,
                bool print_progress=false) {

    return mpi_jac<SType, SARobustBellman<SType>>(mdp, discount, valuefunction, nature,
                    SARobustBellman<SType>(policy),
                    iterations_pi, maxresidual_pi,
                    iterations_vi, maxresidual_vi, 
                    print_progress);
}

// ******************************************************************
// Helper functions for a Python or R interface
// ******************************************************************

/**
Converts a string representation of nature response to the appropriate nature response call.
This function is useful when the code is used within a python or R libraries. The values values
correspond to the function definitions, and ones that are currently supported are:

- robust_unbounded
- optimistic_unbounded
- robust_l1
- optimistic_l1
*/
/*inline NatureResponse<prec_t> string_to_nature(string nature){
    if(nature == "robust_unbounded") return robust_unbounded;
    if(nature == "optimistic_unbounded") return optimistic_unbounded;
    if(nature == "robust_l1") return robust_l1;
    if(nature == "optimistic_l1") return optimistic_l1;
    throw invalid_argument("Unknown nature.");
}*/


}
