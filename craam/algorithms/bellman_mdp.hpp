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
#include "craam/algorithms/nature_declarations.hpp"
#include "craam/algorithms/values.hpp"
#include "values_mdp.hpp"

namespace craam { namespace algorithms {

// **************************************************************************
// Helper classes to handle computing of the best response
// **************************************************************************

/**
A Bellman update class for solving regular Markov decision processes. This class
abstracts away from particular model properties and the goal is to be able to
plug it in into value or policy iteration methods for MDPs.

Many of the methods are parametrized by the type of the state.

The class also allows to use an initial policy specification. See the
constructor for the definition.

The class does not own the MDP.
*/
class PlainBellman {
protected:
    /// MDP definition
    const MDP& mdp;
    /// Partial policy specification (action -1 is ignored and optimized)
    const indvec initial_policy;

public:
    /**
      * Provides the type of policy for each state (int represents a deterministic
      *  policy)
      */
    using policy_type = long;
    /// Type of the state
    using state_type = State;

    /// Constructs the update with no constraints on the initial policy
    PlainBellman(const MDP& mdp) : mdp(mdp), initial_policy(0) {}

    /**
     * A partial policy that can be used to fix some actions
     *  @param policy policy[s] = -1 means that the action should be optimized in
     * the state policy of length 0 means that all actions will be optimized
     */
    PlainBellman(const MDP& mdp, indvec policy)
        : mdp(mdp), initial_policy(move(policy)) {}

    /// Number of MDP states
    size_t state_count() const { return mdp.size(); }

    /**
     *  Computes the Bellman update and returns the optimal action.
     *  @returns New value for the state and the policy
     */
    pair<prec_t, policy_type> policy_update(long stateid, const numvec& valuefunction,
                                            prec_t discount) const {

        // check whether this state should only be evaluated
        if (initial_policy.empty() || initial_policy[stateid] < 0) { // optimizing
            prec_t newvalue;
            policy_type action;

            tie(action, newvalue) =
                value_max_state(mdp[stateid], valuefunction, discount);
            return make_pair(newvalue, action);
        } else { // fixed-action, do not copy
            return {value_fix_state(mdp[stateid], valuefunction, discount,
                                    initial_policy[stateid]),
                    initial_policy[stateid]};
        }
    }

    /**
   *  Computes value function update using the current policy
   * @returns New value for the state
   */
    prec_t compute_value(const policy_type& action, long stateid,
                         const numvec& valuefunction, prec_t discount) const {
        return value_fix_state(mdp[stateid], valuefunction, discount, action);
    }

    /** Returns a reference to the transition probabilities
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    const Transition& transition(long stateid, const policy_type& action) const {
        assert(stateid >= 0 && size_t(stateid) < state_count());
        const State& s = mdp[stateid];
        if (s.is_terminal()) {
            if (action < 0)
                return Transition::empty_tran;
            else
                throw invalid_argument("Unknown action taken in a terminal state.");
        } else {
            return static_cast<const Transition&>(s[action]);
        }
    }

    /** Returns the reward for the action
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    prec_t reward(long stateid, const policy_type& action) const {
        const State& s = mdp[stateid];
        if (s.is_terminal()) {
            if (action < 0)
                return 0;
            else
                throw invalid_argument("Unknown action taken in a terminal state.");
        } else {
            return s[action].mean_reward();
        }
    }
};

/**
 * The class abstracts some operations of value / policy iteration in order to
 * generalize to various types of robust MDPs. It can be used in place of response
 * in mpi_jac or vi_gs to solve robust MDP objectives.
 *
 * The class does not own the MDP and nature.
 *
 * @see PlainBellman for a plain implementation
 */
class SARobustBellman {
protected:
    /// MDP definition
    const MDP& mdp;
    /// Reference to the function that is used to call the nature
    const SANature& nature;
    /// Partial policy specification (action -1 is ignored and optimized)
    const indvec initial_policy;

public:
    /// action of the decision maker, distribution of nature
    using policy_type = pair<long, numvec>;
    /// Type of the state
    using state_type = State;

    /**
      Constructs the object from a policy and a specification of nature. Action are
      optimized only in states in which policy is -1 (or < 0)
      @param policy Index of the action to take for each state
      @param nature Function that describes nature's response
      */
    SARobustBellman(const MDP& mdp, const SANature& nature, indvec policy)
        : mdp(mdp), nature(nature), initial_policy(move(policy)) {}

    /**
      Constructs the object from a specification of nature. No decision maker's
      policy is provided.
      @param nature Function that describes nature's response
      */
    SARobustBellman(const MDP& mdp, const SANature& nature)
        : mdp(mdp), nature(nature), initial_policy(0) {}

    size_t state_count() const { return mdp.size(); }

    /**
      Computes the Bellman update and updates the action in the solution to the best
      response It does not update the value function in the solution.
      @param solution Solution to update
      @param state State for which to compute the Bellman update
      @param stateid  Index of the state
      @param valuefunction Value function
      @param discount Discount factor
      @returns New value for the state
      */
    pair<prec_t, policy_type> policy_update(long stateid, const numvec& valuefunction,
                                            prec_t discount) const {
        prec_t newvalue = 0;
        policy_type action;
        numvec transition;

        // check whether this state should only be evaluated or also optimized
        // optimizing action
        if (initial_policy.empty() || initial_policy[stateid] < 0) {
            long actionid;
            tie(actionid, transition, newvalue) =
                value_max_state(mdp[stateid], valuefunction, discount, stateid, nature);
            action = make_pair(actionid, move(transition));
        }
        // fixed-action, do not copy
        else {
            prec_t newvalue;
            const long actionid = initial_policy[stateid];
            tie(transition, newvalue) = value_fix_state(
                mdp[stateid], valuefunction, discount, actionid, stateid, nature);
            action = make_pair(actionid, move(transition));
        }
        return make_pair(newvalue, move(action));
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
    prec_t compute_value(const policy_type& action, long stateid,
                         const numvec& valuefunction, prec_t discount) const {
        return value_fix_state(mdp[stateid], valuefunction, discount, action.first,
                               action.second);
    }

    /** Returns a reference to the transition probabilities
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    Transition transition(long stateid, const policy_type& action) const {
        assert(stateid >= 0 && size_t(stateid) < state_count());
        const State& s = mdp[stateid];
        if (s.is_terminal()) {
            if (action.first < 0)
                return Transition::empty_tran;
            else
                throw invalid_argument("Unknown action taken in a terminal state.");
        } else {
            return s[action.first].mean_transition(action.second);
        }
    }

    /** Returns the reward for the action
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    prec_t reward(long stateid, const policy_type& action) const {
        const State& s = mdp[stateid];
        if (s.is_terminal()) {
            if (action.first < 0)
                return 0;
            else
                throw invalid_argument("Unknown action taken in a terminal state.");
        } else {
            return s[action.first].mean_reward(action.second);
        }
    }
};

/**
 * The class abstracts some operations of value / policy iteration in order to
 * generalize to various types of robust MDPs. It can be used in place of
 * response in mpi_jac or vi_gs to solve robust MDP objectives for s-rectangular
 * ambiguity. When a policy is specified for a given state then it evolves
 * simply according to the nominal transition probabilities.
 *
 * @see PlainBellman for a plain implementation
 */
class SRobustBellman {
protected:
    const MDP& mdp;
    /// Reference to the function that is used to call the nature
    const SNature& nature;
    /// Partial policy specification (action -1 is ignored and optimized)
    const indvec initial_policy;

public:
    /// distribution the decision maker, distribution of nature
    using policy_type = pair<numvec, numvecvec>;
    // Type of the state
    using state_type = State;

    /**
      Constructs the object from a policy and a specification of nature. Action are
      optimized only in states in which policy is -1 (or < 0)
      @param policy Index of the action to take for each state
      @param nature Function that describes nature's response
      */
    SRobustBellman(const MDP& mdp, const SNature& nature, indvec policy)
        : mdp(mdp), nature(nature), initial_policy(move(policy)) {}

    /**
      Constructs the object from a specification of nature. No decision maker's
      policy is provided.
      @param nature Function that describes nature's response
      */
    SRobustBellman(const MDP& mdp, const SNature& nature)
        : mdp(mdp), nature(nature), initial_policy(0) {}

    /// Number of MDP states
    size_t state_count() const { return mdp.size(); }

    // **** BEGIN: Bellman Interface Methods  ********

    /**
      Computes the Bellman update.

      @param solution Solution to update
      @param state State for which to compute the Bellman update
      @param stateid  Index of the state
      @param valuefunction The full value function
      @param discount Discount factor

      @returns Best value and action (decision maker and nature)
      */
    pair<prec_t, policy_type> policy_update(long stateid, const numvec& valuefunction,
                                            prec_t discount) const {
        prec_t newvalue = 0;
        policy_type action_response;
        numvec action;
        vector<numvec> transitions;

        const State& state = mdp[stateid];

        if (state.is_terminal())
            return make_pair(-1, make_pair(numvec(0), vector<numvec>(0)));

        // check whether this state should only be evaluated or also optimized
        // optimizing action
        if (initial_policy.empty() || initial_policy[stateid] < 0) {
            tie(action, transitions, newvalue) =
                nature(stateid, compute_probabilities(state),
                       compute_zvalues(state, valuefunction, discount));

        }
        // fixed-action, do not copy
        else {
            long actionid = initial_policy[stateid];
            newvalue = value_fix_state(state, valuefunction, discount, actionid);
            transitions = vector<numvec>(state.size()); // create an entry for each state
            // but set only the one that is relevant
            transitions[actionid] = state[actionid].get_probabilities();
            // set the actual action value
            action = numvec(state.size(), 0.0);
            action[actionid] = 1.0;
        }

        assert(action.size() == state.size());
        action_response = make_pair(move(action), move(transitions));
        return make_pair(newvalue, action_response);
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
    prec_t compute_value(const policy_type& action, long stateid,
                         const numvec& valuefunction, prec_t discount) const {
        return value_fix_state(mdp[stateid], valuefunction, discount, action.first,
                               action.second);
    }

    // **** END: Bellman Interface Methods  ********
};

}} // namespace craam::algorithms
