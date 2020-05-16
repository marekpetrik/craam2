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

#include "craam/MDPO.hpp"
#include "craam/algorithms/nature_declarations.hpp"
#include "craam/algorithms/values.hpp"

namespace craam { namespace algorithms {

/**
 * The class abstracts some operations of value / policy iteration in order to
 * generalize to various types of robust MDPs. It can be used in place of
 * response in mpi_jac or vi_gs to solve robust MDP objectives for s-rectangular
 * ambiguity. When a policy is specified for a given state then it evolves
 * simply according to the nominal transition probabilities.
 *
 * This Bellman update computes result as a worst case over a set of possible
 * choices of transition probabilities. Each of the available choices is
 * called **outcome**. The transition probabilities of the MDP itself not
 * changed by the nature.
 *
 * The policy consists of an index for the state and distribution for the nature over
 * over the outcomes.
 *
 * The update for a state value recomputes the worst case, which should ensure
 * the convergence of the modified policy policy iteration in robust cases.
 */
class SARobustOutcomeBellman {
public:
    /// the policy of the decision maker
    using dec_policy_type = long;
    /// the policy of nature: distribution over outcomes
    using nat_policy_type = numvec;
    /// action of the decision maker AND distribution of nature
    using policy_type = pair<typename SARobustOutcomeBellman::dec_policy_type,
                             typename SARobustOutcomeBellman::nat_policy_type>;

protected:
    const MDPO& mdpo;
    /// How to combine the values from a robust solution
    SANature nature;
    /// Partial policy specification (action -1 is ignored and optimized)
    vector<dec_policy_type> decision_policy;
    /// Initial policy specification for the decision maker (should be never changed)
    const vector<dec_policy_type> initial_policy;

public:
    /**
     * @param mdpo MDPO definition. Does not take ownership
     * @param nature Natures response function to outcomes. Uses the mean
     *              value by default.
     * @param initial_policy Fix policy for some states. Negative value
     *         means that the action is not provided and should be optimized
     */
    SARobustOutcomeBellman(const MDPO& mdpo, const SANature& nature = nats::average(),
                           indvec initial_policy = indvec(0))
        : mdpo(mdpo), nature(nature), decision_policy(move(initial_policy)),
          initial_policy(decision_policy) {}

    /// @brief Number of states in the MDPO
    size_t state_count() const { return mdpo.size(); }

    /**
     * Computes the Bellman update.
     *
     * @param solution Solution to update
     * @param state State for which to compute the Bellman update
     * @param stateid Index of the state
     * @param valuefunction The full value function
     * @param discount Discount factor
     *
     * @returns Best value and action (decision maker and nature)
     */
    pair<prec_t, policy_type> policy_update(long stateid, const numvec& valuefunction,
                                            prec_t discount) const {
        try {
            prec_t newvalue = 0;
            policy_type action;
            numvec transition;

            // check whether this state should only be evaluated or also optimized
            // optimizing action
            if (decision_policy.empty() || decision_policy[stateid] < 0) {
                long actionid;
                tie(actionid, transition, newvalue) = value_max_state(
                    mdpo[stateid], valuefunction, discount, stateid, nature);
                action = make_pair(actionid, move(transition));
            }
            // fixed-action, do not copy
            else {
                const long actionid = decision_policy[stateid];
                tie(transition, newvalue) = value_fix_state(
                    mdpo[stateid], valuefunction, discount, actionid, stateid, nature);
                action = make_pair(actionid, move(transition));
            }
            return make_pair(newvalue, move(action));
        } catch (ModelError& e) {
            e.set_state(stateid);
            throw e;
        }
    }

    /** Computes the Bellman update for a given policy.
            The function is called for the particular state. */
    prec_t compute_value(const policy_type& action_pol, long stateid,
                         const numvec& valuefunction, prec_t discount) const {
        return value_fix_state(mdpo[stateid], valuefunction, discount, action_pol.first,
                               action_pol.second);
    }

    /** Returns a reference to the transition probabilities
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    Transition transition(long stateid, const policy_type& action) const {
        assert(stateid >= 0 && size_t(stateid) < state_count());
        const StateO& s = mdpo[stateid];
        if (s.is_terminal()) {
            return Transition::empty_tran();
        } else {
            return s[action.first].mean_transition(action.second);
        }
    }

    /**
     * Sets the policy that will be used by the update. The value -1 for a state
     * means that the action will be optimized.
     *
     * If the length is 0, then the decision maker's policy is replaced by the initial
     * policy (or an equivalent).
     */
    void set_decision_policy(
        const vector<dec_policy_type>& policy = vector<dec_policy_type>(0)) {
        if (policy.empty()) {
            if (initial_policy.empty()) {
                // if it is empty, then this should have no effect,
                // but it prevents repeated shortening of the vector
                fill(decision_policy.begin(), decision_policy.end(), -1);

            } else {
                decision_policy = initial_policy;
            }

        } else {
            assert(policy.size() == mdpo.size());
            decision_policy = policy;
        }
    }

    /** Returns the reward for the action
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    prec_t reward(long stateid, const policy_type& action) const {
        const StateO& s = mdpo[stateid];
        if (s.is_terminal()) {
            return 0;
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
 * This Bellman update computes result as a worst case over a set of possible
 * choices of transition probabilities. Each of the available choices is
 * called **outcome**. The transition probabilities of the MDP itself are not
 * changed by the nature.
 *
 * The class checks that, for each state, all actions have the same number of
 * outcomes and the weights of these outcomes are all the same.
 *
 * The class abstracts some operations of value / policy iteration in order to
 * generalize to various types of robust MDPs. It can be used in place of
 * response in mpi_jac or vi_gs to solve robust MDP objectives for s-rectangular
 * ambiguity. When a policy is specified for a given state then it evolves
 * simply according to the nominal transition probabilities.
 *
 * @see PlainBellman for a plain implementation
 */
class SRobustOutcomeBellman {
public:
    /// action type of the decision maker
    using dec_policy_type = numvec;
    /// the policy of nature (this is the distribution over outcomes)
    using nat_policy_type = numvec;
    /// distribution the decision maker, distribution of nature
    using policy_type = pair<typename SRobustOutcomeBellman::dec_policy_type,
                             typename SRobustOutcomeBellman::nat_policy_type>;
    /// Type of the state
    using state_type = State;

protected:
    /// The model used to compute the response
    const MDPO& mdpo;
    /// Reference to the function that is used to call the nature
    const SNatureOutcome& nature;
    /// Partial policy specification for the decision maker (action -1 is ignored and optimized)
    vector<dec_policy_type> decision_policy;
    /// Initial policy specification for the decision maker (should be never changed)
    const vector<dec_policy_type> initial_policy;

public:
    /**
     * Constructs the object from a policy and a specification of nature. Action are
     * optimized only in states in which policy is -1 (or < 0)
     * @param policy Fixed randomized policy for a subset of all states.
     *               If empty or omitted then all states are optimized.
     *               An empty vector for a specific state means that the
     *               action will be optimized for that state.
     * @param nature Function that computes the nature's response
     */
    SRobustOutcomeBellman(const MDPO& mdpo, const SNatureOutcome& nature,
                          vector<dec_policy_type> policy = vector<dec_policy_type>(0))
        : mdpo(mdpo), nature(nature), decision_policy(move(policy)),
          initial_policy(decision_policy) {

        // check that the outcomes and their weights are the same for all actions in one state
        for (size_t idstate = 0; idstate < mdpo.size(); ++idstate) {
            const StateO& state = mdpo[idstate];
            if (state.is_terminal()) continue; // skip states with no actions
            // there is at least one action
            const auto& dst0 = state.get_action(0).get_distribution();
            for (size_t idaction = 1; idaction < state.size(); ++idaction) {
                const auto& dst = state.get_action(idaction).get_distribution();
                // check the outcome counts
                if (dst.size() != dst0.size()) {
                    throw ModelError(
                        "Number of outcomes must match across all actions in "
                        "a single states",
                        idstate, idaction);
                }
                // check that the distributions are the same (approximately)
                for (size_t idoutcome = 0; idoutcome < dst.size(); ++idoutcome) {
                    if (std::abs(dst0[idoutcome] - dst[idoutcome]) > EPSILON) {
                        throw ModelError("Distribution of outcomes must match across all "
                                         "actions in a single state",
                                         idstate, idaction, idoutcome);
                    }
                }
            }
        }
    }

    // **** BEGIN: Bellman Interface Methods  ********

    /// Number of MDP states
    size_t state_count() const { return mdpo.size(); }

    /**
     * Computes the Bellman update.
     *
     * @param solution Solution to update
     * @param state State for which to compute the Bellman update
     * @param stateid  Index of the state
     * @param valuefunction The full value function
     * @param discount Discount factor
     *
     * @returns Best value and action distribution (decision maker and nature).
     *    The distribution is empty for actions that have 0 probability assigned
     *    to them.
     */
    pair<prec_t, policy_type> policy_update(long stateid, const numvec& valuefunction,
                                            prec_t discount) const {

        const StateO& state = mdpo[stateid];

        if (state.is_terminal()) return {0, {numvec(0), numvec(0)}};

        // check whether this state should only be evaluated or also optimized
        numvec init_policy =
            decision_policy.empty() ? numvec(0) : decision_policy[stateid];

        auto [action, transitions, newvalue] =
            nature(stateid, init_policy, state.get_action(0).get_distribution(),
                   compute_zvalues(state, valuefunction, discount));

        assert(!isinf(newvalue));
        assert(action.size() == state.size());
        policy_type action_response = make_pair(move(action), move(transitions));

        return {newvalue, move(action_response)};
    }

    /**
     * Computes value function using the provided policy. Used in policy evaluation.
     *
     * @param solution Solution used to infer the current policy
     * @param state State for which to compute the Bellman update
     * @param stateid Index of the state
     * @param valuefunction Value function
     * @param discount Discount factor
     * @returns New value for the state
     */
    prec_t compute_value(const policy_type& action, long stateid,
                         const numvec& valuefunction, prec_t discount) const {
        try {
            return value_fix_state(mdpo[stateid], valuefunction, discount, action.first,
                                   action.second);
        } catch (ModelError& e) {
            e.set_state(stateid);
            throw e;
        }
    }

    /**
     * Returns a reference to the transition probabilities
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    Transition transition(long stateid, const policy_type& action) const {
        assert(stateid >= 0 && size_t(stateid) < state_count());
        const StateO& s = mdpo[stateid];
        if (s.is_terminal()) {
            return Transition::empty_tran();
        } else {
            // compute the weighted average of transition probabilies
            assert(s.size() == action.first.size());
            Transition result;
            for (size_t ai = 0; ai < s.size(); ai++) {
                // make sure that the action is being taken
                if (action.first[ai] > EPSILON) {
                    // recall that all the weight of the nature is the same for
                    // each action
                    result.probabilities_add(action.first[ai],
                                             s[ai].mean_transition(action.second));
                }
            }
            return result;
        }
    }

    /** Returns the reward for the action
     *
     * @param stateid State for which to get the transition probabilites
     * @param action Which action is taken
     */
    prec_t reward(long stateid, const policy_type& action) const {
        const StateO& s = mdpo[stateid];
        if (s.is_terminal()) {
            return 0;
        } else {
            prec_t result = 0;

            assert(s.size() == action.first.size());
            for (size_t ai = 0; ai < s.size(); ai++) {
                // only consider actions that have non-zero transition probabilities
                if (action.first[ai] > EPSILON) {
                    result += action.first[ai] * s[ai].mean_reward(action.second);
                }
            }
            return result;
        }
    }

    /**
     * Sets the policy that will be used by the update. The value -1 for a state
     * means that the action will be optimized.
     *
     * If the length is 0, then the decision maker's policy is replaced by the initial
     * policy (or an equivalent).
     */
    void set_decision_policy(
        const vector<dec_policy_type>& policy = vector<dec_policy_type>(0)) {
        if (policy.empty()) {
            if (initial_policy.empty()) {
                // if it is empty, then this should have no effect,
                // but it prevents repeated shortening of the vector
                fill(decision_policy.begin(), decision_policy.end(), numvec(0));
            } else {
                decision_policy = initial_policy;
            }
        } else {
            assert(policy.size() == mdpo.size());
            decision_policy = policy;
        }
    }

    // **** END: Bellman Interface Methods  ********
};

}} // namespace craam::algorithms
