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
 * called **outcome**. The transition probabilities of the MDP itself are ignored.
 *
 * The policy consists of an index for the state and distribution for the policy
 *
 * The update for a state value recomputes the worst case, which should ensure
 * the convergence of the modified policy policy iteration in robust cases.
 */
class SARobustOutcomeBellman {
public:
    /// the policy of the decision maker
    using dec_policy_type = long;
    /// the policy of nature
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
                prec_t newvalue;
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
        try {
            return value_fix_state(mdpo[stateid], valuefunction, discount,
                                   action_pol.first, action_pol.second);
        } catch (ModelError& e) {
            e.set_state(stateid);
            throw e;
        }
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

}} // namespace craam::algorithms
