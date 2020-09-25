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

#pragma once

#include "craam/Samples.hpp"
#include "craam/definitions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace craam { namespace msen {

using namespace std;
using namespace util::lang;

template <class Sim> class ThresholdPolicy {

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    ThresholdPolicy(const Sim& sim, long max_inventory,
                    random_device::result_type seed = random_device{}())
        : sim(sim), max_inventory(max_inventory), gen(seed) {}

    /** Returns an action accrding to the s,S policy, orders required amount to have
    the inventory to max level. */
    long operator()(State current_state) {
        return max(0l, max_inventory - current_state);
    }

private:
    /// Internal reference to the originating simulator
    const Sim& sim;
    State max_inventory;
    /// Random number engine
    default_random_engine gen;
};

/// Behaves like a vector of continuous values from 0 to n
class LightArray {
public:
    /// @param n Size of the array
    LightArray(size_t n) : n(n) {}

    /// Size of the array`
    size_t size() const noexcept { return n; }

    /// Just returns the same value back
    long operator[](long index) const noexcept {
        assert(index >= 0 && size_t(index) < n);
        return index;
    }

protected:
    size_t n;
};

/**
 * A simple inventory model simulator allowing for fixed and variable costs,
 * inventory levels and demands.
 *
 * State 0 represents the maximum possible backlog
 */
class InventorySimulator {

public:
    /// Type of states: invenotry level
    using State = long;
    /// Type of actions: how much to purchase
    using Action = long;

    /**
     * Build a model simulator for the inventory problem. The default
     * behavior is to start with an empty inventory level
     *
     * @param demands Probabilities of discrete demand values. Use set_min_demand
     *                  to offset the demands (if needed)
     * @param costs List of costs: 0. variable, 1. fixed, 2. holding, 3. backlog
     * @param sale_price Price of the sale of the product
     * @param limits List of 0. maximum inventory, 1. maximum backlog, 2. maximum order size
     *                  The limits are inclusive (include the last inventory)
     */
    InventorySimulator(numvec demands, array<prec_t, 4> costs, prec_t sale_price,
                       array<long, 3> limits)
        : demand_dist(demands.cbegin(), demands.cend()), demands_prob(move(demands)),
          purchase_cost(costs[0]), delivery_cost(costs[1]), holding_cost(costs[2]),
          backlog_cost(costs[3]), sale_price(sale_price), max_inventory(limits[0]),
          max_backlog(limits[1]), max_order(limits[2]) {

        assert(abs(1.0 - accumulate(demands_prob.cbegin(), demands_prob.cend(), 0.0)) <=
                   EPSILON &&
               "Demand distribution must sum to 1.0");
    }

    /// Sets the seed
    void set_seed(random_device::result_type seed = random_device{}()) { gen.seed(seed); }

    /// Sets the offset of all demand values
    void set_min_demand(long min_demand) { this->min_demand = min_demand; }

    /// Returns the initial state which corresponds to the
    /// empty inventory (and no backlog)
    State init_state() const { return max_backlog; }

    bool end_condition(State inventory) const { return inventory < 0; }

    /**
     * Returns a sample of the reward and a state following
     * a state using random demand distribution.
     *
     * See @a transition_dem for details.
     *
     * @param current_state Current state. The inventory level is
     *                      current_state - max_backlog
     * @param action_order Action obtained from the policy
     *
     * @returns a pair of reward & next inventory level
     */
    pair<prec_t, State> transition(State current_state, Action action_order) noexcept {
        // Generate demand from the normal demand distribution
        long demand = demand_dist(gen) + min_demand;
        // call the transition that computes the demand
        return transition_dem(current_state, action_order, demand);
    }

    /**
     * Returns the next state for a given demand value.
     *
     * Assumes that the order arrives before any of the purchases are made.
     * Holding and backlog costs are applied using the inventory levels in
     * the next state. Sale price s received at the time of sale even if
     * the sale is back-logged.
     *
     * Selling more than is available is simply clamped to the inventory level
     *
     * @param current_state Current state. The inventory level is
     *                      current_state - max_backlog
     * @param action_order Action obtained from the policy
     * @param demand Demand level to assume
     *
     * @returns a pair of reward & next inventory level
     */
    pair<prec_t, State> transition_dem(State current_state, Action action_order,
                                       long demand) const noexcept {
        assert(current_state >= 0 && current_state < state_count());
        assert(action_order >= 0 && action_order <= max_order);
        assert(demand >= 0);

        const long current_inventory = current_state - max_backlog;

        // Compute the next inventory level.
        long next_inventory =
            max(action_order + current_inventory - demand, -max_backlog);
        // Back-calculate how many items were sold
        const long sold_amount = current_inventory - next_inventory + action_order;
        // Clamp the inventory from above to make sure that anything unsold that does not fit is discarded
        next_inventory = min(next_inventory, max_inventory);
        // Compute the obtained revenue
        const prec_t revenue = sold_amount * sale_price;
        // Compute the expense of purchase, holding cost, and backlog cost
        const prec_t expense = action_order * purchase_cost +
                               (action_order > 0 ? delivery_cost : 0.0) +
                               holding_cost * max(next_inventory, 0l) +
                               backlog_cost * -min(next_inventory, 0l);
        // Reward is equivalent to the profit & obtained from revenue & total expense
        const prec_t reward = revenue - expense;

        return {reward, next_inventory + max_backlog};
    }

    /// Returns the number of states
    long state_count() const noexcept { return max_inventory + 1 + max_backlog; }

    /// Returns the number of actions
    long action_count() const noexcept { return max_order; }

    /// Returns an element that supports size and []
    LightArray get_valid_actions(State s) const noexcept {
        return LightArray(action_count());
    }

    /**
     * Calls the function F for all transition probabilities. Can be used to
     * construct an MDP with the transition probabilities that correspond to
     * this inventory management problem.
     *
     * @tparam F A type that can be called as
     * (long fromid, long actionid, long toid, prec_t probability, prec_t reward)
     *
     * @param f MDP construction function
     */
    template <class F> void build_mdp(F&& f) const {
        for (State statefrom = 0; statefrom < state_count(); ++statefrom) {
            for (Action action = 0; action < action_count(); ++action) {
                for (size_t demand = min_demand;
                     demand < min_demand + demands_prob.size(); ++demand) {
                    State stateto;
                    prec_t reward;
                    // simulate a single step of the transition probabilities
                    std::tie(reward, stateto) = transition_dem(statefrom, action, demand);
                    const prec_t probability = demands_prob[demand];
                    // run the method that add the transition probability
                    // and possibly update progress
                    f(statefrom, action, stateto, probability, reward);
                }
            }
        }
    }

    //template <class method> generate_model() {}

protected:
    /// Distribution for the demand
    discrete_distribution<long> demand_dist;
    /// Discrete distribution of demands
    numvec demands_prob;
    // cost structure
    prec_t purchase_cost, delivery_cost, holding_cost, backlog_cost;
    // price to sell the good
    prec_t sale_price;
    // limits on inventory levels
    long max_inventory, max_backlog, max_order;
    /// Random number engine
    default_random_engine gen;
    /// Minimum demand if an offset is needed
    long min_demand = 0;
};

///Inventory policy to be used
using ModelInventoryPolicy = ThresholdPolicy<InventorySimulator>;

}} // namespace craam::msen
