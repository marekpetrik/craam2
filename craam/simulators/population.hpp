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

/**
 * A simulator of an exponential population model with a control action.
 * The effectiveness of the control action depends on the population level.
 */
class PopulationSim {

public:
    /// The type of the growth model
    enum class Growth { Exponential, Logistic };

protected:
    /// Population starts and limits
    long carrying_capacity, init_population;
    /// The expectation and std of the growth rate
    /// for each action and population level
    numvecvec mean_growth_rate, std_growth_rate;
    /// Rewards for each action and population
    /// level
    numvecvec rewards;
    /// Number of available actions: each one represents a
    /// different treatment type or intensity
    uint actioncount;
    /// How growth rate interacts with the population level
    Growth growth_model;
    /// Random number when generating new states
    default_random_engine gen;

public:
    /// Type of state: current population
    using State = long;
    /// Type of actions: whether to apply control/treatment or not
    using Action = long;

    /**
     * Initializes to an exponential growth model by default. The possible
     * population levels are from 0 to the carrying capacity (inclusive).
     *
     * @param carrying_capacity Maximum possible  population
     * @param initial_population Population at the start of the simulation
     * @param actioncount Number of actions available to the decision maker,
     *          each action represents a different treatment type or intensity
     *          and has a different growth rate associated with it
     * @param mean_growth_rate Mean of the population growth rate for
     *                          each action and each population level:
     *                          mean_growth_rate[action][population]
     * @param std_growth_rate Standard deviation of the growth rate for
     *                          each action and each population level:
     *                          std_growth_rate[action][population]
     * @param rewards Received rewards for each action and each population level:
     *                          reward[action][population]
     * @param std_observation Standard deviation for the observation from the
     *                          actual underlying population
     * @param seed Seed for random number generation
     */
    PopulationSim(long carrying_capacity, long init_population, uint actioncount,
                  numvecvec mean_growth_rate, numvecvec std_growth_rate,
                  numvecvec rewards, Growth growth_model = Growth::Exponential,
                  random_device::result_type seed = random_device{}())
        : carrying_capacity(carrying_capacity), init_population(init_population),
          mean_growth_rate(mean_growth_rate), std_growth_rate(std_growth_rate),
          rewards(rewards), actioncount(actioncount), growth_model(growth_model),
          gen(seed) {

        // check whether the provided growth rate parameters are of the correct
        // dimensions

        if (mean_growth_rate.size() != actioncount)
            throw invalid_argument(
                "Growth rate exp size must match the number of actions.");
        if (std_growth_rate.size() != actioncount)
            throw invalid_argument(
                "Growth rate std size must match the number of actions.");
        if (rewards.size() != actioncount)
            throw invalid_argument(
                "Growth rate std size must match the number of actions.");

        for (long ai = 0; ai < actioncount; ai++) {
            if (State(mean_growth_rate[ai].size()) != 1 + carrying_capacity)
                throw invalid_argument("Mean growth rate size (" +
                                       std::to_string(mean_growth_rate[ai].size()) +
                                       ") must match carrying capacity + 1 for action " +
                                       std::to_string(ai));

            if (State(std_growth_rate[ai].size()) != 1 + carrying_capacity)
                throw invalid_argument("Std of growth rate size (" +
                                       std::to_string(std_growth_rate[ai].size()) +
                                       ") must match carrying "
                                       "capacity + 1 for action " +
                                       std::to_string(ai));

            if (State(rewards[ai].size()) != 1 + carrying_capacity)
                throw invalid_argument("Rewards size (" +
                                       std::to_string(rewards[ai].size()) +
                                       ") must match carrying capacity + 1 for action " +
                                       std::to_string(ai));
        }
    }

    /// Returns the initial state
    long init_state() const { return init_population; }

    /// The simulation does not have a defined end
    bool end_condition(State population) const { return false; }

    /**
      * Returns a sample of the reward and a population level following an action & current population level
      *
      * The actions represent different types of intensities of management.
      *
      * The treatment action carries a fixed cost control_reward (or a negative reward of -4000) of applying the treatment.
      * There is a variable population dependent
      * cost invasive_reward (-1) that represents the economic (or ecological) damage of the invasive species.
      *
      * The realized growth rate is a random variable with normal noise around the
      * true expected rate.
      *
      * @param current_population Current population level
      * @param action Which control action to take
      *
      * \returns a pair of reward & next population level
      s*/
    pair<prec_t, State> transition(State current_population, Action action) {

        if (action >= actioncount) {
            throw invalid_argument("Action must be less than actioncount.");
        }
        if (current_population > carrying_capacity) {
            throw invalid_argument("Population must be at most the carrying capacity.");
        }

        const prec_t exp_growth_rate_act = mean_growth_rate[action][current_population];
        const prec_t std_growth_rate_act = std_growth_rate[action][current_population];
        normal_distribution<prec_t> growth_rate_distribution(exp_growth_rate_act,
                                                             std_growth_rate_act);

        // make sure that the growth rate cannot be negative
        prec_t growth_rate = std::max(0.0, growth_rate_distribution(gen));

        long next_population = 0;

        if (growth_model == Growth::Exponential) {
            next_population = clamp(long(std::round(growth_rate * current_population)),
                                    0l, carrying_capacity);
        } else if (growth_model == Growth::Logistic) {
            // see https://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth
            auto pop_increase = (growth_rate - 1.0) * current_population *
                                (carrying_capacity - current_population) /
                                prec_t(carrying_capacity);
            next_population = clamp(long(std::round(current_population + pop_increase)),
                                    0l, carrying_capacity);
        } else {
            throw invalid_argument("Unsupported population model.");
        }

        prec_t reward = rewards[action][current_population];
        return {reward, next_population};
    }

    Growth get_growth() const { return growth_model; }
    void set_growth(Growth model) { growth_model = model; }

    size_t state_count() const { return carrying_capacity + 1; }
    size_t action_count(State s) const { return actioncount; }
};

/**
A policy for population management that depends on the population
threshold & control probability.

\tparam Sim Simulator class for which the policy is to be constructed.
            Must implement an instance method actions(State).
 */
class PopulationPol {

public:
    using State = typename PopulationSim::State;
    using Action = typename PopulationSim::Action;

    PopulationPol(const PopulationSim& sim, long threshold_control = 0,
                  prec_t prob_control = 0.5,
                  random_device::result_type seed = random_device{}())
        : sim(sim), threshold_control(threshold_control), prob_control(prob_control),
          gen(seed) {
        control_distribution = binomial_distribution<int>(1, prob_control);
    }

    /**
     * Provides a control action depending on the current population level.
     * If the population level is below a certain threshold,
     * the policy is not to take the control measure. Otherwise,
     * it takes a control measure with a specific probability (which introduces
     * randomness in the policy).
    */
    long operator()(long current_state) {
        if (current_state >= threshold_control) return control_distribution(gen);
        return 0;
    }

protected:
    /// Internal reference to the originating simulator
    const PopulationSim& sim;
    long threshold_control;
    prec_t prob_control;
    /// Random number engine
    default_random_engine gen;
    binomial_distribution<int> control_distribution;
};
}} // namespace craam::msen
