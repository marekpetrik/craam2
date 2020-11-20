// [[Rcpp::plugins(cpp2a)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(rcraam)]]

#include <Rcpp.h>
#include <Rmath.h>
#include <vector>
#include <cmath>
#include <random>
#include <array>
#include <utility>
#include <algorithm>

#include "craam/simulation.hpp"

// Based on the python code


struct CancerState {
    // these are the initial values
    double C = 0;
    double P = 7.13;
    double Q = 41.2;
    double Q_p = 0;

    /// Default construct that builds the initial cancer state
    CancerState() = default;

    /// Copies the parameter values into the class
    CancerState(double C, double P, double Q, double Q_p) :
    C(C), P(P), Q(Q), Q_p(Q_p) {}

    /// Reads state from a list
    CancerState(Rcpp::List state){
        C = state["C"];
        P = state["P"];
        Q = state["Q"];
        Q_p = state["Q_p"];
    }

    /// Converts the state to a list
    operator Rcpp::List() const {
        return Rcpp::List::create(
            Rcpp::_["C"] = C,
            Rcpp::_["P"] = P,
            Rcpp::_["Q"] = Q,
            Rcpp::_["Q_p"] = Q_p
        );
    }
};

// Default patient configuration to match Elita's code
struct Config {
    double kde = 0.24;
    double lambda_p = 0.121;
    double k_qpp = 0.0031;
    double k_pq = 0.0295;
    double gamma = 0.729;
    double delta_qp = 0.00867;
    double k = 100;
    double dose_penalty = 1.0;
    double transition_noise = 2.0;

    Config() = default;

    Config(Rcpp::List conf){
        kde = conf["kde"];
        lambda_p = conf["lambda_p"];
        k_qpp = conf["k_qpp"];
        k_pq = conf["k_pq"];
        gamma = conf["gamma"];
        k = conf["k"];
        dose_penalty = conf["dose_penalty"];
        transition_noise = conf["transition_noise"];
    }
    
    operator Rcpp::List () const {
        return Rcpp::List::create(
            Rcpp::_["kde"] = kde,
            Rcpp::_["lambda_p"] = lambda_p,
            Rcpp::_["k_qpp"] = k_qpp,
            Rcpp::_["k_pq"] = k_pq,
            Rcpp::_["gamma"] = gamma,
            Rcpp::_["delta_qp"] = delta_qp,
            Rcpp::_["k"] = k,
            Rcpp::_["dose_penalty"] = dose_penalty,
            Rcpp::_["transition_noise"] = transition_noise
        );
    }
};


/// Computes the next transition and the reward
/// the noise should be generated from a normal distribution
std::pair<double, CancerState> 
next_state(const CancerState& state, bool action, const Config& config){

    const double P_star = state.P + state.Q + state.Q_p;

    std::array<float, 4> noise;
    for(size_t i = 0; i < 4; ++i){
        noise[i] = 1 + R::rnorm();
    }

    const double C = (1 - config.kde) * (state.C + action ? 1.0 : 0.0);

    const double P = (state.P + config.lambda_p * state.P * (1.0 - P_star / config.k) + config.k_qpp * state.Q_p
         - config.k_pq * state.P - config.gamma * C * config.kde * state.P);

    const double Q = state.Q + config.k_pq * P - config.gamma * C * config.kde * state.Q;
    
    const double Q_p = (state.Q_p + config.gamma * C * config.kde * Q - config.k_qpp * state.Q_p
           - config.delta_qp * state.Q_p);


    const double P_star_new = P + Q + Q_p;
    const double reward = (P_star - P_star_new) - config.dose_penalty * C;

    // adds the noise to the state transition
    return {reward, CancerState(C * noise[0], P * noise[1], Q * noise[2], Q_p * noise[3])};
}

// [[Rcpp::export]]
Rcpp::List default_config(){
    return Config();
}

// [[Rcpp::export]]
Rcpp::List init_state(){
    return CancerState();
}

// [[Rcpp::export]]
Rcpp::List cancer_transition(Rcpp::List state, bool action, Rcpp::List config){
    auto [reward, nextstate] = next_state(CancerState(state), action, Config(config));

    return Rcpp::List::create(
            Rcpp::_["state"] = static_cast<Rcpp::List>(nextstate),
            Rcpp::_["reward"] = reward
        );
}

/// Computes the distance between two state. 
/// Used when constructing representative states for aggregation
/// It is just euclidean distance for now.
double state_distance(const CancerSimulator& s1, const CancerSimulator& s2){
    using pow = std::pow;
    using sqrt = std::sqrt;

    return sqrt(pow(s1.C - s2.C, 2) + pow(s1.P - s2.P, 2) + pow(s1.Q - s2.Q, 2) +
                pow(s1.Q_p - s2.Q_p));
}

/// This is a dumb exhaustive search, replace by some KNN techniques for improved performance
size_t closest_state(const CancerState& state, const std::vector<CancerState> rep_states){

    std::vector<double> distances(rep_states.size());
#pragma omp parallel for
    for(size_t i = 0; i < rep_states.size(); ++i){
        distances[i] = state_distance(state, rep_states[i]);
    }
    return std::distance(distances.cbegin(), std::min_element(distances.cbegin(), distances.cend()));
}

/**
 * Simulator used to construct an MDP of the problem
 */
class CancerSimulator {
protected:
    Config config;                          ///< Configuration  of the simulator
    std::vector<CancerState> rep_states;    ///< Representative states

public:
    using State = uint_t;
    using Action = uint_t;

    CancerSimulator(Config config, std::vector<CancerState> states) :
        config(config), normal_distribution(0,1), rep_states(rep_states) {}


    /// Returns a sample of the reward and a decision state following
    /// an expectation state
    std::pair<double, State> transition(State state, Action action) const{
        if(action > 1) throw std::runtime_error(unknown action);

        const CancerState& cstate = rep_states.at(state);

        const auto [reward, cnextstate] = next_state(cstate, bool(action), config);
        const State next_state = closest_state(cstate, rep_states);
        return {reward, next_state};
    }

    /// Checks whether the decision state is terminal
    bool end_condition(State s) const {
        return false;
    }

    /// State dependent actions, with discrete number of actions
    /// (long id each action)
    /// use -1 if infinite number of actions are available
    Action action_count(State state) const {
        return 2;
    }

    /// State dependent action with the given index
    Action action(State state, uint_t index) const {
        return index;
    }

    /// Returns the number of valid states
    /// makes sense only with 0-based state index
    long state_count() const {
        return rep_states.size();
    }
}

std::vector<CancerState> parse_states(const Rcpp::List& rep_states){
    craam::numvec 
        vC = rep_states["C"], vQ = rep_states["Q"], 
        vP = rep_states["P"], vQ_p = rep_states["Q_p"];

    vector<CancerState> states;
    states.reserve(vC.size());

    for(size_t i = 0; i < states.size(); ++i){
        states.push_back(CancerState(vC[i], vP[i], vQ[i], vQ_p[i]));
    }
    return states;
}


/// Constructs an MDP and a mapping from state indexes to state values
/// @param rep_states Representative states (centers of aggregate states)
/// @param state_samplecount How many samples to get from each state
// [[Rcpp::export]]
Rcpp::List cancer_mdp(Rcpp::List config, Rcpp:DataFrame rep_states, size_t state_samplecount) {
    
    const CancerSimulator simulator(config, parse_states(rep_states));

    const craam::MDP mdp = craam::msen::buildmdp(simulator, state_samplecount);

    return mdp_to_dataframe(mdp);
}
