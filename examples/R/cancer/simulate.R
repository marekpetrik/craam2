library(Rcpp)
library(rcraam)
library(ggplot2)
library(dplyr)
library(tidyr)

rm(list=ls())

theme_set(theme_light())
sourceCpp("cancer_sim.cpp")

# --- Params ----
state_count <- 1000
discount <- 0.9
sim_runs <- 1000

training_count <- 100
test_count <- 200


def_config <- default_config()
def_config$noise <- "gamma"
def_config$noise_std <- 0.2
def_config$dose_penalty <- 1.6

# --- Initialize -----

state <- init_state()

#print(cancer_transition(state, FALSE,  def_config))
#print(cancer_transition(state, TRUE, def_config))

# *** generate a bunch of random states (to serve as centers of the aggregate states) ***

cat("Gathering samples ... \n")
samples_all <- simulate_random(def_config, state_count, 200)

#plot distributions from random policy

plot_density <- function() {
    ggplot(pivot_longer(samples_all$states_from, c(C,P,Q,Q_p), "predictor"), 
           aes(x = value)) + stat_density() + facet_wrap(vars(predictor)) +
            lims( x = c(-10,10))
}

samples_rep <- sample_n(samples_all$states_from, state_count)
# NOTE!: the state indexes are 1-based because state 0 is assumed to be the terminal state
samples_rep_state <- mutate(samples_rep, idstate = row_number())


# assumption: the first sample is the initial state
# note: the state indexes are 1-based because state 0 is assumed to be the terminal state
init_state_num <- as.integer(class::knn1(samples_rep, samples_all$states_from[1,], 1:nrow(samples_rep)))

# *** construct the MDP of the simulator

cat("Building the MDP ... \n")
mdp <- cancer_mdp(def_config, samples_rep, 1000, TRUE)
cat("MDP building complete. \n")
max_state <- max(max(mdp$idstateto), max(mdp$idstatefrom))

cat("*******\n")
cat("\nPredicted return: \n")
sol <- solve_mdp(mdp, discount, show_progress = 0)
# sol <- rsolve_mdp_sa(mdp, discount, nature = "l1u", nature_par = 0.05)
ret_value <- sol$valuefunction %>% filter(idstate == init_state_num)
cat("Optimized: ", ret_value$value, "\n")

policy_never <- data.frame(idstate = seq(0, max_state), idaction = 0)
sol_never <- solve_mdp(mdp, discount, show_progress = FALSE, policy_fixed = policy_never)
cat("Never: ", (sol_never$valuefunction %>% filter(idstate == init_state_num))$value, "\n" )

policy_always <- data.frame(idstate = seq(0, max_state), idaction = 1)
sol_always <- solve_mdp(mdp, discount, show_progress = FALSE, policy_fixed = policy_always)
cat("Always: ", (sol_always$valuefunction %>% filter(idstate == init_state_num))$value, "\n" )
cat("*******\n\n")

# **** Simulate the return *****

# join state features and actions and strips the terminal state idstate = 0
state_policy <- inner_join(sol$policy, samples_rep_state, by = "idstate") %>%
    mutate(idaction = idaction) %>% filter(idaction >= 0)

# simulates the policy
simulated_policy <- simulate_proximity(def_config, select(state_policy, C, P, Q, Q_p), 
                   as.integer(state_policy$idaction), sim_runs, 200)

simulated_random <- simulate_random(def_config, sim_runs, 200)
simulated_never <- simulate_trivial(def_config, 0, sim_runs, 200)
simulated_always <- simulate_trivial(def_config, 1, sim_runs, 200)

compute_return <- function(simulation){
    sum(simulation$rewards * discount^simulation$steps) / simulation$runs
}

cat("*******\n")
cat("Simulated returns:\n")
cat("Optimized: ", compute_return(simulated_policy), "\n")
cat("Random: ", compute_return(simulated_random), "\n")
cat("Never: ", compute_return(simulated_never), "\n")
cat("Always: ", compute_return(simulated_always), "\n")
cat("*******\n\n")

library(C50)
library(rpart)
library(rpart.plot)

rpart_c <- rpart(idaction ~ C + P + Q + Q_p, data = mutate(state_policy, idaction = as.factor(idaction)))
#print(tidyrules::tidyRules(rpart_c))
prp(rpart_c)

# ---- Generate simulated data -------

sim_data <- with(simulated_policy, {
  cbind(
      states_from %>% rename_all(function(x){paste0("from_",x)}),
      states_to %>% rename_all(function(x){paste0("to_",x)}),
      idaction = actions)
})

# ---- Fit linear regression --------

cat("*******\n")
cat("Linear model fitting statistics: \n")

targets <- select(sim_data, starts_with("to_")) %>% colnames()

for (idaction_fit in c(0,1)) {
    sim_data2 <- filter(sim_data, idaction == idaction_fit)
    for (t in targets) {
        lr <- lm(get(t) ~ from_C + from_P + from_Q + from_Q_p, data = sim_data2)
        cat("Action", idaction_fit, "Target", t, "R2:", summary(lr)$r.squared, "\n")    
    }
}
cat("*******\n\n")

# ------ Fit a Stan model --------

cat("*******\n")
cat("Stan results:\n")

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data_limit <- 200 # the max number of datapoints to use from the simulation

# the model that we fit does not assume that the noise is the same for 
# each dimension of the problem

lin_params <- list()
# lin_params contains the weights for the linear model for each 
# target and action. 
# The weights are fitted in the following order (see model.matrix below):
# "(Intercept)" "from_C"      "from_P"      "from_Q"      "from_Q_p" 
gamma_params <- list()
for (t in targets) {
    lin_params[[t]] <- list()
    gamma_params[[t]] <- list()
    
    for (idaction_t in c(0,1)) {
        # only select data for the appropriate action
        # and where the target value is sufficiently far from 0
        # and is non-negative
        sim_subset <- sim_data %>% 
            filter(idaction == idaction_t, get(t) > 0.1) %>%
            head(data_limit)
        
        X <- model.matrix(get(t) ~ from_C + from_P + from_Q + from_Q_p,
                          data = sim_subset)
        y <- sim_subset[[t]]
        
        standata <- list( N = nrow(X), K = ncol(X), X = X, y = y)
        
        fit <- stan(file = 'fit_gamma.stan', data = standata, chains = 1, iter = 2000)
        
        # alpha is both shape and rate (they are assumed to be the same)
        # of the gamma distribution
        alpha_samples <- extract(fit, "alpha")
        w_samples <- extract(fit, "w")
        
        # the gamma standard deviation is sqrt(1/alpha)
        cat("Action", idaction_t, "Target", t, "\n")
        cat("True gamma sd:", def_config$noise_std, 
            "predicted gamma sd:", mean(sqrt(1/alpha_samples$alpha)), "\n" )
        
        
        lin_params[[t]][[toString(idaction_t)]] <- w_samples
        gamma_params[[t]][toString(idaction_t)] <- alpha_samples
    }    
}

cat("*******\n")

# ------ Construct linear models ---------

# states: samples_rep_state
# initial states: index is in init_state_number


count_next_state <- 20

# constructs an MDP from a single sample
# of the linear model definition
# 
# Depends on many global variables
# 
# @param index index of the sample that should be used for each model
sample_to_mdp <- function(index){
    
    # expected next value for each state
    # replicate for the two available actions
    next_deterministic <- rbind(samples_rep_state %>% mutate(idaction = 0), 
                        samples_rep_state %>% mutate(idaction = 1))
    
    # NOTE: assumes that the order of the actions is as constructed above
    # this is a little bit dangerous, but convenient
    # be careful when changing things
    for (t in targets) {
        next_deterministic[t] <- 
            c(drop(model.matrix(~C + P + Q + Q_p, data = samples_rep_state) %*% 
                       lin_params[[t]][["0"]]$w[index,]),
              drop(model.matrix(~C + P + Q + Q_p, data = samples_rep_state) %*% 
                       lin_params[[t]][["1"]]$w[index,]))
    }
    
    # Generate the random distribution over next states
    # do this just by sampling from the posterior instead of using the 
    # gamma probabilities. Computing the density for 4 independent
    # gamma distribution would be a bit too hairy
    next_stoch_all <- list()
    for (next_index in 1:count_next_state){
        next_stochastic <- next_deterministic
        for (t in targets) {
            alpha0 <- gamma_params[[t]][["0"]][index]
            alpha1 <- gamma_params[[t]][["1"]][index]
            
            # again an assumption on the same number of rows per action
            next_stochastic[t] <- next_stochastic[t] * 
                c(rgamma(nrow(next_deterministic)/2, shape = alpha0, rate = alpha0),
                  rgamma(nrow(next_deterministic)/2, shape = alpha1, rate = alpha1))    
            next_stoch_all[[next_index]] <- next_stochastic %>% mutate(next_index = next_index)
        }
    }
    next_stoch_all <- bind_rows(next_stoch_all)
    
    # generate the next state distribution
    # first find the next state indexes
    next_stoch_all$idstateto <- 
        class::knn1(select(samples_rep_state, -idstate), 
                    next_stoch_all %>% 
                        select(starts_with("to_")) %>% 
                        rename_with(function(x){sub("to_", "", x)}), 
                    samples_rep_state$idstate) %>%
            as.integer()
    
    # IMPORTANT: This should match the definition in the simulator
    next_stoch_all <- mutate(next_stoch_all, reward = 
                             to_P + to_Q + to_Q_p - P - Q - Q_p - def_config$dose_penalty * C)
        
    
    next_stoch_all <- rename(next_stoch_all, idstatefrom = idstate) %>%
        select(idstatefrom, idaction, idstateto, reward, next_index)
    
    mdp <- next_stoch_all %>% group_by(idstatefrom, idaction, idstateto) %>%
        summarize(probability = n() / count_next_state, 
                  reward = sum(reward) / count_next_state, 
                  .groups = "drop")
    
    return(mdp)
}


mdps <- list()
for(k in 1:(training_count, test_count)){
    mdps[[k]] <- sample_to_mdp(k) %>% mutate(idoutcome = k)
}






