library(Rcpp)
library(rcraam)
library(ggplot2)
library(dplyr)

state_count <- 5000

sourceCpp("cancer_sim.cpp")

def_config <- default_config()

state <- init_state()

#print(cancer_transition(state, FALSE,  def_config))
#print(cancer_transition(state, TRUE, def_config))

# *** generate a bunch of random states (to serve as centers of the aggregate states) ***

# *** construct the MDP of the simulator

cat("Gathering samples ... \n")
samples_all <- simulate_random(def_config, state_count, 50)
samples_rep <- sample_n(samples_all$states_from, state_count)

# assumption: the first sample is the initial state
init_state <- class::knn1(samples_rep, samples_all$states_from[1,], 1:nrow(samples_rep))
# note: the state indexes are 1-based because state 0 is assumed to be the terminal state

cat("Building the MDP ... \n")
mdp <- cancer_mdp(def_config, samples_rep, 500, TRUE)
cat("MDP building complete. ")
sol <- solve_mdp(mdp, 0.8)

ret_value <- sol$valuefunction %>% filter(idstate == init_state)

cat("Computed return: ", ret_value$value)




