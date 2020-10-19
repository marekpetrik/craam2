#!/usr/bin/Rscript

# This scripts creates a riverswim-style problem dataset, samples data from it, 
# computes the posterior distribution, and then saves samples from this distribution
#  
# WARNING: This problem is inspired by the classical riverswim problem, but
#          its specific parameters are quite different.
# 
# The states in this problem are arranged as a chain with the index increasing
# from left to right.
# 
# The rewards in this problem depend on state, action, and the next state and
# are assumed to be known.
# 
# The prior assumptions in this model are:
#  - action a0 is known
#  - action a1 transitions only to 3 possible states (left, middle, and right)
#    and all transition probabilities are the same
#  - attempting to go over each end of the chain keeps the state unchanged
#  - rewards are assumed to be known
# 
# See the domains README.md file for details about the files that are created


# remove anything in the namespace just in case this being
# run repeatedly interactively
rm(list = ls())

library(rcraam)
library(dplyr)
library(readr)
library(rjags)

## ----- Parameters --------

# data output (platform independent construction)
folder_output <- file.path('domains', 'riverswim')  

# riverswim problem specification
state.count <- 20                 # number of states in the riverswim problem
init.dist <- rep(1/state.count, state.count)  # initial distribution p0
left.reward <- 5                  # reward whenever  taking action a0
prize.reward <- 100               # reward in the highest state (the risky reward)
probabilities.true <- c(0.2, 0.3, 0.5)  # true transition probabilities for action 1 (move: left, stay, right)
stopifnot(abs(1.0 - sum(probabilities.true)) < 1e-6)
discount <- 0.9                   # discount rate

# transition samples
sample.seed <- 1994               # reproducibility               
samples <- 15                     # number of transition samples per episode
episodes <- 1                     # number of episodes to sample from

# posterior samples
postsamples_train <- 100          # number of posterior training samples
postsamples_test <- 100           # number of posterior test samples
n_chains <- 4                     # number of MCMC chains
posterior.seeds <- c(2016, 5874, 12, 99)  # reproducibility, one seed per chain
stopifnot(n_chains == length(posterior.seeds))


## ------ Construct riversim-like MDP ------

#' Constructs a riverswim mdp. 
#' 
#' See the top of the file for a more-detailed description of this problem.
#' 
#' @param probabilities 3-values vector with probabilities
#'  of going left, staying put, going right when
#'  taking action a1
make_riverswim <- function(probabilities){

  # check the arguments for correctness
  stopifnot(length(probabilities) == 3)
  stopifnot(abs(1.0 - sum(probabilities)) < 1e-6)
  
  # safe action that just moves left
  a0left <- data.frame(idstatefrom = seq(0,state.count - 1), idaction = 0,
                       idstateto = pmax(0, seq(-1, state.count - 2)),
                       probability = 1,
                       reward = left.reward)
  
  # the risky action right (and an identifier where it moves)
  a1left <- data.frame(idstatefrom = seq(0,state.count - 1), 
                       idaction = 1,
                       idstateto = pmax(0, seq(-1, state.count - 2)), 
                       probability = probabilities[1],
                       reward = 0)
  
  a1middle <- data.frame(idstatefrom = seq(0, state.count - 1), 
                         idaction = 1,
                         idstateto = seq(0,state.count - 1), 
                         probability = probabilities[2],
                         reward = 0)
  
  a1right <- data.frame(idstatefrom = seq(0, state.count - 1), 
                        idaction = 1,
                        idstateto = pmin(seq(1, state.count), state.count - 1),
                        probability = probabilities[3], 
                        reward = 0)
  
  # assign the prize reward in the right-most state (when staying in the state)
  a1middle <- within(a1middle, {
    reward[idstatefrom == state.count - 1] <- prize.reward})
  a1right <- within(a1right, {
    reward[idstatefrom == state.count - 1] <- prize.reward})
  
  return(rbind(a0left, a1left, a1middle, a1right) %>% na.fail())
}

## ------- Construct the true MDP -------------

cat("Constructing true MDP ...\n")
mdp.true <- make_riverswim(probabilities.true)

# compute the true value function
sol.true <- solve_mdp(mdp.true, discount, show_progress = FALSE)
vf.true <- arrange(sol.true$valuefunction, idstate)$value 
cat("True optimal return", vf.true %*% init.dist, 
    "policy:", sol.true$policy$idaction, "\n")

## ----- Generate Samples --------------

cat("Generating transition samples ... \n")

# construct a randomized policy
ur.policy = data.frame(
  idstate = rep(seq(0, state.count-1), 2), 
  idaction = c(rep(0, state.count), rep(1, state.count)),
  probability = c(rep(0.5, state.count), rep(0.5, state.count)))

# generate samples from the swimmer domain
transition_samples <- simulate_mdp(mdp.true, 0, ur.policy, episodes = episodes, 
                                   horizon = samples, seed = sample.seed)

## ----- Fit a JAGS Model ----------------

#' Generate a sampled MDPO (from the posterior). 
#' 
#' The function assumes a Dirichlet prior
#' 
#' @param simulation  Simulation results
#' @param n_posterior Number of posterior samples to construct
mdpo_riverswim_bayes <- function(simulation, n_posterior){
  
  sampling_thin <- 10  # thinning for MCMC
  
  # restrict the samples
  # 1) select action a1 because that is the only uncertain action
  # 2) ignore the edge cases to not have to deal with going off the edge of the
  #    chain. 
  transitions <- simulation %>%
    filter(idstatefrom > 0, idstatefrom < state.count - 1, idaction == 1) %>%
    mutate(difference = idstateto - idstatefrom + 2)
  
  # make sure that difference is a 1-based and has 3 values at most
  stopifnot(max(transitions$difference) <= 3)
  stopifnot(min(transitions$difference) >= 1)
  
  # create random number seeds for all chains
  inits <- lapply(posterior.seeds, 
                  function(s){list(.RNG.seed = s, 
                                   .RNG.name = "base::Super-Duper")})
  
  # JAGS is an overkill reall here, but serves as a demonstration
  # the Dirichlet posterior can be computed analytically
  model_spec <- textConnection("
      model { for (i in 1:N){ difference[i] ~ dcat(beta) }
              beta ~ ddirch(dprior)}")
  jags <- jags.model(model_spec,
        data = list(difference = transitions$difference,
                    N = nrow(transitions),
                    dprior = c(1,1,1)),
               n.chains = n_chains, n.adapt = 100, 
               inits = inits,
               quiet = TRUE)
  # warmup
  update(jags, 100, progress.bar = "none", inits = inits) 
  # thin samples in order to decrease their correlation
  post_samples <- coda.samples(jags, c('beta'), 
    n.iter = n_posterior * sampling_thin, 
    thin = sampling_thin, progress.bar = "none", inits = inits)
  # combine samples from the individual chains
  raw_samples <- do.call(rbind, lapply(post_samples, as.matrix))
  
  trans.prob <- lapply(1:nrow(raw_samples), function(i){
    mdp <- make_riverswim(unname(raw_samples[i,]))
    mdp$idoutcome = i - 1
    return(mdp)
  })
  
  mdpo <- bind_rows(trans.prob) %>% na.fail()
  
  return(mdpo)
}

## ----- Generate MDPO from the posterior -----------

cat("Generating posterior samples (may take a minute) ... \n")
# generate posterior
mdpo <- mdpo_riverswim_bayes(transition_samples, postsamples_train + postsamples_test)

# make sure that all probabilities sum to 1 (select all s,a,o with that do not sum to 1)
invalid <- 
  mdpo %>% group_by(idstatefrom, idaction, idstateto, idoutcome) %>% 
    summarize(sumprob = sum(probability), .groups = "keep") %>%
    filter(sumprob > 1 + 1e-6, sumprob < 1 - 1e-6)

stopifnot(nrow(invalid) == 0)

# split into test and training sets (idoutcome is 0-based)
mdpo_train <- mdpo %>% filter(idoutcome < postsamples_train)
mdpo_test <- mdpo %>% filter(idoutcome >= postsamples_train) %>%
  mutate(idoutcome = idoutcome - postsamples_train ) 

## ------- Save results in the directory ------

cat("Writing results to ", folder_output, " .... \n")
if(!dir.exists(folder_output)) dir.create(folder_output, recursive = TRUE)

initial_df <- data.frame(idstate = seq(0,state.count - 1), probability = init.dist)
parameters_df <- data.frame(parameter = c("discount"), 
                            value = c(0.9))

write_csv(initial_df, file.path(folder_output, "initial.csv.xz"))
write_csv(parameters_df, file.path(folder_output, "parameters.csv"))
write_csv(mdp.true, file.path(folder_output, 'true.csv.xz'))
write_csv(mdpo_train, file.path(folder_output, 'training.csv.xz'))
write_csv(mdpo_test, file.path(folder_output, 'test.csv.xz'))

