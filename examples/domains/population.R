#!/usr/bin/Rscript

# Version 1.0
# 
# This script creates a dataset for the pest population control problem. The goal
# is to choose one of several pesticides to control a pest based on its current 
# size. 
#
# The posterior is estimated using a JAGS models. 
#
# Known limitations:
#   - JAGS model is currently used to only estimate the mean effectiveness of the pesticide.
#     The standard deviation of the effectiveness is assumed to be known
#   - Because of the dependence on JAGS, seeds may be insufficient to guarantee exact
#     reproducibility if any version of the packages changes
#   - JAGS inference currently runs using only a single core
# 
# Other considerations:    
#   - The JAGS program models an exponential population growth, while the baseline model 
#     that of logistic growth. These two models are similar long as the pest populations remain
#     low. Also, this discrepancy is irrelevant from the view of the posterior evaluation.
#     It means, though, that the prior is mis-specified (which is likely to happen in practice too)
#
# See the domains README.md file for details about the files that are created

rm(list = ls())  # clear the workspace to prevent bugs in interactive runs

cat("**** Requires pixz to compress output data ****\n")

suppressPackageStartupMessages({
  library(rcraam)
  library(dplyr)
  library(setRNG)
  library(readr)
  library(rjags)
  library(runjags)
  library(progress)
  library(parallel)})

## ----- Problem Specification Parameters -------------

# data output (platform-independent construction)
folder_output <- file.path('domains', 'population')  

# general parameters
discount <- 0.9

# data generation parameters (data used to compute posterior)
episodes <- 1
steps_per_episode <- 300

# overall population model specification
max.population <- 50
init.population <- 10
actions <- 5
external.pop <- 3

# transition probability parameters
#    general effectiveness function
corr <- (seq(0,max.population) - (max.population/2))^2 
#    relative effectiveness of the pesticides 
growth.app <- 0.2 + corr / max(corr) 
#    first action: no control, second action: pesticide
#    rows: actions, columns: population level (must match the specs above)
exp.growth.rate <- rbind(rep(2.0, max.population + 1), 
                         growth.app, growth.app, growth.app, growth.app)
stopifnot( all(dim(exp.growth.rate) == c(actions, max.population + 1)) )
#    standard deviations of individual actions
#    rows: actions, colums: population level (must match the specs above)
sd.growth.rate <- rbind(rep(0.6, max.population + 1), rep(0.6, max.population + 1), 
                        rep(0.5, max.population + 1), rep(0.4, max.population + 1),
                        rep(0.3, max.population + 1))
stopifnot( all(dim(sd.growth.rate) == c(actions, max.population + 1)) )

# reward parameters
#      rewards decrease with increasing population, and there is an extra penalty
#      for applying the pesticide
rewards <- matrix(rep(-seq(0,max.population)^2, actions), nrow = actions, byrow = TRUE)
rewards <- rewards + 1000 # add harvest return
spray.cost <- 800
rewards[2,] <- rewards[2,] - spray.cost
rewards[3,] <- rewards[3,] - spray.cost * 1.05
rewards[4,] <- rewards[4,] - spray.cost * 1.10
rewards[5,] <- rewards[5,] - spray.cost * 1.15

# posterior sample specification
postsamples_train <- 1000          # training samples
postsamples_test <- 1000           # test samples
postsamples <- postsamples_train + postsamples_test # total number of samples

# specific JAGS sampling parameters
n_chains <- 8                     # number of chains
n_skip <- 100                     # number of MCMC samples to skip when thinning
n_adapt <- 5000                   # number of samples to gather to tune the model
n_update <- 20000                 # samples to converge to limiting distribution

# set seeds for reproducibility
simulation_seed <- 1
posterior_seeds <- c(2365, 1, 25, 458, 965, 147, 78, 457)
# make sure one seed per chain
stopifnot(n_chains == length(posterior_seeds) )


## ------ Construct the true MDP model ------------
pop.model.mdp <- 
  rcraam::mdp_population(max.population, init.population, 
                        exp.growth.rate, sd.growth.rate, 
                        rewards, external.pop, external.pop/2, "logistic")

## ----- Sample from an optimal policy -------------

cat("Generating and solving true MDP ...\n")
# solve for the optimal policy
mdp_sol <- solve_mdp(pop.model.mdp, discount, show_progress = FALSE)
cat("MDP policy:", mdp_sol$policy$idaction, "\n")

# simulate the model
rpolicy <- mutate(mdp_sol$policy, probability = 1.0) 

cat("Sampling data from the optimal policy ... \n")
sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 
                            steps_per_episode, episodes, seed = simulation_seed)

## ------ Fit a model to the solution ---------------------

cat("Building JAGS model .... \n")
# define JAGS model
# N - number of data points
# M - number of actions
model_spec <- textConnection(
   "model {
      for (i in 1:N){
          ext[i] ~ dnorm(ext_mu, ext_std)
          next_mean[i] <- ifelse(act[i] == 0, mu * pop[i],
                        mu0 * pop[i] + mu1 * pop[i]^2 + mu2 * pop[i]^3)
          pop_next[i] ~ dnorm(next_mean[i] + ext[i], sigma[act[i] + 1]) }
      for (j in 1:M){ sigma[j] ~ dunif(0,5) }
      mu ~ dunif(0.5, 3)
      mu0 ~ dnorm(1.0, 1)
      mu1 ~ dnorm(0.0, 10)
      mu2 ~ dnorm(0.0, 10)
      ext_mu ~ dunif(0, 20)
      ext_std ~ dunif(0, 5) }")

# create random number seeds for all chains
inits <- lapply(posterior_seeds, function(s){
  list(.RNG.seed = s, .RNG.name = "base::Super-Duper")})

# TODO should parallelize the JAGS inference 

cat("Compiling JAGS model ...\n")
# inits parameter serves to make sure that the results are reproducible
pop.jags <- jags.model(model_spec, data = list(
                            pop = sim.samples$idstatefrom,
                            pop_next = sim.samples$idstateto , 
                            act = sim.samples$idaction,
                            N = nrow(sim.samples), M = nrow(exp.growth.rate)),
                   n.chains = n_chains, n.adapt = n_adapt, inits = inits)
cat("Warmup samples ...\n")
# warmup
update(pop.jags, n_update)

cat("Gathering samples ...\n")
# QUESTION: Is thinning helpful here? This is different from 
# uses in which we only care about the statistics. The goal is to 
# de-correlate samples
post_samples <- jags.samples(pop.jags, c('mu', 'mu0',  'mu1', 'mu2'), 
                             n.iter = round(n_skip / n_chains * postsamples), 
                             thin = n_skip, inits = inits)
    
### ------ Formulate Bayesian MDP ---------------------

cat("Constructing Bayesian MDP samples ....\n")

mdp.bayesian <- NULL         # Bayesian MDP data.frame

# generate possible states
population_range <- seq(0, max.population)

# construct a progress bar
# compute the number of samples that need to be processed
d <- dim(post_samples$mu)
pb <- progress_bar$new(format = "(:spin) [:bar] :percent", 
                       total = d[2] * d[3])

# constructs an MDP for the given chain and iteration of the chain
make_mdp <- function(chain_iteration){
  chain <- chain_iteration[1]
  iteration <- chain_iteration[2]
  #cat("chain", chain, "iteration", iteration, "\n")
  pb$tick()
  # create an empty matrix of rates
  rates <- matrix(0, ncol = ncol(exp.growth.rate), nrow = dim(exp.growth.rate))
  # fill in rate parameters
  rates[1,] <- post_samples$mu[1,iteration,chain]
  rates[2:dim(rates)[1],] <-
    post_samples$mu0[1,iteration,chain] + 
    post_samples$mu1[1,iteration,chain] * population_range + 
    post_samples$mu2[1,iteration,chain] * population_range^2;
  # construct the MDP
  rcraam::mdp_population(max.population, init.population,
                         rates, sd.growth.rate, rewards, 
                         external.pop, external.pop/2, "exponential")
}


# generate all possible values for chains and iterations
sample_parameters <- expand.grid(chain = 1:d[3], iteration = 1:d[2])

cluster <- makeCluster(detectCores())    # initialize parallel computation
clusterExport(cluster, varlist = c("pb", "exp.growth.rate", "post_samples",
                                   "population_range","max.population",
                                   "init.population","sd.growth.rate", 
                                   "rewards", "external.pop"))
bayes_MDPs <- parApply(cluster, sample_parameters, 1, make_mdp)
stopCluster(cluster)
pb$terminate()

idoutcome <- 0               # index of the posterior sample
# add idoutcome to all MDPs
for (l in seq_along(bayes_MDPs)){
    bayes_MDPs[[l]]$idoutcome <- idoutcome
    idoutcome <- idoutcome + 1
}
mdpo <- bind_rows(bayes_MDPs)

## --- Generate and save the output files ----

cat("Writing results to ", folder_output, " .... \n")
if(!dir.exists(folder_output)) dir.create(folder_output, recursive = TRUE)
 
# split into test and training sets (idoutcome is 0-based)
mdpo_train <- mdpo %>% filter(idoutcome < postsamples_train)
mdpo_test <- mdpo %>% filter(idoutcome >= postsamples_train) %>%
  mutate(idoutcome = idoutcome - postsamples_train ) 


# construct initial distribution
initial_dist <- rep(0, max.population + 1)
initial_dist[init.population] <- 1.0
initial_df = data.frame(idstate = population_range, initial_dist)

# parameters
parameters_df <- data.frame(parameter = c("discount"), value = c(0.9))

# save all the files
write_csv(initial_df, file.path(folder_output, "initial.csv.xz"))
write_csv(parameters_df, file.path(folder_output, "parameters.csv"))
write_csv(pop.model.mdp, file.path(folder_output, 'true.csv.xz'))
write_csv(mdpo_train, file.path(folder_output, 'training.csv'))
# compression using parallel xz
cat("  compressing training ... \n")
system(paste("pixz", file.path(folder_output, 'training.csv')))
write_csv(mdpo_test, file.path(folder_output, 'test.csv'))
# compression using parallel xz
cat("  compressing test ... \n")
system(paste("pixz", file.path(folder_output, 'test.csv')))
