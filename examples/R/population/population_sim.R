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
# 
# Other considerations:    
#   - The JAGS program models an exponential population growth, while the baseline model 
#     that of logistic growth. These two models are similar long as the pest populations remain
#     low. Also, this discrepancy is irrelevant from the view of the posterior evaluation.
#     It means, though, that the prior is mis-specified (which is likely to happen in practice too)
#
# See the domains README.md file for details about the files that are created

rm(list = ls())  # clear the workspace to prevent bugs in interactive runs

library(rcraam)
library(dplyr)
library(setRNG)
library(readr)
library(rjags)
library(runjags)

## ----- Problem Specification Parameters -------------

# data output (platform-independent construction)
folder_output <- file.path('domains', 'population')  

# general parameters
discount <- 0.9

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
#    rows: actions, colums: population level (must match the specs above)
exp.growth.rate <- rbind(rep(2.0, max.population + 1), 
                         growth.app, growth.app, growth.app, growth.app)
stopifnot( all(dim(exp.growth.rate) == c(actions, max.population + 1)) )
#    standard deviations of individual actions
#    rows: actions, colums: population level (must match the specs above)
sd.growth.rate <- rbind(rep(0.6, max.population + 1), rep(0.6, max.population + 1), 
                        rep(0.5, max.population + 1), rep(0.4, max.population + 1),
                        rep(0.3, max.population + 1))
stopifnot( all(dim(std.growth.rate) == c(actions, max.population + 1)) )

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
postsamples_train <- 1000
postsamples_test <- 1000

seed <- 1
set.seed(seed)


## ------ Construct the true MDP model ------------
pop.model.mdp <- rcraam::mdp_population(max.population, init.population, 
                                        exp.growth.rate, sd.growth.rate, 
                                        rewards, external.pop, external.pop/2, 
                                        "logistic")

## ----- Sample from an optimal policy -------------

# solve for the optimal policy
mdp_sol <- solve_mdp(pop.model.mdp, discount,show_progress=FALSE)
cat("MDP policy:", mdp_sol$policy$idaction, "\n")

# simulate the model
rpolicy <- mutate(mdp_sol$policy, probability = 1.0) # needs a randomized policy for now

set.seed(1)
sim.samples <- simulate_mdp(pop.model.mdp,0,rpolicy,100,5,seed=seed)

## ------ Fit a model to the solution ---------------------


set.seed(1)
sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1,seed=seed)

pop <- sim.samples$idstatefrom
pop.next <- sim.samples$idstateto
act <- sim.samples$idaction

generate_posterior <- function(){
    model_str <- 
    "
    model {
      for (i in 1:N){
        ext[i] ~ dnorm(ext_mu, ext_std)
        next_mean[i] <- ifelse(act[i] == 0, 
                      mu * pop[i],
                      mu0 * pop[i] + mu1 * pop[i]^2 + mu2 * pop[i]^3)
        pop_next[i] ~ dnorm(next_mean[i] + ext[i], sigma[act[i] + 1])
      }
      for (j in 1:M){
        sigma[j] ~ dunif(0,5)
      }
      mu ~ dunif(0.5, 3)
      mu0 ~ dnorm(1.0, 1)
      mu1 ~ dnorm(0.0, 10)
      mu2 ~ dnorm(0.0, 10)
      ext_mu ~ dunif(0, 20)
      ext_std ~ dunif(0, 5) 
    }
    "
    model_spec <- textConnection(model_str)
    pop.jags <- jags.model(model_spec,
                       data = list(pop = pop, pop_next = pop.next, act = act,
                                   N = length(pop), M = dim(exp.growth.rate)[1]),
                       n.chains=8,
                       n.adapt=5000)
    
    # warmup
    update(pop.jags, 20000)
    set.seed(seed)
    # QUESTION: Is thinning helpful here? This is different from 
    # uses in which we only care about the statistics
    post_samples <- jags.samples(pop.jags, c('mu', 'mu0',  'mu1', 'mu2'), round(100/8*post_sample_count), 100)
    #post_samples <- autorun.jags(jags)
    #set.seed(seed)
    saveRDS(post_samples, file="postsamples.rds")
    
    ### ------ Plot the true vs estimated effectiveness --------
    
    population_range <- seq(0,max.population)
    efficiency_true <- data.frame(population = population_range, type = "true", 
                           rate = growth.app)
    
    # returns the function that estimates the pest growth 
    # after action1 is applied
    efficiency_sampled <- matrix(0, nrow = dim(post_samples$mu0)[2] * dim(post_samples$mu0)[3],
                                    ncol = length(population_range))
    k <- 1
    for(i in 1:dim(post_samples$mu0)[2]){
      for(j in 1:dim(post_samples$mu0)[3]){
        efficiency_sampled[k,] <- post_samples$mu0[1,i,j] + 
                post_samples$mu1[1,i,j] * population_range   + 
                post_samples$mu2[1,i,j] * population_range^2
        k <- k + 1
      }
    }
    
    efficiency_sampled.df <- reshape2::melt(efficiency_sampled, value.name = "Rate")
    colnames(efficiency_sampled.df) <- c("Sample", "Population", "Rate")
    efficiency_sampled.df$Population <- efficiency_sampled.df$Population - 1
    
    # density plot
    ggplot(efficiency_sampled.df, aes(x = Population, y = Rate)) + 
            geom_hex() + 
            geom_line(mapping=aes(x=population, y=rate), data=efficiency_true, color = "red")
    
    # plot example responses
    ggplot(efficiency_sampled.df %>% filter(Sample %% 10 == 0), 
                   aes(x = Population, y = Rate, group = Sample)) + 
      geom_line(linetype = 3) +
      geom_line(mapping=aes(x=population, y=rate), 
                data=efficiency_true, color = "red", 
                inherit.aes = FALSE)
}

### ------ Plot the true vs estimated effectiveness --------

population_range <- seq(0,max.population)
efficiency_true <- data.frame(population = population_range, type = "true", 
                       rate = growth.app)

# returns the function that estimates the pest growth 
# after action1 is applied
efficiency_sampled <- matrix(0, nrow = dim(post_samples$mu0)[2] * dim(post_samples$mu0)[3],
                                ncol = length(population_range))

### ------ Formulate Bayesian MDP ---------------------

generate_posterior <- function(){
    mdp.bayesian <- NULL
    idoutcome <- 0
    # NOTE: the code assumes that the standard deviation of the actions is known
    for (iteration in 1:dim(post_samples$mu)[2]) { 
        cat(".")
        # create an empty matrix of rates
        rates <- matrix(0, ncol = dim(exp.growth.rate)[2],
                           nrow = dim(exp.growth.rate)[1])
        # TODO: enable this one
        for (chain in 1:dim(post_samples$mu)[3]) {     # loop over chains 
            rates[1,] <- post_samples$mu[1,iteration,chain]
    		    rates[2:dim(rates)[1],] <-
    		                 post_samples$mu0[1,iteration,chain] + 
                		     post_samples$mu1[1,iteration,chain] * population_range + 
                		     post_samples$mu2[1,iteration,chain] * population_range^2;
    		    
            pop.model.mdp <- rcraam::mdp_population(max.population, init.population,
                                                  rates, sd.growth.rate, rewards, 
                                                  external.pop, external.pop/2, "exponential")
            pop.model.mdp['idoutcome'] <- idoutcome
            mdp.bayesian <- rbind(mdp.bayesian,pop.model.mdp)
            idoutcome <- idoutcome + 1
        }
    }
    write_csv(mdp.bayesian, "population_bayes_model.csv")
}





