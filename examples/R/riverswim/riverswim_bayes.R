library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)
loadNamespace("tidyr")
loadNamespace("reshape2")

## ----- Parameters --------

init.dist <- rep(1/6,6)
discount <- 0.9
confidence <- 0.8
bayes.samples <- 500

samples <- 200
sample.seed <- 2019
episodes <- 1

## ----- Initialization ------

mdp.truth <- read_csv("riverswim2_mdp.csv", 
                      col_types = cols(idstatefrom = 'i',
                                       idaction = 'i',
                                       idstateto = 'i',
                                       probability = 'd',
                                       reward = 'd'))
rewards.truth <- mdp.truth %>% select(-probability)


# construct a biased policy to prefer going right
# this is to ensure that the "goal" state is sampled
ur.policy = data.frame(idstate = c(seq(0,5), seq(0,5)), 
                     idaction = c(rep(0,6), rep(1,6)),
                     probability = c(rep(0.2, 6), rep(0.8, 6)))

# compute the true value function
sol.true <- solve_mdp(mdp.truth, discount, show_progress = FALSE)
vf.true <- sol.true$valuefunction$value
cat("True optimal return", vf.true %*% init.dist, "policy:", sol.true$policy$idaction, "\n")

## ----- Generate Samples --------------

# generate samples from the swimmer domain
simulation <- simulate_mdp(mdp.truth, 0, ur.policy, episodes = episodes, 
                           horizon = samples, seed = sample.seed)

## ---- Solve Empirical MDP ---------
  
mdp.empirical <- mdp_from_samples(simulation)
sol.empirical <- solve_mdp(mdp.empirical, discount, show_progress = FALSE)
vf.empirical <- sol.empirical$valuefunction$value
cat("Empirical solution estimate: ", vf.empirical %*% init.dist, "policy:", sol.empirical$policy$idaction,"\n")


## ----  Uninformative Bayesian Posterior Sampling ---------


#' Generate a sample MDP from dirichlet distribution
#' @param simulation Simulation results
#' @param rewards.df Rewards for each idstatefrom, idaction, idstateto
#' @param outcomes Number of outcomes to generate
mdpo_bayes <- function(simulation, rewards.df, outcomes){
  priors <- rewards.df %>% select(-reward) %>% unique() 
  
  # compute sampled state and action counts
  # add a uniform sample of each state and action to work as the dirichlet prior
  sas_post_counts <- simulation %>% 
    select(idstatefrom, idaction, idstateto) %>%
    rbind(priors) %>%
    group_by(idstatefrom, idaction, idstateto) %>% 
    summarize(count = n()) 
  
  # construct dirichlet posteriors
  posteriors <- sas_post_counts %>% 
    group_by(idstatefrom, idaction) %>% 
    arrange(idstateto) %>% 
    summarize(posterior = list(count), idstatesto = list(idstateto)) 
  
  # draw a dirichlet sample
  trans.prob <- 
    mapply(function(idstatefrom, idaction, posterior, idstatesto){
      samples <- do.call(function(x) {rdirichlet(outcomes,x)}, list(posterior) )
      # make sure that the dimensions are named correctly
      dimnames(samples) <- list(seq(0, outcomes-1), idstatesto)
      reshape2::melt(samples, varnames=c('idoutcome', 'idstateto'), 
                     value.name = "probability" ) %>%
        mutate(idstatefrom = idstatefrom, idaction = idaction)
    },
    posteriors$idstatefrom,
    posteriors$idaction,
    posteriors$posterior,
    posteriors$idstatesto,
    SIMPLIFY = FALSE)
  
  mdpo <- bind_rows(trans.prob) %>% 
    full_join(rewards.df, 
              by = c('idstatefrom', 'idaction','idstateto')) %>%
    na.fail()
  return(mdpo)
}

mdp.bayesian <- mdpo_bayes(simulation, rewards.truth, bayes.samples)

#sol.bayesexp <- rsolve_mdpo_sa(mdp.bayesian, discount, "exp", NULL, show_progress = FALSE)
#cat("Expected Bayesian Solution: ", sol.bayesexp$valuefunction$value %*% init.dist, "policy:", sol.bayes.exp$policy$idaction, "\n")


## ----- Frequentist Confidence Interval ------

#' Constructs an MDP and a confidence interval
#' See e.g. Russel 2019
#' Confidence is: sqrt{ 1 / n_{s,a}  log (S A 2^S / \delta) }
#' where delta is the confidence
#' 
#' Assumes that only transitions that have rewards associates with them are possible
#' @param simulation Samples from the simulations
#' @param rewards.df Like an MDP description, just with the rewards only 
#'                   and no probabilites. Used to determine rewards and 
#'                   which transitions are possible (and considerd by the robust sol)
rmdp.frequentist <- function(simulation, rewards.df){
  mdp.nominal <- mdp_from_samples(simulation)
  
  # count the number of samples for each state and action
  sa_counts <- simulation %>% select(idstatefrom, idaction) %>%
    group_by(idstatefrom, idaction) %>%
    summarize(count = n())

  sa.count <- nrow(sa_counts)                   # number of valid state-action pairs
  # count the number of possible transitions from each state and action
  tran.count <- rewards.truth %>% 
    group_by(idstatefrom, idaction) %>% 
    summarize(tran_count = n())
  
  budgets <- full_join(sa_counts, tran.count, by = c('idstatefrom','idaction')) %>% 
    mutate(budget = pmin(2,sqrt(1/count * log(sa.count * 2^tran_count / confidence)))) %>%
    rename(idstate = idstatefrom) %>% select(-count) 
  
  mdp.nominal <- full_join(mdp.nominal %>% select(-reward), 
                           rewards.df, 
                           by = c('idstatefrom', 'idaction', 'idstateto')) %>% 
    tidyr::replace_na(list(probability = 0))
  return (list(mdp.nominal = mdp.nominal, budgets = budgets))
}

model.freq <- rmdp.frequentist(simulation, rewards.truth)
sol.freq <- rsolve_mdp_sa(model.freq$mdp.nominal, discount, "l1", model.freq$budgets, 
                          show_progress = FALSE)

cat("Hoeffding Confidence Region: ", sol.freq$valuefunction$value %*% init.dist, "policy:", sol.freq$policy$idaction, "\n")


## ---- Bayesian Confidence Region -----

rmdp.bayesian <- function(mdp.bayesian, global = FALSE){
# adjust the confidence level
  if(global){
    # provides a bound only on the return, not the value function
    confidence.rect <- 1-confidence
  }else{
    # provides a bound on the value function and the return
    confidence.rect <- (1-confidence)/sa.count    
  }


  # construct the mean bayesian model
  mdp.mean.bayes <- mdp.bayesian %>% group_by(idstatefrom, idaction, idstateto) %>%
    summarize(probability = mean(probability), reward = mean(reward))
  mean.probs <- mdp.mean.bayes %>% rename(probability_mean=probability) %>%
                  select(-reward)
  
  # compute L1 distances
  budgets <- 
    inner_join(mean.probs, 
               mdp.bayesian, 
               by = c('idstatefrom', 'idaction', 'idstateto')) %>%
      mutate(diff = abs(probability - probability_mean)) %>%
      group_by(idstatefrom, idaction, idoutcome) %>%
      summarize(l1 = sum(diff)) %>%
      group_by(idstatefrom, idaction) %>%
      summarize(budget = quantile(l1, 1-confidence.rect)) %>%
      rename(idstate = idstatefrom)
  
  return(list(mdp.mean = mdp.mean.bayes,
              budgets = budgets))
}

model.bayes.loc <- rmdp.bayesian(mdp.bayesian, FALSE) 
sol.bcr <- rsolve_mdp_sa(model.bayes.loc$mdp.mean, discount, "l1", 
                         model.bayes.loc$budgets, show_progress = FALSE)
cat("Local BCR: ", sol.bcr$valuefunction$value %*% init.dist, "policy:", sol.bcr$policy$idaction, "\n")

model.bayes.glo <- rmdp.bayesian(mdp.bayesian, TRUE) 
sol.gbcr <- rsolve_mdp_sa(model.bayes.glo$mdp.mean, discount, "l1", 
                         model.bayes.glo$budgets, show_progress = FALSE)
cat("Global BCR: ", sol.gbcr$valuefunction$value %*% init.dist, "policy:", sol.gbcr$policy$idaction, "\n")

## ---- RSVF-like (simplified) -------

confidence.rect <- (1-confidence)/sa.count
sa.count <- nrow( unique(rewards.truth %>% select(idstatefrom, idaction)))
sol.rsvf <- rsolve_mdpo_sa(mdp.bayesian, discount, "evaru", 
                            list(alpha = confidence.rect, beta = 1.0), show_progress = FALSE)
cat("RSVF: ", sol.rsvf$valuefunction$value %*% init.dist, "policy:", sol.rsvf$policy$idaction, "\n")

## ---- NORBU ----------

# not divided by the confidence
sol.norbu <- rsolve_mdpo_sa(mdp.bayesian, discount, "eavaru", 
               list(alpha = 1-confidence, beta = 1.0), show_progress = FALSE)
cat("NORBU: ", sol.norbu$valuefunction$value %*% init.dist, "policy:", sol.norbu$policy$idaction, "\n")

