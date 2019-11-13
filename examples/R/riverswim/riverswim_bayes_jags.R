library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)
loadNamespace("tidyr")
loadNamespace("reshape2")
loadNamespace("stringr")

## ----- Parameters --------

description <- "riverswim3_mdp.csv"

init.dist <- rep(1/6,6)
discount <- 0.9
confidence <- 0.9
bayes.samples <- 500

samples <- 100
sample.seed <- 1984
episodes <- 1

## ----- Initialization ------

mdp.truth <- read_csv(description, 
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

## ----- Fit JAGS Model ----------------

#' Generate a sample MDP from dirichlet distribution
#' @param simulation Simulation results
#' @param rewards.df Rewards for each idstatefrom, idaction, idstateto
#' @param outcomes Number of outcomes to generate
mdpo_bayes <- function(simulation, rewards.df, outcomes){
  library(rjags)
  
  simulation.subset <- simulation %>%
    filter(idstatefrom > 0, idstatefrom < 5, idaction == 1) %>%
    mutate(difference = idstateto - idstatefrom + 2)
  
  model_spec <- textConnection("
      model {
        for (i in 1:N){
          difference[i] ~ dcat(beta) 
        }
        beta ~ ddirch(dprior)
      }
      ")
  jags <- 
    jags.model(model_spec,
               data = list(difference = simulation.subset$difference ,
                           N = nrow(simulation.subset),
                           dprior = c(1,1,1)),
               n.chains=4, n.adapt=100)
  update(jags, 1000) # warmup
  
  post_samples <- 
    coda.samples(jags, c('beta'), bayes.samples, thin = 4)
  
  raw_samples <- do.call(rbind, lapply(1:4, function(i) {as.matrix(post_samples[[i]])}))
  
  # subsample transitions for each case
  beta1 <- 
    rewards.df %>% filter((idstatefrom > idstateto) | 
                            (idstatefrom == 0 & idstateto == 0)) %>%
                  filter(idaction == 1)
  beta2 <- 
    rewards.df %>% filter(idstatefrom == idstateto & idaction == 1)
    
  beta3 <- 
    rewards.df %>% filter((idstatefrom < idstateto) | 
                            (idstatefrom == 5 & idstateto == 5)) %>%
      filter(idaction == 1)
  
  gen.beta.mdp <- function(beta, idoutcome.n){
    rbind(beta1 %>% mutate(probability = beta[1]),
          beta2 %>% mutate(probability = beta[2]),
          beta3 %>% mutate(probability = beta[3])) %>%
      mutate(idoutcome = idoutcome.n)
  }
  trans.prob <- mapply(gen.beta.mdp, raw_samples, 0:(nrow(raw_samples)-1)) 
  
  mdpo <- bind_rows(trans.prob) %>% 
    full_join(rewards.df, 
              by = c('idstatefrom', 'idaction','idstateto')) %>%
    na.fail()
  return(mdpo)
}

mdp.bayesian <- mdpo_bayes(simulation, rewards.truth, bayes.samples)

#' Evaluate the policy with respect to all possible 
#' Bayesian outcomes. 
#' 
#' Returns the return values.
#' 
#' @param mdp.bayesion MDPO with outcomes
#' @param policy Deterministic policy to be evaluated
bayes.returns <- function(mdp.bayesian, policy, maxcount = 100){
  sapply(unique(mdp.bayesian$idoutcome[1:maxcount]),
         function(outcome){
           sol <- mdp.bayesian %>% filter(idoutcome == outcome) %>% 
             solve_mdp(discount, policy_fixed = policy, 
                       show_progress = FALSE, 
                       algorithm = "pi")          
           sol$valuefunction$value %*% init.dist
         })
}

#' Prints experiment result statistics.
#' 
#' It also prints its guarantees, solution quality and 
#' posterior expectation of how well it is likely to work
#' 
#' @param name Name of the algorithm that produced the results
#' @param mdp.bayesian MDP with outcomes representing bayesian samples
#' @param solution Output from the algorithm's solution
report_solution <- function(name, mdp.bayesian, solution){
  cat("**", stringr::str_pad(name, 15, 'right'), 
      solution$valuefunction$value %*% init.dist, "****\n")
  cat("    Policy", solution$policy$idaction,"\n")
  
  cat("    Return predicted:", solution$valuefunction$value %*% init.dist)
  sol.tr <- solve_mdp(mdp.truth, discount, 
                      policy_fixed = solution$policy,
                      show_progress = FALSE)
  cat(", true:", sol.tr$valuefunction$value %*% init.dist, "\n")
  posterior.returns <- bayes.returns(mdp.bayesian, solution$policy)
  cat("    Posterior mean:", mean(posterior.returns), ", v@r:", 
      quantile(posterior.returns, 1-confidence), ", av@r:", 
      mean(posterior.returns[posterior.returns < 
                          quantile(posterior.returns, 1-confidence)]), "\n")
}

## ---- Solve Empirical MDP ---------

mdp.empirical <- mdp_from_samples(simulation)
sol.empirical <- solve_mdp(mdp.empirical, discount, show_progress = FALSE)
report_solution("Empirical", mdp.bayesian, sol.empirical)

## ----- Solve Bayesian MDP ---------

sol.bayesexp <- rsolve_mdpo_sa(mdp.bayesian, discount, "exp", NULL, show_progress = FALSE)
report_solution("Bayesian", mdp.bayesian, sol.bayesexp)

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

report_solution("Hoeff CR", mdp.bayesian, sol.freq)

## ---- Bayesian Confidence Region -----

rmdp.bayesian <- function(mdp.bayesian){
# adjust the confidence level

  # provides a bound on the value function and the return
  sa.count <- nrow( unique(mdp.bayesian %>% select(idstatefrom, idaction)))
  confidence.rect <- (1-confidence)/sa.count    

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

model.bayes.loc <- rmdp.bayesian(mdp.bayesian) 
sol.bcr <- rsolve_mdp_sa(model.bayes.loc$mdp.mean, discount, "l1", 
                         model.bayes.loc$budgets, show_progress = FALSE)
report_solution("Local BCR: ", mdp.bayesian, sol.bcr)

## ---- Global Bayesian Confidence Region -----

#' Uses a single global solution and relies on rectangularization. 
#' It picks the confidence fraction of outcomes ordered
#' by the mean budget size across all states
rmdp.bayesian.global <- function(mdp.bayesian){
  # construct the mean bayesian model
  mdp.mean.bayes <- mdp.bayesian %>% group_by(idstatefrom, idaction, idstateto) %>%
    summarize(probability = mean(probability), reward = mean(reward))
  mean.probs <- mdp.mean.bayes %>% rename(probability_mean=probability) %>%
    select(-reward)
  
  # compute L1 distances for each outcome averaged/maxed over states
  distances <- 
    inner_join(mean.probs, 
               mdp.bayesian, 
               by = c('idstatefrom', 'idaction', 'idstateto')) %>%
    mutate(diff = abs(probability - probability_mean)) %>%
    group_by(idstatefrom, idaction, idoutcome) %>%
    summarize(l1 = sum(diff)) 
  
  outcome.mean.dist <- distances %>%
    group_by(idoutcome) %>%
    summarize(l1 = max(l1)) %>%
    arrange(l1)

  # the set of all outcomes that need to be covered by the ambiguity set
  cover.set <- outcome.mean.dist$idoutcome[1:ceiling(nrow(outcome.mean.dist) * confidence)]
  
  budgets <- inner_join(distances, 
                        data.frame(idoutcome = cover.set),
                        by = "idoutcome") %>% 
    group_by(idstatefrom, idaction) %>% 
    summarize(budget = max(l1)) %>%
    rename(idstate = idstatefrom)

  
  return(list(mdp.mean = mdp.mean.bayes,
              budgets = budgets))
}


model.bayes.glo <- rmdp.bayesian.global(mdp.bayesian) 
sol.gbcr <- rsolve_mdp_sa(model.bayes.glo$mdp.mean, discount, "l1", 
                          model.bayes.glo$budgets, show_progress = FALSE)
report_solution("Global BCR", mdp.bayesian, sol.gbcr)

## ---- RSVF-like (simplified) -------

sa.count <- nrow( unique(rewards.truth %>% select(idstatefrom, idaction)))
confidence.rect <- (1-confidence)/sa.count
sol.rsvf <- rsolve_mdpo_sa(mdp.bayesian, discount, "evaru", 
                            list(alpha = confidence.rect, beta = 1.0), show_progress = FALSE)
report_solution("RSVF", mdp.bayesian, sol.rsvf)

## ---- NORBU ----------

sol.norbu <- rsolve_mdpo_sa(mdp.bayesian, discount, "eavaru", 
               list(alpha = 1-confidence, beta = 1.0), show_progress = FALSE)
report_solution("NORBU: ", mdp.bayesian, sol.norbu)

# to see why this is wrong, try running it with a confidence 0.1 or something small
sol.norbu.w <- rsolve_mdpo_sa(mdp.bayesian, discount, "evaru", 
                            list(alpha = 1-confidence, beta = 1.0), show_progress = FALSE)
report_solution("NORBU(wrong): ", mdp.bayesian, sol.norbu.w)

