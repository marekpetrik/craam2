library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)
requireNamespace("tidyr")

init.dist <- rep(1/6,6)
discount <- 0.9
mdp.truth <- read_csv("riverswim_mdp.csv", 
                      col_types = cols(idstatefrom = 'i',
                                       idaction = 'i',
                                       idstateto = 'i',
                                       probability = 'd',
                                       reward = 'd'))
rewards.truth <- mdp.truth %>% select(-probability)
confidence <- 0.8

# construct a biased policy to prefer going right
# this is to ensure that the "goal" state is sampled
ur.policy = data.frame(idstate = c(seq(0,5), seq(0,5)), 
                     idaction = c(rep(0,6), rep(1,6)),
                     probability = c(rep(0.2, 6), rep(0.8, 6)))

# compute the true value function
sol.true <- solve_mdp(mdp.truth, discount, show_progress = FALSE)
vf.true <- sol.true$valuefunction$value
cat("True optimal return", vf.true %*% init.dist, "\n")

## ----- Generate Samples --------------

# generate samples from the swimmer domain
simulation <- simulate_mdp(mdp.truth, 0, ur.policy, episodes = 1, horizon = 300, seed = 2019)

## ---- Solve Empirical MDP ---------
  
mdp.empirical <- mdp_from_samples(simulation)
sol.empirical <- solve_mdp(mdp.empirical, discount, show_progress = FALSE)
vf.empirical <- sol.empirical$valuefunction$value
cat("Empirical solution estimate: ", vf.empirical %*% init.dist, "\n")

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

cat("Hoeffding Confidence Region: ", sol.freq$valuefunction$value %*% init.dist, "\n")

## ----  Uninformative Bayesian Posterior Sampling ---------


#' Generate a sample MDP from dirichlet distribution
#' @param posteriors Posterior distributions for each state and action
#' @param rewards.df Rewards for each idstatefrom, idaction, idstateto
#' @param outcomes Number of outcomes to generate
sample_mdp_bayes <- function(posteriors, rewards.df, outcomes){

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
  trans.prob <- sapply(posteriors$dist, 
                       function(z){do.call(function(x) {rdirichlet(1,x)}, 
                                           list(z) )})
  dimnames(trans.prob) <- list(idstateto = NULL, idtransition = NULL)
  
  trans.frame <- melt(trans.prob, value.name = "probability") %>%
    mutate(idstateto = idstateto - 1)
  # join probabilities back with samples
  # and join it with the original mdp to get rewards
  mdp.sampled <- inner_join(trans.frame, posteriors %>% select(-dist),
                            by = 'idtransition') %>%
    select(-idtransition) %>%
    full_join(rewards.df %>% select(idstatefrom, idaction, idstateto, reward), 
              by = c('idstatefrom', 'idaction', 'idstateto')) %>%
    select(idstatefrom, idaction, idstateto, probability, reward)
  # the full join above is needed to include rewards for states that
  # were not sampled
  mdp.sampled$reward[is.na(mdp.sampled$reward)] <- 0
  
  if(!is.null(outcome)){
    mdp.sampled$idoutcome <- outcome
  }
  
  return(mdp.sampled)  
}

mdp.bayesian <- sample_mdp_bayes(posteriors, mdp.truth %>% select(-probability))

solve_mdp(mdp.bayesian, discount)

## ---- Robust Bayesian Model -------

mdpo.bayes <- do.call(rbind, lapply(seq(0,20), 
                                    function(ido){sample_mdp_bayes(posteriors, mdp.truth, ido)}))

sol.bay <- rsolve_mdpo_sa(mdpo.bayes, discount, "eavaru", 
               list(alpha = 0.5, beta = 1.0), show_progress = FALSE)

