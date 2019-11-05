library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)


discount <- 0.9
mdp <- read_csv("riverswim_mdp.csv")

# construct a biased policy to prefer going right
# this is to ensure that the "goal" state is sampled
ur.policy = data.frame(idstate = c(seq(0,5), seq(0,5)), 
                     idaction = c(rep(0,6), rep(1,6)),
                     probability = c(rep(0.2, 6), rep(0.8, 6)))

# generate samples from the swimmer domain
# TODO: add a seed to the method
simulation <- simulate_mdp(mdp, 0, ur.policy, episodes = 1, horizon = 300)

## ---- Solve Empirical MDP ---------
  
mdp.empirical <- mdp_from_samples(simulation)
sol <- solve_mdp(mdp.empirical, discount)

## ----  Bayesian Posterior Sampling ---------

# compute sampled state and action counts
# add a uniform sample of each state and action to work as the dirichlet prior
counts <- simulation %>% 
  select(idstatefrom, idaction, idstateto) %>%
  rbind(expand.grid(idstatefrom = seq(0,5), 
                    idaction = seq(0,1), 
                    idstateto = seq(0,5))) %>%
  group_by(idstatefrom, idaction, idstateto) %>% 
  summarize(count = n()) 

# compute dirichlet posteriors
posteriors <- counts %>% 
  group_by(idstatefrom, idaction) %>% 
  arrange(idstateto) %>% 
  summarize(dist = list(count)) 
posteriors$idtransition <- seq(nrow(posteriors))


#' Generate a sample from dirichlet distribution
sample_mdp_bayes <- function(posteriors, mdp, outcome = NULL){
  trans.prob <- sapply(posteriors$dist, 
                       function(z){do.call(function(x) {rdirichlet(1,x)}, 
                                           list(z) )})
  dimnames(trans.prob) <- list(idstateto = NULL, idtransition = NULL)
  
  trans.frame <- melt(trans.prob, value.name = "probability") %>%
    mutate(idstateto = idstateto - 1)
  # join probabilities back with samples
  # and join it with the original mdp to get rewards
  mdp.sampled <- inner_join(trans.frame, posteriors %>% select(-dist)) %>%
    select(-idtransition) %>%
    full_join(mdp %>% select(idstatefrom, idaction, idstateto, reward)) %>%
    select(idstatefrom, idaction, idstateto, probability, reward)
  mdp.sampled$reward[is.na(mdp.sampled$reward)] <- 0
  
  if(!is.null(outcome)){
    mdp.sampled$idoutcome <- outcome
  }
  
  return(mdp.sampled)  
}

mdp.bayesian <- sample_mdp_bayes(posteriors, mdp %>% select(-probability))

solve_mdp(mdp.bayesian, discount)


## ---- Robust Bayesian Model -------

mdpo.bayes <- do.call(rbind, lapply(seq(0,20), 
                                    function(ido){sample_mdp_bayes(posteriors, mdp, ido)}))

rsolve_mdpo_sa(mdpo.bayes, discount, "eavaru", 
               list(alpha = 0.1, beta = 0.5), list(algorithm = "vi"))

