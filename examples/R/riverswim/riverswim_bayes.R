library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)

init.state <- 1
discount <- 0.9
mdp.truth <- read_csv("riverswim_mdp.csv", col_types = cols())
confidence <- 0.8

# construct a biased policy to prefer going right
# this is to ensure that the "goal" state is sampled
ur.policy = data.frame(idstate = c(seq(0,5), seq(0,5)), 
                     idaction = c(rep(0,6), rep(1,6)),
                     probability = c(rep(0.2, 6), rep(0.8, 6)))

# compute the true value function
vf.true <- solve_mdp(mdp.truth, discount)$valuefunction$value
cat("True optimal return", vf.true[init.state], "\n")

# generate samples from the swimmer domain
simulation <- simulate_mdp(mdp.truth, 0, ur.policy, episodes = 1, horizon = 300, seed = 2019)

## ---- Solve Empirical MDP ---------
  
mdp.empirical <- mdp_from_samples(simulation)
sol.empirical <- solve_mdp(mdp.empirical, discount)
vf.empirical <- sol.empirical$valuefunction$value
cat("Empirical solution estimate: ", vf.empirical[init.state], "\n")

## ----- Frequentist Confidence Interval ------
# See e.g. Russel 2019
# Confidence is: sqrt{ 1 / n_{s,a}  log (S A 2^S / \delta) }
# where delta is the confidence

sa_counts <- simulation %>% select(idstatefrom, idaction) %>%
              group_by(idstatefrom, idaction) %>%
              summarize(count = n())

state.count <- max(sa_counts$idstatefrom) + 1
sa.count <- nrow(sa_counts)                   # number of valid state-action pairs
budgets <- sa_counts %>% 
  mutate(budget = pmin(2,sqrt(1/count * log(sa.count * 2^state.count / confidence)))) %>%
  rename(idstate = idstatefrom) %>% select(-count)

#mdp.empirical %>% 

sol.freq <- rsolve_mdp_sa(mdp.empirical, discount, "l1", budgets)

## ----  Uninformative Bayesian Posterior Sampling ---------

# compute sampled state and action counts
# add a uniform sample of each state and action to work as the dirichlet prior
sas_counts <- simulation %>% 
  select(idstatefrom, idaction, idstateto) %>%
  rbind(expand.grid(idstatefrom = seq(0,5), 
                    idaction = seq(0,1), 
                    idstateto = seq(0,5))) %>%
  group_by(idstatefrom, idaction, idstateto) %>% 
  summarize(count = n()) 

# construct dirichlet posteriors
posteriors <- sas_counts %>% 
  group_by(idstatefrom, idaction) %>% 
  arrange(idstateto) %>% 
  summarize(dist = list(count)) 
posteriors$idtransition <- seq(nrow(posteriors))


#' Generate a sample from dirichlet distribution
sample_mdp_bayes <- function(posteriors, mdp, outcome = NULL){
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
    full_join(mdp %>% select(idstatefrom, idaction, idstateto, reward), 
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
               list(alpha = 0.5, beta = 1.0), algorithm = "mppi")

