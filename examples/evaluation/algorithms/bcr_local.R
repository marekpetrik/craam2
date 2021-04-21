# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)

#' Constructs a Bayesian credible region for an MDPO
#' 
#' The lower bound guarantee is on the total return and the value function
#' 
#' The parameter risk_w is used to scale down the size of the ambiguity set
#' (also known as the budget)
#' 
#' @param mdpo MDP with Bayesian outcomes, it is important 
#' that for each outcome, idstatefrom, idaction, idstateto are unique
rmdp_bayesian <- function(mdpo, confidence){

  # computes a bound on the value function and the return
  # the sa.count is necessary because of a union bound being used
  sa.count <- nrow( unique(mdpo %>% select(idstatefrom, idaction)))
  # maximum allowed probability of failure
  failure.rect <- (1 - confidence) / sa.count    

  # budget scaling to add artificial soft-robustness
  budget_scaling <- params$risk_weight
  
  # compute the number of unique outcomes in the mdpo, 
  # this is needed to compute the mean transition probability
  n_outcomes <- mdpo %>% select(idoutcome) %>% unique() %>% nrow()

  # construct the mean Bayesian model (nominal probabilities)
  mdp.mean.bayes <- mdpo %>% group_by(idstatefrom, idaction, idstateto) %>%
    summarize(probability = sum(probability) / n_outcomes, 
              reward = sum(reward) / n_outcomes, .groups = "drop")
  
  # NOTE: it may be tempting to use means(probability) above, but that is not consistent
  # because number of models that contain the given transition varies depending depends
  # on the state
  
  mean.probs <- mdp.mean.bayes %>% rename(probability_mean = probability) %>%
                  select(-reward)
  
  # compute L1 distances
  budgets <- 
    inner_join(mean.probs, 
               mdpo, 
               by = c('idstatefrom', 'idaction', 'idstateto')) %>%
      mutate(diff = abs(probability - probability_mean)) %>%
      group_by(idstatefrom, idaction, idoutcome) %>%
      summarize(l1 = sum(diff), .groups = "keep") %>%
      group_by(idstatefrom, idaction) %>%
      summarize(budget = budget_scaling * quantile(l1, 1 - failure.rect), .groups = "keep") %>%
      rename(idstate = idstatefrom) 
  
  return(list(mdp.mean = mdp.mean.bayes,
              budgets = budgets))
}

#' Computes policy and an estimated return 
#'
#' Uses Bayesian credible regions with the size determined using the union bound 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
algorithm_main <- function(mdpo, initial, discount){
	# read the confidence from the global value
	confidence <- params$confidence

	# build the rmdp
	rmdp <- rmdp_bayesian(mdpo, confidence) 
	
	# solve rmdp
	solution <- rsolve_mdp_sa(rmdp$mdp.mean, discount, "l1", 
														rmdp$budgets, show_progress = FALSE)
	
	# compute expected return 
	ret <- full_join(solution$valuefunction, initial, by = "idstate" ) %>%
		mutate(pv = probability * value) %>% na.fail()
	
	# return
	list(policy = solution$policy, estimate = sum(ret$pv))
}
