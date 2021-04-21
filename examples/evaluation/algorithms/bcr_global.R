# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)

#' Constructs a Bayesian credible region for an MDPO
#' 
#' Uses a single global solution and relies on rectangularization. 
#' It picks the confidence fraction of outcomes ordered
#' by the mean budget size across all states
#' 
#' @param mdpo with Bayesian outcomes, it is important 
#' that for each outcome, idstatefrom, idaction, idstateto are unique
rmdp_bayesian_global <- function(mdpo, confidence){

  # compute the number of unique outcomes in the mdpo, 
  # this is needed to compute the mean transition probability
  n_outcomes <- mdpo %>% select(idoutcome) %>% unique() %>% nrow()

  # budget scaling to add artificial soft-robustness
  budget_scaling <- params$risk_weight
  
  # construct the mean Bayesian model (nominal probabilities)
  mdp.mean.bayes <- mdpo %>% group_by(idstatefrom, idaction, idstateto) %>%
    summarize(probability = sum(probability) / n_outcomes, 
              reward = sum(reward) / n_outcomes, .groups = "drop")
  
  # NOTE: it may be tempting to use means(probability) above, but that is not consistent
  # because number of models that contain the given transition varies depending depends
  # on the state
  
  mean.probs <- mdp.mean.bayes %>% rename(probability_mean = probability) %>%
                  select(-reward)
  
  # compute L1 distances for each outcome averaged/maxed over states
  distances <- 
    inner_join(mean.probs, mdpo,
               by = c('idstatefrom', 'idaction', 'idstateto')) %>%
    mutate(diff = abs(probability - probability_mean)) %>%
    group_by(idstatefrom, idaction, idoutcome) %>%
    summarize(l1 = sum(diff), .groups = "keep") 
  
  outcome.mean.dist <- distances %>% group_by(idoutcome) %>%
    summarize(l1 = max(l1), .groups = "keep") %>% arrange(l1)

  # the set of all outcomes that need to be covered by the ambiguity set
  cover.set <- outcome.mean.dist$idoutcome[1:ceiling(nrow(outcome.mean.dist) * confidence)]
  
  budgets <- inner_join(distances, data.frame(idoutcome = cover.set),
                        by = "idoutcome") %>% 
    group_by(idstatefrom, idaction) %>% 
    summarize(budget = budget_scaling * max(l1), .groups = "keep") %>%
    rename(idstate = idstatefrom)
  
  return(list(mdp.mean = mdp.mean.bayes,
              budgets = budgets))
}

algorithm_main <- function(mdpo, initial, discount){
    
	# read the confidence from the global value
	confidence <- params$confidence

  # build the rmdp
  rmdp <- rmdp_bayesian_global(mdpo, confidence) 
    
  # solve rmdp
  solution <- rsolve_mdp_sa(rmdp$mdp.mean, discount, "l1", 
														rmdp$budgets, show_progress = FALSE)
	
	# construct policy 
	ret <- full_join(solution$valuefunction, initial, by = "idstate" ) %>%
		mutate(pv = probability * value) %>% na.fail()
	
	# return
	list(policy = solution$policy, estimate = sum(ret$pv))
}
