# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)


#' Dynamic RMDP with VaR objective and union bound confidence modification
#'
#' Uses Bayesian credible regions with the size determined using the union bound 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
algorithm_main <- function(mdpo, initial, discount){
	# read confidence from the global value
	confidence <- params$confidence
	
	# basic implementation
	source("lib/norbu_lib.R", local = TRUE)

	sa.count <- nrow( unique(mdpo %>% select(idstatefrom, idaction)))

	# apply rectangularization and the union bound
	confidence.rect <- (1 - confidence) / sa.count

	return(norbu(mdpo, initial, discount, "evaru", 
	             1 - confidence.rect, params$risk_weight))
}
