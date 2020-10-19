
# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)


#' Dynamic RMDP with robust CVaR objective
#'
#' Uses Bayesian credible regions with the size determined using the union bound 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
algorithm_main <- function(mdpo, initial, discount){
	# basic implementation
	source("lib/norbu_lib.R", local = TRUE)

	return(norbu(mdpo, initial, discount, "eavaru", params$confidence, 1.0))
}
