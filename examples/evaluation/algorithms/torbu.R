# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)

#' Computes policy and an estimated return using a MILP formulation
#'
#' Uses Bayesian credible regions with the size determined using the union bound 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
algorithm_main <- function(mdpo, initial, discount){

	# read the global confidence
	confidence <- params$confidence
	risk_weight <- params$risk_weight

	# get a location for a temporary file
	log_file <- tempfile();

	cat("    Logging to:", log_file, "\n")

  # Set parameters for the MILP gurobi environment
  gurobi_set_param("nonconvex", "LogFile", log_file)
	# change to 1 to enable logging
  gurobi_set_param("nonconvex", "OutputFlag", "0")
  gurobi_set_param("nonconvex", "LogToConsole", "0");

	# how many independent MILP solvers to run (helps when there are many threads)
	par_runs <- ceiling(parallel::detectCores() / 8)
  gurobi_set_param("nonconvex", "ConcurrentMIP", as.character(par_runs));  
	
  gurobi_set_param("nonconvex", "MIPGap", "0.05");
  gurobi_set_param("nonconvex", "TimeLimit", "1000")

	solution <- srsolve_mdpo(mdpo, initial, discount, 
						alpha = 1 - confidence, beta = risk_weight)

	list(policy = solution$policy, estimate = solution$objective)
}
