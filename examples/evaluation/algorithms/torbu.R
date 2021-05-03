# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)

#' Computes policy and an estimated return using a MILP formulation
#' 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
algorithm_main <- function(mdpo, initial, discount){

    # read the global confidence
    confidence <- params$confidence
    risk_weight <- params$risk_weight

    # the maximum time allowed to run (algorithm fails if this time limit expires)
    # in seconds
    time_limit <- params$time_limit
    cat("    Running with a time limit:", time_limit, "s.\n")

    # get a location for a temporary file
    log_file <- tempfile();

    cat("    Logging to:", log_file, "\n")

    # Set parameters for the MILP gurobi environment
    gurobi_set_param("nonconvex", "LogFile", log_file)
    # change to 1 to enable logging
    gurobi_set_param("nonconvex", "OutputFlag", "1")
    gurobi_set_param("nonconvex", "LogToConsole", "1");

    # how many independent MILP solvers to run (helps when there are many threads)
    par_runs <- 4 #ceiling(parallel::detectCores() / 12)
    gurobi_set_param("nonconvex", "ConcurrentMIP", as.character(par_runs));    

    gurobi_set_param("nonconvex", "MIPGap", "0.05");
    cat("    This computation cannot be terminated without killing the R process!\n")
    gurobi_set_param("nonconvex", "TimeLimit", as.character(time_limit))


    # trim the number of outcomes used depending on the number of states and actions
    sa.count <- nrow(unique(mdpo %>% select(idstatefrom, idaction)))
    out.count <- nrow(unique(mdpo %>% select(idoutcome)))
    
    # assume that outcomes identifiers start with 0 and are contiguous
    out.use <- max(1, min(out.count - 1, 200e3 / sa.count))    # at least 2
    cat("    Trimming outcomes to", out.use + 1, " in the interest of speed. \n")
    mdpo.trim <- mdpo %>% filter(idoutcome <= out.use)

    solution <- srsolve_mdpo(mdpo.trim, initial, discount, 
                        alpha = 1 - confidence, beta = risk_weight)

    list(policy = solution$policy, estimate = solution$objective)
}
