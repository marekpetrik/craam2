# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)

#' Computes policy and an estimated return using a MILP formulation and uses 
#' scenario aggregation
#'
#' Uses Bayesian credible regions with the size determined using the union bound 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
algorithm_main <- function(mdpo, initial, discount){
    
    # the maximum time allowed to run (algorithm fails if this time limit expires)
    # in seconds
    time_limit <- params$time_limit
    cat("    Running with a time limit:", time_limit, "s.\n")

    # read the global confidence
    confidence <- params$confidence
    risk_weight <- params$risk_weight

    #mdpo <<- mdpo
    #initial <<- initial
    #discount <<- discount
    #stop("Not ready")
    
    # *** reduce the number of outcomes

    # number of clusters computed based on the confidence level
    # make sure that there can be at least 3 elements lower than 
    # the value at risk 
    nclusters <- min(5/(1 - params$confidence), 50)
    cat("    Clustering outcomes to", nclusters, "clusters.\n");

    cat("    Computing clairvoyant solution to cluster outcomes.\n")
    # compute optimal state value functions
    all_values <- revaluate_mdpo_rnd(mdpo, discount, show_progress = FALSE)

    # compute state-action value functions
    all_values_q <- all_values %>% group_by(idoutcome) %>%
        group_modify(~ compute_qvalues(filter(mdpo, idoutcome == .y$idoutcome), discount, .x) )

    all_values_wide <- tidyr::pivot_wider(all_values_q, idoutcome, 
                                          names_from = c(idstate, idaction), values_from = qvalue) 

    all_values_matrix <- model.matrix(~ . - idoutcome -1, data = all_values_wide)
    clustered <- kmeans(all_values_matrix, nclusters, 3)
    cat("    Clustering: between_ss/tot_ss =", clustered$betweenss / clustered$totss, "(better closer to 1)\n")

    #library(ggfortify)
    #autoplot(clustered, data = all_values_wide)

    # remember that the id's are 0-based
    outcome_cluster <- data.frame(idoutcome = as.integer(seq_along(clustered$cluster) - 1), 
                                  idoutcome_new = as.integer(clustered$cluster - 1))
    cluster_size <- data.frame(idoutcome_new = as.integer(seq_along(clustered$size) - 1),
                               ncount = clustered$size)

    # compute the mean over the observations
    # NOTE: do not compute a mean, because the number of outcomes per idstateto may differ
    #       instead sum the values and then normalize
    mdpo_c <-
        full_join(mdpo, outcome_cluster, by = "idoutcome") %>%
            group_by(idstatefrom, idaction, idoutcome_new, idstateto) %>%
            summarize(probability_sum = sum(probability), reward_sum = sum(probability * reward), .groups = "drop")  %>%
            full_join(cluster_size, by = "idoutcome_new") %>%
            mutate(probability = probability_sum / ncount, reward = reward_sum / ncount,
               idoutcome = idoutcome_new) %>%
            select(-ncount, -probability_sum, -reward_sum, -idoutcome_new) %>%
            na.fail()

    cat("    Done clustering. Starting MILP.\n")
    # code to check that the probabilities sum to 1
    #max((mdpo_c %>% group_by(idstatefrom, idaction, idoutcome) %>%
    #    summarize(s = sum(probability)))$s)

    # get a location for a temporary file
    log_file <- tempfile();
    cat("    Logging MILP to:", log_file, "\n")

    # Set parameters for the MILP gurobi environment
    gurobi_set_param("nonconvex", "LogFile", log_file)
    # change to 1 to enable logging
    gurobi_set_param("nonconvex", "OutputFlag", "1")
    gurobi_set_param("nonconvex", "LogToConsole", "1");

    # how many independent MILP solvers to run (helps when there are many threads)
    par_runs <- 4 #ceiling(parallel::detectCores() / 12)
    gurobi_set_param("nonconvex", "ConcurrentMIP", as.character(par_runs));  

    gurobi_set_param("nonconvex", "MIPGap", "0.05");
    cat("    Running with time limit", time_limit, "s ...\n")
    cat("    This computation cannot be terminated without killing the R process!\n")
    gurobi_set_param("nonconvex", "TimeLimit", as.character(time_limit))

    solution <- srsolve_mdpo(mdpo_c, initial, discount, alpha = 1 - confidence, beta = risk_weight)

    list(policy = solution$policy, estimate = solution$objective)
}
