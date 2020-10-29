#!/usr/bin/Rscript

# This script can be used to automate the evaluation and reporting of 
# the performance of various Bayesian robust and soft-robust algorithms.
# 
# The script is flexible and works with pre-built domains 
#
#
# Use the command browser() for debugging the methods (in their source)
#

suppressPackageStartupMessages({
  library(rcraam)
  library(dplyr)
  library(readr)
  library(progress)})


# more space for the huxtable output
options(width = 150)
# for interactive bug reduction
rm(list = ls())

test_on_train <- FALSE
if(test_on_train){
  cat(" ***** Reporting test results on the training set ***********")
}

## ------ Output file ---------

# where to save the final result
output_file <- "eval_results.csv"    
# whether to print partial results during the computation
print_partial <- TRUE

## ------ Define domains ------

# domains_path (where the csv files are stored)
domains_path <- "domains"

# domains_source (where the domains may be downloaded from)
domains_source <- "http://data.rmdp.xyz/domains"   # no trailing "/"

# list all domains that are being considered; each one should be
# in a separate directory that should have 3 files:
#   - true.csv.xz
#   - training.csv.xz  (posterior optimization samples)
#   - test.csv.xz      (posterior evaluation samples)
domains <- list(
#  riverswim = "riverswim",
#  pop_small = "population_small",
  population = "population"
)

domains_paths <- lapply(domains, function(d){file.path(domains_path, d)})

## ----- Define algorithms --------

# list of algorithms, each one implemented in a separate 
# input file with a function: 
# result = algorithm_main(mdpo, initial, discount), where the result is a list:
#    result$policy = computed policy
#    result$estimate = estimated return (whatever metric is optimized)
# and the parameters are:
#    mdpo: dataframe with idstatefrom, idaction, idstateto, idoutcome, probability, reward
#    initial: initial distribution, dataframe with idstate, probability (sums to 1)
#    discount: discount rate [0,1]
# 
# It is a good practice for each algorithm to the risk parameters 
# it is using. They may be reading the parameters from 
# the global environment. That is convenient but fragile
algorithms_path <- "algorithms"

algorithms <- list(
  #nominal = "nominal.R",
  #bcr_l = "bcr_local.R",
  #bcr_g = "bcr_global.R",
  #rsvf2 = "rsvf2.R",
  #norbu_r = "norbu_r.R",
  #norbu_sr = "norbu_sr.R",
  #norbuv_r = "norbuv_r.R",
  torbu = "torbu.R"
)

# construct paths to algorithms
algorithms_paths <- lapply(algorithms, function(a){file.path(algorithms_path, a)} ) 

## ------ Check domain availability ----

cat("Checking if domains are available ...\n")

if(!dir.exists(domains_path)) dir.create(domains_path)
for(idpath in seq_along(domains_paths)){
  if(dir.exists(domains_paths[[idpath]])){
    cat("Domain", names(domains)[[idpath]], "available, using cached version.\n")
  } else {
    cat("Domain", names(domains)[[idpath]], "unavailable, downloading...\n")
    cat("  Creating", domains_paths[[idpath]], "...\n")
    dir.create(domains_paths[[idpath]])
    withCallingHandlers({
      domain_files <- c("parameters.csv", "true.csv.xz", "initial.csv.xz", 
                        "training.csv.xz","test.csv.xz")
      for(dfile in domain_files){
        urlf <- paste(domains_source, domains[[idpath]], dfile, sep = "/")
        targetf <- file.path(domains_paths[[idpath]], dfile)
        cat("Downloading", urlf, "to", targetf, "\n")
        download.file(urlf, targetf)
      }
    }, 
    error = function(e){
      cat("Download error! Stopping.\n")
      unlink(domains_paths[[idpath]], recursive = TRUE, force = TRUE)
      stop(e)
    })
  }
}


## ----- Parameters --------

# the algorithms can read and use these parameters
params <- new.env()
with(params, {
  confidence <- 0.9             # value at risk confidence (1.0 is the worst case)
  risk_weight <- 0.5            # weight on the cvar when computing the soft-robust objective
  
  cat("Using confidence =", confidence, ", risk_weight =", risk_weight, "\n") 
})

## ---- Helper Methods: Evaluation -----------


#' Evaluate a policy with respect to the true model
#' 
#' @param mdp True MDP with outcomes
#' @param policy Deterministic policy to be evaluated
#' @param initial Initial distribution (dataframe)
#' @param discount Discount factor
compute_true_return <- function(mdp_true, policy, initial, discount){
    sol <- solve_mdp(mdp_true, discount, policy_fixed = policy,
              show_progress = FALSE)
    ret <- full_join(sol$valuefunction, initial,by = "idstate" ) %>%
           mutate(pv = probability * value) %>% na.fail()
    sum(ret$pv)
}

#' Compute statistics for a computed (robust) policy
#' 
#' @param name Name of the algorithm that produced the results
#' @param mdp.bayesian MDP with outcomes representing Bayesian samples
#' @param solution Output from the algorithm (policy and predicted)
compute_statistics <- function(mdpo, mdp_true, solution, initial, discount){
  # make sure that the policy is randomized (if no probabilities, just add the column)
  policy <- solution$policy
  if(!("probability" %in% names(policy)))
    policy$probability <- 1.0
  # compute the returns
  bayes_returns <- revaluate_mdpo_rnd(mdpo, discount, policy, initial, TRUE)$return
  
  # assume a uniform distribution over the outcomes
  bayes_dst <- rep(1/length(bayes_returns), length(bayes_returns))
  true_return <- compute_true_return(mdp_true, solution$policy, initial, discount)
  
  # compute cvar and other metrics
  cvar_val <- avar(bayes_returns, bayes_dst, 1 - params$confidence)$value
  mean_val <- mean(bayes_returns) 
  list(
    predicted = solution$estimate,
    true = true_return,
    var = unname(quantile(bayes_returns, 1 - params$confidence)),
    cvar = cvar_val,
    mean = mean_val,
    soft_rbst = (1 - params$risk_weight) * mean_val + 
      params$risk_weight * cvar_val
  )
}

## ---- Helper Methods:  -------

#' Loads problem domain information
#'
#' @param dir_path Path to the directory with the required files
load_domain <- function(dir_path){
  cat("    Loading parameters ... \n")
  parameters <- read_csv(file.path(dir_path, "parameters.csv"), col_types = cols())
  
  cat("    Loading true MDP ... ")
  true_mdp <- read_csv(file.path(dir_path, "true.csv.xz"), col_types = cols())
  sa.count <- select(true_mdp, idstatefrom, idaction) %>% unique() %>% nrow()
  cat(sa.count, "state-actions \n")
    
  cat("    Loading initial distribution ... ")
  initial <- read_csv(file.path(dir_path, "initial.csv.xz"), col_types = cols())
  s.count <- select(initial, idstate) %>% unique() %>% nrow()
  cat(s.count, "states \n")
  
  
  # TRAINING
  # unzip and save the raw csv file to speed up loading of the file
  cat("    Loading training file ... ")
  training_file <- file.path(dir_path, "training.csv.xz")
  stopifnot(file.exists(training_file))
  if(file.size(training_file) < 10e6){
    training <- read_csv(training_file, col_types = cols())
  } else {  
    training_raw <- tools::file_path_sans_ext(training_file)
    if(!file.exists(training_raw)) {
      cat("      Large, decompressing using pixz ... \n")
      system2("pixz", paste("-d -k", training_file))
    }
    training <- read_csv(training_raw, col_types = cols())
  }
  # make sure that the training data is sorted with increasing idoutcome
  training <- arrange(training, idoutcome)

  sa.count <- select(training, idstatefrom, idaction) %>% unique() %>% nrow()
  out.count <- select(training, idoutcome) %>% unique() %>% nrow()
  cat(sa.count, "state-actions,", out.count, "outcomes \n")
  

  # TEST
  # unzip and save the raw csv file to speed up loading of the file
  cat("    Loading test file ... ")
  test_file <- file.path(dir_path, "test.csv.xz")
  stopifnot(file.exists(test_file))
  if(file.size(test_file) < 10e6){
    test <- read_csv(test_file, col_types = cols()) 
  } else {
    test_raw <- tools::file_path_sans_ext(test_file)
    if(!file.exists(test_raw)) {
      cat("      Large, decompressing using pixz ... \n")
      system2("pixz", paste("-d -k", test_file))
    }
    test <- read_csv(test_raw, col_types = cols() )
  }
  # make sure that the test data is sorted with increasing idoutcome
  test <- arrange(test, idoutcome)
  
  sa.count <- select(test, idstatefrom, idaction) %>% unique() %>% nrow()
  out.count <- select(test, idoutcome) %>% unique() %>% nrow()
  cat(sa.count, "state-actions,", out.count, "outcomes \n")

  list(
    discount = filter(parameters, parameter == "discount")$value[[1]],
    initial_dist = initial,
    true_mdp = true_mdp,
    training_mdpo = training,
    test_mdpo = test
  )
}

## ------ Printing -----------------

#' Pretty prints the results
#' @param results_frame Algorithm domain results
print_results <- function(results_frame) {
  tryCatch({
    if(requireNamespace("huxtable", quietly = TRUE)){
      huxtable::print_screen(huxtable::hux(results_frame) %>% 
                             huxtable::set_all_borders() %>% 
                             huxtable::set_bold(row=1, col=huxtable::everywhere, value=TRUE),
                   colnames = FALSE, color = TRUE, compact = FALSE)
      cat("\n")
    } else {
      cat("Install huxtable to get pretty results!\n")
      print(results)
    }
  })
}



## ------ Main Method ----------------

#' Main evaluation function
#' 
#' Runs all the algorithms over the domains and returns the summary of the results 
#'
#' The result has one column for the domain, one column for the algorithms, 
#' and each statistic gets one column too.
#'
#' @param domains_paths List of paths to domain directories
#' @param algorithms_paths List of paths to algorithm R implementation files
#'
#' @return Dataframe that contain the results
main_eval <- function(domains_paths, algorithms_paths){
  
  # This will contain the results, one column for domain name, algorithm name, and each statistic
  results <- list()
  iteration <- 1
  
  cat("Running algorithms ... \n")

  # iterate over all domains
  for (i_dom in seq_along(domains_paths)) {
    domain_name <- names(domains_paths[i_dom])
    cat("*** Processing domain", domain_name, " ...\n")
    domain_spec <- load_domain(domains_paths[[i_dom]])
    
    # iterate over all algorithms
    for (i_alg in seq_along(algorithms_paths)) {

      algorithm_name <- names(algorithms_paths[i_alg])
      cat("  Running algorithm", algorithm_name, " ... \n")
      
      # load the algorithm into its own separate environment
      alg_env <- new.env() # make sure that algorithm runs are isolated as much as possible
      sys.source(algorithms_paths[[i_alg]], alg_env, keep.source = TRUE, chdir = TRUE)    
      
      # call algorithm
      solution <- with(alg_env, {
        algorithm_main(domain_spec$training_mdpo, domain_spec$initial_dist, 
                      domain_spec$discount) } )
      stopifnot("policy" %in% names(solution))
      stopifnot("estimate" %in% names(solution))

      if(!is.null(solution)){
        cat("  Evaluating ... \n")
        # compute and store stats
        if(!test_on_train){
          statistics <- with(domain_spec, { 
            compute_statistics(test_mdpo, true_mdp, solution, initial_dist, discount) } )
        } else {
          statistics <- with(domain_spec, { 
            compute_statistics(training_mdpo, true_mdp, solution, initial_dist, discount) } )
        }
        statistics$domain <- domain_name
        statistics$algorithm <- algorithm_name

        results[[iteration]] <- statistics
        # keep track of the iteration
        iteration <- iteration + 1
      } else {
        cat("  No solution returned, skipping evaluation ...\n");
      }
      # TODO: It would be good to detach and better isolate the execution
      results_table <- bind_rows(results) %>% relocate(domain, algorithm)
      if(print_partial){
        cat("Results so far (saving just in case): \n")
        print_results(results_table)
      }
      write_csv(results_table, output_file)
    }
  }

  cat("Done computing, formatting...\n")
  results_table <- bind_rows(results) %>% relocate(domain, algorithm)
  cat("Done.\n")
  return (results_table)
}

results <- main_eval(domains_paths, algorithms_paths)

cat("*** Results: \n")
print_results(results)
write_csv(results, output_file)
