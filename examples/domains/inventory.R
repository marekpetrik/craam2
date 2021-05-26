#!/usr/bin/Rscript

# This scripts creates an inventory management problem dataset, samples data from it, 
# computes the posterior distribution, and then saves samples from this distribution
#  
# The states in this problem domain represent the inventory level. The actions 
# represent how much of the product is ordered in each time step. The transitions
# are stochastic because the demand is stochastic distributed according to 
# the Poisson distribution.
#
# The rewards in this problem depend on state, action, and the next state and
# are assumed to be known.
#
# The demand distribution is uncertain. The demand comes from a Poisson distribution
# with rate lambda. The prior distribution is Gamma with parameters: 
# k = shape, and theta = scale.
# 
# If the demand samples are x_1, \ldots, x_n then posterior distribution is: 
# (https://en.wikipedia.org/wiki/Conjugate_prior)
# k' = k + sum_i x_i
# theta' = theta / (n*theta + 1)
#
# The sampling policy is to just purchase the maximum available goods;
# that way we sample uncensored demand and can compute the posterior
# distribution analytically
#  
# 
# See the domains README.md file for details about the files that are created


# remove anything in the namespace just in case this being
# run repeatedly interactively
rm(list = ls())

suppressPackageStartupMessages({
library(rcraam)
library(dplyr)
library(readr)
library(progress)
library(parallel)
})

normalize <- function(x) {x / sum(x)}

## ----- Parameters --------

sample.seed <- 1984

# data output (platform independent construction)
folder_output <- file.path('domains', 'inventory')  

# inventory management problem specification
inventory_params <- function(lambda){
    list(
        variable_cost = 2.49,
        fixed_cost = 0.49,
        holding_cost = 0.05,
        backlog_cost = 0.15,
        sale_price = 4.99,
        max_inventory = 20,      # number of states - 1
        max_backlog = 0,         # no backlogging allowed
        max_order = 10,          # number of actions - 1
        demands = normalize(dpois(0:30, lambda)),
        seed = sample.seed)
}

# prior distribution for the demand
# shape k and scale theta
prior_gamma <- c(shape = 4, scale = 6)

# k' = k + sum_i x_i
# theta' = theta / (n*theta + 1)
gamma_posterior <- function(demands){
    c(shape = unname(prior_gamma["shape"] + sum(demands)),
      scale = unname(prior_gamma["scale"] / (length(demands) * prior_gamma["scale"] + 1))
      )
}

# true demand poisson parameter
lambda_true <- 10

# true configuration of the problem
# the only element that is assumed to be unknown
# are the demand transition probabilities
inventory_true <- inventory_params(lambda_true)

# initial distribution p0
init_dist <- c(1.0, rep(0, inventory_true$max_inventory))  
discount <- 0.95                # discount rate

stopifnot(abs(1.0 - sum(inventory_true$demands)) < 1e-6)

# transition samples
samples <- 4                      # number of transition samples per episode
episodes <- 1                     # number of episodes to sample from

# posterior samples
postsamples_train <- 100          # number of posterior training samples
postsamples_test <- 200           # number of posterior test samples

## ------ Construct inventory MDP ------

inventory_mdp <- function(lambda){
    params <- inventory_params(lambda)
    return(mdp_inventory(params))
}

cat("Constructing true MDP ...\n")
mdp_true <- inventory_mdp(lambda_true)

# compute the true value function
sol_true <- solve_mdp(mdp_true, discount, show_progress = FALSE)
vf_true <- arrange(sol_true$valuefunction, idstate)$value 
cat("True optimal return", vf_true %*% init_dist, "policy:", sol_true$policy$idaction, "\n")

## ----- Generate Samples --------------

cat("Generating transition samples ... \n")

# construct a full order policy
full_policy = data.frame(
  idstate = seq(0, inventory_true$max_inventory), 
  idaction = inventory_true$max_order - 1,
  probability = 1.0)

# generate samples from the inventory domain following the optimal policy
transition_samples <- simulate_mdp(mdp_true, 0, full_policy, episodes = episodes, 
                                   horizon = samples, seed = sample.seed)

## ------ Compute the posterior -------------------

cat("True demand:", lambda_true, "\n")

cat("Computing posterior ...\n")
demands <- inventory_true$max_inventory - transition_samples$idstateto
posterior <- gamma_posterior(demands)

if(requireNamespace("ggplot2")){
    cat("Plotting prior and posterior distributions.\n")
    samples_plot <-
        data.frame(
            type = c(rep("prior", 1000),
                     rep("posterior", 1000)),
            value = c(rgamma(1000, shape = prior_gamma["shape"], scale = prior_gamma["scale"]),
                      rgamma(1000, shape = posterior["shape"], scale = posterior["scale"]))
            )

    print(ggplot2::ggplot(samples_plot, ggplot2::aes(x=value,color=type,group=type)) + 
            ggplot2::geom_density())
}


## ----- Generate MDPO from the posterior -----------

cat("Generating posterior samples (may take a minute) ... \n")
lambdas_posterior <- rgamma(postsamples_train + postsamples_test, 
                            shape = posterior["shape"], scale = posterior["scale"])

cat("Cores: ", detectCores(), "\n")

pb <- progress_bar$new(format = "(:spin) [:bar] :percent", 
                       total = length(lambdas_posterior))

# construct the inventory MDP
make_mdp <-  function(l) {
    pb$tick();
    R <- inventory_mdp(lambdas_posterior[l]) 
    R$idoutcome <- l - 1
    return (R)
}

#cluster <- makeCluster(detectCores())
#clusterExport(cluster, varlist = c("lambdas_posterior", "inventory_mdp", "inventory_params",
#                                   "normalize", "sample.seed", "mdp_inventory", "make_mdp", "pb"
#                                   ))

#bayes_MDPs <- parLapply(cluster, seq_along(lambdas_posterior), make_mdp)

bayes_MDPs <- lapply(seq_along(lambdas_posterior), make_mdp)

#stopCluster(cluster)
pb$terminate()
cat("Binding rows .... \n")
mdpo <- bind_rows(bayes_MDPs)

cat("Checking that probabilities sum to one .... \n")
# make sure that all probabilities sum to 1 (select all s,a,o with that do not sum to 1)
invalid <- 
  mdpo %>% group_by(idstatefrom, idaction, idstateto, idoutcome) %>% 
    summarize(sumprob = sum(probability), .groups = "keep") %>%
    filter(sumprob > 1 + 1e-6, sumprob < 1 - 1e-6)

stopifnot(nrow(invalid) == 0)

# split into test and training sets (idoutcome is 0-based)
mdpo_train <- mdpo %>% filter(idoutcome < postsamples_train)
mdpo_test <- mdpo %>% filter(idoutcome >= postsamples_train) %>%
  mutate(idoutcome = idoutcome - postsamples_train ) 

## ------- Save results in the directory ------

cat("Writing results to ", folder_output, " .... \n")
if(!dir.exists(folder_output)) dir.create(folder_output, recursive = TRUE)

state.max <- max(max(mdpo$idstatefrom), max(mdpo$idstateto)) 

initial_df <- data.frame(idstate = seq(0,state.max), probability = init_dist)
parameters_df <- data.frame(parameter = c("discount"), 
                            value = c(discount))

write_csv(initial_df, file.path(folder_output, "initial.csv.xz"))
write_csv(parameters_df, file.path(folder_output, "parameters.csv"))
write_csv(mdp_true, file.path(folder_output, 'true.csv.xz'))

# compression using parallel xz
write_csv(mdpo_train, file.path(folder_output, 'training.csv.xz'))
#cat("  compressing training ... \n")
# pixz corrupts data!
#system2("xz", file.path(folder_output, 'training.csv'))

# compression using parallel xz
write_csv(mdpo_test, file.path(folder_output, 'test.csv.xz'))
#cat("  compressing test ... \n")
# pixz corrupts data!
#system2("xz", file.path(folder_output, 'test.csv'))
