source('utils.R')
source('experiment_helpers.R')
library(rcraam)
library(dplyr)
library(readr)

compare_gurobi <- FALSE

results_list <- list()
# global variables - yuck
states <- 5
bell_iters = 200
time_limit = 500  # in seconds, max runtime for a method


actions <- 2
cat("Running with ", states, " states...\n")
discount = 0.995


# construct an inventory management problem with the given number of actions
problem.name <- "inventory"
inv <- inventory.default()
inv$max_order <- actions
inv$max_inventory <- 2 * inv$max_order
inv$max_backlog <- inv$max_order
mdp <- mdp_inventory(inv)
mdp <- mdp_clean(mdp)

write_csv(mdp, paste0("inventory_spec_",states,".csv"))


# set configuation parameters
budget <- 0.2
budget_s <- 1.2  # this is larger because this is the sum over all actions
discount <- 0.995

# solve a regular MDP
solmdp <- solve_mdp(mdp, discount, algorithm = "pi", show_progress = FALSE)
report(solmdp, "MDP")

# **** sa-rectangular, no weights
budgets_sa <- mdp %>% select(idstate=idstatefrom, idaction) %>%
    unique() %>% mutate(budget=budget)

report(rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, algorithm = "vi_j",
                     iterations = 50000, show_progress=FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_vi_l1_nw")

report(rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, algorithm = "mppi",
                     iterations = 50000, show_progress=FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_mppi_l1_nw")

report(rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, algorithm = "vppi",
                     iterations = 50000, show_progress=FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_vppi_l1_nw")

report(rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, algorithm = "ppi",
                     iterations = 50000, show_progress=FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_ppi_l1_nw")

if(compare_gurobi){
    suppressWarnings(report(rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, algorithm = "vi_j",
                                          iterations = bell_iters, show_progress=FALSE, timeout = time_limit,
                                          maxresidual = 0), "sa_bell_l1_nw"))

    suppressWarnings(report(rsolve_mdp_sa(mdp, discount, "l1_g", budgets_sa, algorithm = "vi_j",
                                          iterations = bell_iters, show_progress=FALSE, timeout = time_limit,
                                          maxresidual = 0), "sa_bell_l1g_nw"))
}

# **** s-rectangular, no weights
cat("\n\n")
budgets_s <- mdp %>% select(idstate=idstatefrom) %>%
    unique() %>% mutate(budget=budget_s)

report(rsolve_mdp_s(mdp, discount, "l1", budgets_s, algorithm = "vi_j",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_vi_l1_nw")

report(rsolve_mdp_s(mdp, discount, "l1", budgets_s, algorithm = "mppi",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_mppi_l1_nw")

report(rsolve_mdp_s(mdp, discount, "l1", budgets_s, algorithm = "vppi",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_vppi_l1_nw")

report(rsolve_mdp_s(mdp, discount, "l1", budgets_s, algorithm = "ppi",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_ppi_l1_nw")

if(compare_gurobi){
    suppressWarnings(report(rsolve_mdp_s(mdp, discount, "l1", budgets_s, algorithm = "vi_j",
                                         iterations = bell_iters, show_progress = FALSE, timeout = time_limit,
                                         maxresidual = 0, ), "s_bell_l1_nw"))

    suppressWarnings(report(rsolve_mdp_s(mdp, discount, "l1_g", budgets_s, algorithm = "vi_j",
                                         iterations = bell_iters, show_progress = FALSE, timeout = time_limit,
                                         maxresidual = 0), "s_bell_l1g_nw"))
}

# **** weights
weights <- abs(solmdp$valuefunction$value -
                   mean(solmdp$valuefunction$value)) + 10 # 10 is regularization
weights <- weights / max(weights)

# construct a dataframe that has the weight to the destination states
weights.df <- data.frame(idstate = solmdp$valuefunction$idstate,
                         weight = weights) %>%
    full_join(mdp, by = c(idstate = "idstateto"))   %>%
    rename(idstateto = idstate) %>%
    select(idstatefrom, idaction, idstateto, weight = weight)

# **** sa-rectangular, weights
cat("\n\n")
report(rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df),
                     algorithm = "vi_j",
                     iterations = 50000, show_progress = FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_vi_l1_w")

report(rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df),
                     algorithm = "mppi",
                     iterations = 50000, show_progress = FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_mppi_l1_w")

report(rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df),
                     algorithm = "vppi",
                     iterations = 50000, show_progress = FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_vppi_l1_w")

report(rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df),
                     algorithm = "ppi",
                     iterations = 50000, show_progress = FALSE, timeout = time_limit,
                     maxresidual = 0.1), "sa_ppi_l1_w")

if(compare_gurobi){
    suppressWarnings(report(rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df), algorithm = "vi_j",
                                          iterations = bell_iters, show_progress=FALSE, timeout = time_limit,
                                          maxresidual = 0, ), "sa_bell_l1_w"))

    suppressWarnings(report(rsolve_mdp_sa(mdp, discount, "l1w_g", list(budgets = budgets_sa, weights = weights.df), algorithm = "vi_j",
                                          iterations = bell_iters, show_progress=FALSE, timeout = time_limit,
                                          maxresidual = 0), "sa_bell_l1g_w"))
}

# **** s-rectangular, weights
cat("\n\n")
report(rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df),
                    algorithm = "vi_j",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_vi_l1_w")

report(rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df),
                    algorithm = "mppi",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_mppi_l1_w")

report(rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df),
                    algorithm = "vppi",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_vppi_l1_w")

report(rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df),
                    algorithm = "ppi",
                    iterations = 50000, show_progress = FALSE, timeout = time_limit,
                    maxresidual = 0.1), "s_ppi_l1_w")

if(compare_gurobi){
    suppressWarnings(report(rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df), algorithm = "vi_j",
                                         iterations = bell_iters, show_progress = FALSE, timeout = time_limit,
                                         maxresidual = 0), "s_bell_l1_w"))

    suppressWarnings(report(rsolve_mdp_s(mdp, discount, "l1w_g", list(budgets = budgets_s, weights = weights.df), algorithm = "vi_j",
                                         iterations = bell_iters, show_progress = FALSE, timeout = time_limit,
                                         maxresidual = 0), "s_bell_l1g_w"))
}



