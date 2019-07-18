# want to compare the time to run:
#
# method:
# value iteration with gurobi
# value iteration with fast
# ppi with fast optimization
#
# rectangularity:
# sa-rectangular
# s-rectangular
#
# ambiguity :
# weighted
# unweighted
#
# size:
# 0.01
# 0.1
# 0.5
# 1.0
# 2.0

# install from gitlab.com/RLSquared/craam2
library(rcraam)
library(dplyr)
library(stringr)
library(DBI)

# open the connection and create the table if it does not exist
datalog <- dbConnect(RSQLite::SQLite(), "results.sqlite")
if(!("timing" %in% dbListTables(datalog))){
  dbCreateTable(datalog, "timing", c(problem = "text",
                                     states = "int",
                                     rectangularity = "text",
                                     discount = "real",
                                     algorithm = "text",
                                     ambiguity = "text",
                                     budget = "real",
                                     runtime = "real"))
}

timing.table <- tbl(datalog, "timing")

# saves the result of the run to a database
save.result <- function(FUN, problem, budget, discount, states, algorithm){

  alg.parsed <- stringr::str_split(algorithm, "_")[[1]]
  rectangularity <- alg.parsed[1]
  algorithm.p <- alg.parsed[2]
  ambiguity <- paste0(alg.parsed[3],alg.parsed[4])
  
  # decide whether the solution should be computed
  # skip the computation if failed for fewer states or if computed earlier
  matching <- timing.table %>% 
                  filter(problem == !!problem, states == !!states, 
                         algorithm == !!algorithm.p, 
                         rectangularity == !!rectangularity, 
                         discount == !!discount, 
                         ambiguity == !!ambiguity) %>%
                  tbl_df()
  
  # solve it if not solved before or the solution is quick
  if(nrow(matching) == 0 || min(matching$runtime) < 1){
    # TODO: Conditions on real numbers ... that could be a problem
    earlier <- timing.table %>% 
      filter(problem == !!problem, states <= !!states, 
             algorithm == !!algorithm.p, 
             rectangularity == !!rectangularity, 
             discount == !!discount, 
             ambiguity == !!ambiguity) %>%
      tbl_df()
    
    if(nrow(earlier) > 0 && min(earlier$runtime) > 5000){
      cat("Failed for fewer states, skipping ", algorithm, " with ", states, " states ... \n");
      return()
    }else{
      cat("Computing ", algorithm, " with ", states, " states ... \n");
      solution <- FUN()    
    }
  }else{
    cat("Already solved, skipping ", algorithm, " with ", states, " states ... \n");
    return()
  }
  
  if(solution$status != 0){
    cat("Failed to solve ", algorithm, " with ", states, " states ... \n");
    runtime <- 1000000
  }else{
    cat("Saving ", algorithm, " with ", states, " states ... \n");
    runtime <- solution$time
  }
  
  newrow <- data.frame(problem = problem, states = states, budget = budget, discount = discount, 
                       rectangularity = rectangularity, algorithm = algorithm.p, 
                       ambiguity = ambiguity, runtime = runtime, stringsAsFactors = FALSE) 
  
  dbAppendTable(datalog, "timing", newrow)   
  return ()
}

budget <- 0.1
config <- list(iterations = 50000, progress = FALSE, timeout = 300, precision = 0.1)
discount <- 0.995

for(actions in seq(2, 100, by = 2)){
  #actions <- 45
  states <- 3 * actions
  cat("Running with ", states, " states...\n")
  
  # construct an inventory management problem with the given number of actions
  problem.name <- "inventory"
  inv <- inventory.default()
  inv$max_order <- actions
  inv$max_inventory <- 2 * inv$max_order
  inv$max_backlog <- inv$max_order
  mdp <- mdp_inventory(inv)
  
  # this would create a smaller test mdp example
  #mdp <- mdp_example("")
  
  
  report <- function(FUN, algorithm){
    save.result(FUN, problem.name, budget, discount, 3*actions, algorithm)
  }
  
  # solve a regular MDP
  solmdp <- solve_mdp(mdp, discount,append(config, list(algorithm="pi")))
  report(function(){solmdp}, "na_mpi_na_na")
  
  # **** sa-rectangular, no weights
  budgets_sa <- mdp %>% select(idstate=idstatefrom, idaction) %>% 
                        unique() %>% mutate(value=budget)
  
  report(function(){rsolve_mdp_sa(mdp, discount, "l1_g", budgets_sa, 
                           append(config, list(algorithm="vi_j")))},
    "sa_vi_l1g_nw")
  report(function(){rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, 
                           append(config, list(algorithm="vi_j")))},
    "sa_vi_l1_nw")
  report(function(){rsolve_mdp_sa(mdp, discount, "l1_g", budgets_sa, 
                           append(config, list(algorithm="ppi")))},
    "sa_ppi_l1g_nw")
  report(function(){rsolve_mdp_sa(mdp, discount, "l1", budgets_sa, 
                           append(config, list(algorithm="ppi")))},
    "sa_ppi_l1_nw")
  
  # **** s-rectangular, no weights
  budgets_s <- mdp %>% select(idstate=idstatefrom) %>% 
                unique() %>% mutate(value=budget)
  
  report(function(){rsolve_mdp_s(mdp, discount, "l1_g", budgets_s, 
                          append(config, list(algorithm="vi_j")))},
    "s_vi_l1g_nw")
  report(function(){rsolve_mdp_s(mdp, discount, "l1", budgets_s, 
                          append(config, list(algorithm="vi_j")))},
    "s_vi_l1_nw")
  report(function(){rsolve_mdp_s(mdp, discount, "l1_g", budgets_s, 
                          append(config, list(algorithm="ppi")))},
    "s_ppi_l1g_nw")
  report(function(){rsolve_mdp_s(mdp, discount, "l1", budgets_s, 
                          append(config, list(algorithm="ppi")))},
    "s_ppi_l1_nw")
  
  # **** weights
  weights <- abs(solmdp$valuefunction - mean(solmdp$valuefunction)) + 10 # 10 is regularization
  weights <- weights / max(weights)
  
  # construct a dataframe that has the weight to the destination states
  weights.df <- data.frame(mdp %>% select(idstateto) %>% unique(),
                           weight = weights) %>%
    full_join(mdp, by = "idstateto")   %>%
    select(idstatefrom, idaction, idstateto, value = weight)
  
  # **** sa-rectangular, weights
  report(function(){rsolve_mdp_sa(mdp, discount, "l1w_g", list(budgets = budgets_sa, weights = weights.df), 
                append(config, list(algorithm="vi_j")))},
    "sa_vi_l1g_w")
  report(function(){rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df), 
                append(config, list(algorithm="vi_j")))},
    "sa_vi_l1_w")
  report(function(){rsolve_mdp_sa(mdp, discount, "l1w_g", list(budgets = budgets_sa, weights = weights.df), 
                append(config, list(algorithm="ppi")))},
    "sa_ppi_l1g_w")
  report(function(){rsolve_mdp_sa(mdp, discount, "l1w", list(budgets = budgets_sa, weights = weights.df), 
                append(config, list(algorithm="ppi")))},
    "sa_ppi_l1_w")
  
  # **** s-rectangular, weights
  report(function(){rsolve_mdp_s(mdp, discount, "l1w_g", list(budgets = budgets_s, weights = weights.df), 
               append(config, list(algorithm="vi_j")))},
    "s_vi_l1g_w")
  report(function(){rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df),
               append(config, list(algorithm="vi_j")))},
    "s_vi_l1_w")
  report(function(){rsolve_mdp_s(mdp, discount, "l1w_g", list(budgets = budgets_s, weights = weights.df), 
               append(config, list(algorithm="ppi")))},
    "s_ppi_l1g_w")  
  report(function(){rsolve_mdp_s(mdp, discount, "l1w", list(budgets = budgets_s, weights = weights.df), 
               append(config, list(algorithm="ppi")))},
    "s_ppi_l1_w")
}

dbDisconnect(datalog)

#rsolve_mdp_sa(mdp, discount, "eavaru", list(alpha = 0.0001, beta = 0.1), append(config, list(algorithm="ppi")))$value
