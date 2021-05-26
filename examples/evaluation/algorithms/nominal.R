# this script is meant to be run from ../evaluator.R
# see ../evaluator.R for the meaning of what is in this file

library(rcraam)
library(dplyr)

#' Main method
algorithm_main <- function(mdpo, initial, discount){
    solution <- rsolve_mdpo_sa(mdpo, discount, "exp", NULL, show_progress = FALSE)
    ret <- full_join(solution$valuefunction, 
                   initial, by = "idstate" ) %>%
            mutate(pv = probability * value) %>% na.fail()
    # fails here if there is a state missing in the intial 
    # probability distribution
  
    list(policy = solution$policy, estimate = sum(ret$pv))
}
