library(rcraam)
library(dplyr)


#' General method that solve the dynamically robust MDP with var or cvar
#'
#' Uses Bayesian credible regions with the size determined using the union bound 
#'
#' @param mdpo MDP with outcomes (dataframe)
#' @param initial initial distribution, dataframe with idstate, probability
#' @param discount discount rate
#' @param risk_type Either "eavaru" (use cvar) or "evaru" (var)
#' @param confidence Confidence in the estimate (1.0) highest confidence
#' @param risk_weight Convex combination of Exp and risk. This is the weight
#'                    put on the risk part
norbu <- function(mdpo, initial, discount, risk_type, confidence, risk_weight){
	stopifnot(is.data.frame(mdpo))
	stopifnot(is.data.frame(initial))
	stopifnot(is.numeric(discount))
	stopifnot(is.character(risk_type))
	stopifnot(is.numeric(confidence))
	stopifnot(is.numeric(risk_weight))

	solution <- rsolve_mdpo_sa(mdpo, discount, risk_type, 
               list(alpha = 1 - confidence, beta = risk_weight), show_progress = FALSE)
	# compute expected return 
	ret <- full_join(solution$valuefunction, initial, by = "idstate" ) %>%
		mutate(pv = probability * value) %>% na.fail()
	
	# return
  list(policy = solution$policy, estimate = sum(ret$pv))
}
