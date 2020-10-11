library(rcraam)
library(dplyr)
library(ggplot2)
library(setRNG)
library(readr)
library(rjags)
library(runjags)
library(ggrepel) # labels that avoid data points
library(forcats) # rename plot labels
library(extrafont) # latex font output in plots
font_install("fontcm") # install computer modern font
loadfonts()

theme_set(theme_light(base_family = "CM Roman"))

error_size <- 0.1
error_type <- "l1u"
extra=FALSE

### ----- Problem definition -------------
max.population <- 50
init.population <- 10
actions <- 5
discount <- 0.9
plot.breaks <- seq(0,0.4,by=0.005)
show.plots <- FALSE
external.pop <- 3
init.dist <- c(1,rep(0, 50))
confidence <- 0.7
post_sample_count <- 100

risk_weights <- seq(0, 1, length.out = 5)
seed=1
set.seed(seed)

corr <- (seq(0,max.population) - (max.population/2))^2 
growth.app <- 0.2 + corr / max(corr) 

# first action: no control, second action: pesticide
exp.growth.rate <- rbind(rep(2.0, max.population+1), growth.app,
                         growth.app, growth.app, growth.app)
sd.growth.rate <- rbind(rep(0.6, max.population+1), rep(0.6, max.population+1), 
                        rep(0.5, max.population+1), rep(0.4, max.population+1),
                        rep(0.3, max.population+1))

# rewards decrease with increasing population, and there is an extra penalty
# for applying the pesticide
rewards <- matrix(rep(- seq(0,max.population)^2, actions), nrow=actions, byrow=TRUE)
rewards <- rewards + 1000 # add harvest return
spray.cost <- 800
rewards[2,] <- rewards[2,] - spray.cost
rewards[3,] <- rewards[3,] - spray.cost * 1.05
rewards[4,] <- rewards[4,] - spray.cost * 1.10
rewards[5,] <- rewards[5,] - spray.cost * 1.15

pop.model.mdp <- rcraam::mdp_population(max.population, init.population, 
                                        exp.growth.rate, sd.growth.rate, 
                                        rewards, external.pop, external.pop/2, "logistic")
### ----- Sample from an optimal policy -------------

# solve for the optimal policy
mdp_sol <- solve_mdp(pop.model.mdp, discount,show_progress=FALSE)
cat("MDP policy:", mdp_sol$policy$idaction, "\n")

# simulate the model
rpolicy <- mutate(mdp_sol$policy, probability = 1.0) # needs a randomized policy for now

set.seed(1)
sim.samples <- simulate_mdp(pop.model.mdp,0,rpolicy,100,5,seed=seed)

### ------ Fit a model to the solution ---------------------

# IMPORTANT: This fits an exponential model, while 
#  the baseline model is logistic; it does not make much
# of a difference as long as the pest populations remain
# low

set.seed(1)
sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1,seed=seed)

pop <- sim.samples$idstatefrom
pop.next <- sim.samples$idstateto
act <- sim.samples$idaction


# check if there is a cached posterior dataset
post_samples <- try({readRDS("postsamples.rds")})
# construct the posterior samples if there is no cache
if(class(post_samples) == "try-error"){
    cat("Unable to load JAGS samples, sampling ... \n")
    model_str <-"
    model {
      for (i in 1:N){
        ext[i] ~ dnorm(ext_mu, ext_std)
        next_mean[i] <- ifelse(act[i] == 0, 
                      mu * pop[i],
                      mu0 * pop[i] + mu1 * pop[i]^2 + mu2 * pop[i]^3)
        pop_next[i] ~ dnorm(next_mean[i] + ext[i], sigma[act[i] + 1])
      }
      for (j in 1:M){
        sigma[j] ~ dunif(0,5)
      }
      mu ~ dunif(0.5, 3)
      mu0 ~ dnorm(1.0, 1)
      mu1 ~ dnorm(0.0, 10)
      mu2 ~ dnorm(0.0, 10)
      ext_mu ~ dunif(0, 20)
      ext_std ~ dunif(0, 5) 
    }
    "
    model_spec <- textConnection(model_str)
    pop.jags <- jags.model(model_spec,
                       data = list(pop = pop, pop_next = pop.next, act = act,
                                   N = length(pop), M = dim(exp.growth.rate)[1]),
                       n.chains=8,
                       n.adapt=5000)
    
    # warmup
    update(pop.jags, 20000)
    set.seed(seed)
    # QUESTION: Is thinning helpful here? This is different from 
    # uses in which we only care about the statistics
    post_samples <- jags.samples(pop.jags, c('mu', 'mu0',  'mu1', 'mu2'), round(100/8*post_sample_count), 100)
    #post_samples <- autorun.jags(jags)
    #set.seed(seed)
    saveRDS(post_samples, file="postsamples.rds")
    
    ### ------ Plot the true vs estimated effectiveness --------
    
    population_range <- seq(0,max.population)
    efficiency_true <- data.frame(population = population_range, type = "true", 
                           rate = growth.app)
    
    # returns the function that estimates the pest growth 
    # after action1 is applied
    efficiency_sampled <- matrix(0, nrow = dim(post_samples$mu0)[2] * dim(post_samples$mu0)[3],
                                    ncol = length(population_range))
    k <- 1
    for(i in 1:dim(post_samples$mu0)[2]){
      for(j in 1:dim(post_samples$mu0)[3]){
        efficiency_sampled[k,] <- post_samples$mu0[1,i,j] + 
                post_samples$mu1[1,i,j] * population_range   + 
                post_samples$mu2[1,i,j] * population_range^2
        k <- k + 1
      }
    }
    
    efficiency_sampled.df <- reshape2::melt(efficiency_sampled, value.name = "Rate")
    colnames(efficiency_sampled.df) <- c("Sample", "Population", "Rate")
    efficiency_sampled.df$Population <- efficiency_sampled.df$Population - 1
    
    # density plot
    ggplot(efficiency_sampled.df, aes(x = Population, y = Rate)) + 
            geom_hex() + 
            geom_line(mapping=aes(x=population, y=rate), data=efficiency_true, color = "red")
    
    # plot example responses
    ggplot(efficiency_sampled.df %>% filter(Sample %% 10 == 0), 
                   aes(x = Population, y = Rate, group = Sample)) + 
      geom_line(linetype = 3) +
      geom_line(mapping=aes(x=population, y=rate), 
                data=efficiency_true, color = "red", 
                inherit.aes = FALSE)
} else {
    cat("Posterior samples loaded from a file. ")
}

### ------ Plot the true vs estimated effectiveness --------

population_range <- seq(0,max.population)
efficiency_true <- data.frame(population = population_range, type = "true", 
                       rate = growth.app)

# returns the function that estimates the pest growth 
# after action1 is applied
efficiency_sampled <- matrix(0, nrow = dim(post_samples$mu0)[2] * dim(post_samples$mu0)[3],
                                ncol = length(population_range))

### ------ Formulate Bayesian MDP ---------------------

mdp.bayesian <- try(read_csv("population_bayes_model.csv"))

if ("try-error" %in% class(mdp.bayesian)) {
    # TODO: This is wrong: this is the true model, which should not be here
    #mdp.bayesian <- rcraam::mdp_population(max.population, init.population,
    #                                        exp.growth.rate, sd.growth.rate,
    #                                        rewards, external.pop, external.pop/2, "logistic")
    #mdp.bayesian['idoutcome'] <- 0
    
    
    mdp.bayesian <- NULL
    idoutcome <- 0
    #TODO: This code assumes that the variance of the different actions is the same
    # which is a limitation, since that is the only/main difference between the different
    # control actions
    for (iteration in 1:dim(post_samples$mu)[2]) { 
        cat(".")
        # create an empty matrix of rates
        rates <- matrix(0, ncol = dim(exp.growth.rate)[2],
                           nrow = dim(exp.growth.rate)[1])
        # TODO: enable this one
        for (chain in 1:dim(post_samples$mu)[3]) {     # loop over chains 
            rates[1,] <- post_samples$mu[1,iteration,chain]
    		    rates[2:dim(rates)[1],] <-
    		                 post_samples$mu0[1,iteration,chain] + 
                		     post_samples$mu1[1,iteration,chain] * population_range + 
                		     post_samples$mu2[1,iteration,chain] * population_range^2;
    		    
            pop.model.mdp <- rcraam::mdp_population(max.population, init.population,
                                                  rates, sd.growth.rate, rewards, 
                                                  external.pop, external.pop/2, "exponential")
            pop.model.mdp['idoutcome'] <- idoutcome
            mdp.bayesian <- rbind(mdp.bayesian,pop.model.mdp)
            idoutcome <- idoutcome + 1
        }
    }
    cat("\n")
    write_csv(mdp.bayesian, "population_bayes_model.csv")
}


### ------ Helper functions ---------------------
results <- list(
  method = c(), risk_weight = c(), predicted = c(),
  expected = c(), var = c(), avar = c()
)

rmdp.bayesian <- function(mdp.bayesian, confidence) {
  # adjust the confidence level

  # provides a bound on the value function and the return
  sa.count <- nrow(unique(mdp.bayesian %>% select(idstatefrom, idaction)))
  confidence.rect <- (1 - confidence) / sa.count

  # construct the mean bayesian model
  mdp.mean.bayes <- mdp.bayesian %>%
    group_by(idstatefrom, idaction, idstateto) %>%
    summarize(probability = mean(probability), reward = mean(reward))
  mean.probs <- mdp.mean.bayes %>%
    rename(probability_mean = probability) %>%
    select(-reward)

  # compute L1 distances
  budgets <-
    inner_join(mean.probs,
	        mdp.bayesian,
      by = c("idstatefrom", "idaction", "idstateto")
    ) %>%
    mutate(diff = abs(probability - probability_mean)) %>%
    group_by(idstatefrom, idaction, idoutcome) %>%
    summarize(l1 = sum(diff)) %>%
    group_by(idstatefrom, idaction) %>%
    summarize(budget = quantile(l1, 1 - confidence.rect)) %>%
    rename(idstate = idstatefrom)

  return(list(
    mdp.mean = mdp.mean.bayes,
    budgets = budgets
  ))
}

bayes.returns <- function(mdp.bayesian, policy, maxcount = 100) {
  outcomes.unique <- head(unique(mdp.bayesian$idoutcome), maxcount)
  sapply(
    outcomes.unique,
    function(outcome) {
      # decide whether the policy being evaluated is randomized
      if (!("probability" %in% colnames(policy))) {
        sol <- mdp.bayesian %>%
          filter(idoutcome == outcome) %>%
          solve_mdp(discount,
            policy_fixed = policy,
            show_progress = FALSE, algorithm = "pi"
          )
      } else {
        sol <- mdp.bayesian %>%
          filter(idoutcome == outcome) %>%
          solve_mdp_rand(discount,
            policy_fixed = policy,
            show_progress = FALSE, algorithm = "pi"
          )
      }
      sol$valuefunction$value %*% init.dist
    }
  )
}

#' Prints experiment result statistics.
#'
#' It also prints its guarantees, solution quality and
#' posterior expectation of how well it is likely to work
#'
#' @param name Name of the algorithm that produced the results
#' @param mdp.bayesian MDP with outcomes representing Bayesian samples
#' @param solution Output from the algorithm's solution
report_solution <- function(name, risk_weight, mdp.bayesian, solution) {
  results$method <<- c(results$method, name)
  results$risk_weight <<- c(results$risk_weight, risk_weight)

  # this is to handle solutions that do not have value functions
  predicted <- ifelse("valuefunction" %in% ls(solution), 
                      solution$valuefunction$value %*% init.dist,
                      solution$objective)
  results$predicted <<- c(results$predicted, predicted)
  
  posterior.returns <- bayes.returns(mdp.bayesian, solution$policy)
  dst <- rep(1 / length(posterior.returns), length(posterior.returns))
  results$expected <<- c(results$expected, mean(posterior.returns))

  results$var <<- c(results$var, quantile(posterior.returns, 1 - confidence))
  results$avar <<- c(results$avar, avar(posterior.returns, dst, 1 - confidence)$value)
  results$policy <- c(results$policy, toString(solution$policy))

  cat("*****", name, "*****", "\n")
  cat("Mean:", mean(posterior.returns), "\n")
  cat("CVAR:", avar(posterior.returns, dst, 1 - confidence)$value, "\n")
}

normalize_transition_probs <- function(mdp){
  mdp %>% group_by(idstatefrom, idaction) %>% 
    summarize(psum = sum(probability)) %>% 
    inner_join(mdp, by = c('idstatefrom', 'idaction')) %>% 
    mutate(probability = probability / psum) %>% select(-psum)
}

## ---- Bayesian Credible Region -----

cat("BCR\n")
model.bayes.loc <- rmdp.bayesian(mdp.bayesian, confidence) 
#TODO: Why is the normalization even needed?
model.bayes.loc$mdp.mean <- normalize_transition_probs(model.bayes.loc$mdp.mean)
for (risk_weight in seq(0.0, 1.4, by = 0.1)) {
  sol.bcr <- rsolve_mdp_sa(
    model.bayes.loc$mdp.mean, discount, "l1",
    model.bayes.loc$budgets %>% mutate(budget = risk_weight * budget),
    show_progress = FALSE
  )
  report_solution("BCR-l", risk_weight, mdp.bayesian, sol.bcr)  
}


## ---- RSVF-like (simplified) -------

sa.count <- nrow(unique(mdp.bayesian %>% select(idstatefrom, idaction)))
confidence.rect <- (1 - confidence) / sa.count
# alpha > 0.5 with VaR makes no sense because it becomes risk seeking
for (risk_weight in seq(2*confidence.rect, 1, length.out = 10)) {
  sol.rsvf <- rsolve_mdpo_sa(mdp.bayesian, discount, "evaru",
    list(alpha = confidence.rect / risk_weight, beta = 1.0),
    show_progress = FALSE)
  report_solution("RSVF", risk_weight, mdp.bayesian, sol.rsvf)
}

## ---- NORBU ----------

for (risk_weight in risk_weights) {
  cat("Norbu", risk_weight, "\n")
  sol.norbu <- rsolve_mdpo_sa(mdp.bayesian, discount, "eavaru",
    list(alpha = 1 - confidence, beta = risk_weight),
    show_progress = FALSE
  )
  report_solution("sa-NORBU", risk_weight, mdp.bayesian, sol.norbu)
}

## ---- s-NORBU ---------

if(FALSE){
  for (risk_weight in risk_weights) {
    cat("s-NORBU", risk_weight, "\n")
    sol.norbu.s <- rsolve_mdpo_s(mdp.bayesian, discount, "eavaru",
      list(alpha = 1 - confidence, beta = risk_weight), show_progress = 0, 
      algorithm = "vi")
    report_solution("s-NORBU", risk_weight, mdp.bayesian, sol.norbu.s)
  }
}

## ---- NORBU VaR version ----------
if(FALSE){
  for (risk_weight in risk_weights) {
    cat("Norbu-v", risk_weight, "\n")
    sol.norbu.w <- rsolve_mdpo_sa(mdp.bayesian, discount, "evaru",
      list(alpha = 1 - confidence, beta = risk_weight),
      show_progress = FALSE)
    report_solution("NORBU-v", risk_weight, mdp.bayesian, sol.norbu.w)
  }
}

## ------ TORBU MILP version ----------

# debugging: output_filename = "/tmp/torbu.lp"
# or output_filename = "/tmp/torbu.mps"

gurobi_set_param("OutputFlag", "1")
gurobi_set_param("LogFile", "/tmp/gurobi.log")
gurobi_set_param("LogToConsole", "1");
gurobi_set_param("ConcurrentMIP", "3");
gurobi_set_param("MIPGap", "0.05");
gurobi_set_param("TimeLimit", "1000")

init.dist.df <- data.frame(idstate = seq(0,length(init.dist)-1),
                           probability = init.dist)

for (risk_weight in c(0,0.2,0.4,0.6,0.8,1.0)) {
  sol.torbu.milp <- srsolve_mdpo(mdp.bayesian , 
                                 init.dist.df, discount, 
                                 alpha = 1 - confidence, beta = risk_weight)

  report_solution("sa-TORBU", risk_weight, mdp.bayesian, sol.torbu.milp)
}


#print(sol.torbu.milp$policy)


## -------- Report results ------------

#x <- "test_population.csv"
#write.csv2(results,x)
results.df <- as.data.frame(results)
results.df$method <- factor(results$method, 
                            levels=c("sa-TORBU", "s-NORBU", "sa-NORBU", "BCR-l", "RSVF"))

results.df$method <- fct_recode(results.df$method, "SoftRob-Opt" = "sa-TORBU", "SoftRob-Apr" = "sa-NORBU",
           "Robust-BCR" = "BCR-l", "Robust-RSVF" = "RSVF")

results.df$risk_weight <- round(results$risk_weight,2)
results.df.plot <- results.df %>% 
  filter(risk_weight <= 1) %>%
  arrange(risk_weight) %>% distinct()
plot <- ggplot(results.df.plot, aes(x = avar, y = expected, color = method, shape=method)) +
        geom_point(size = 2) +
        geom_path(linetype="dashed") +
        geom_label_repel(aes(label=risk_weight),
                  point.padding = 0.25,
                  show.legend = FALSE,
                  data = results.df.plot %>% filter(risk_weight <= 0.1 | risk_weight >= 0.91)) +
        labs(x = paste0("Robust Return: CVaR(",confidence,")"), y = "Average Return",
             shape = "Algorithm", color = "Algorithm")
ggsave("population_comparison.pdf", plot, width = 6, height = 4, units = "in")
print(plot)

write_csv(results.df, "results_df.csv")




