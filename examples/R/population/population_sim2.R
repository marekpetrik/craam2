library(rcraam)
library(dplyr)
library(ggplot2)
theme_set(theme_light())
library(setRNG)

error_size <- 0.1
error_type <- "l1u"

### ----- Problem definition -------------
max.population <- 50
init.population <- 10
actions <- 5
discount <- 0.9
plot.breaks <- seq(0,0.4,by=0.005)
show.plots <- FALSE
external.pop <- 3

seed=1
set.seed(seed)
#setRNG(kind="Wichmann-Hill", seed=c(979,1479,1542), normal.kind="Box-Muller")

corr <- (seq(0,max.population) - (max.population/2))^2 
growth.app <- 0.2 + corr / max(corr) 

# fist action: no control, second action: pesticide
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

### ----- Nominal Solution -------------


# solve for the optimal policy
config <- list(iterations = 50000, progress = FALSE, timeout = 300, precision = 0.1)
#solmdp <- solve_mdp(mdp, discount,append(config, list(algorithm="pi")))
mdp_sol <- solve_mdp(pop.model.mdp, discount,show_progress=FALSE)
cat("done")
cat("MDP policy:", mdp_sol$policy$idaction, "\n")

# simulate the model
rpolicy <- mutate(mdp_sol$policy, probability = 1.0) # needs a randomized policy for now

set.seed(1)
sim.samples <- simulate_mdp(pop.model.mdp,0,rpolicy,100,5,seed=seed)

### ------ Fit a model to the solution ---------------------
set.seed(1)
sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1,seed=seed)
library(rjags)

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
  set.seed(1)
  #jags.seed=1
  model_spec <- textConnection(model_str)
  set.seed(1)
  inits1 <- list(m=1, c=1, precision=1,
  .RNG.name="base::Super-Duper", .RNG.seed=1)
  inits2 <- list(m=1, c=1, precision=1,
  .RNG.name="base::Wichmann-Hill", .RNG.seed=2)
  inits3 <- list(m=1, c=1, precision=1,
  .RNG.name="base::Marsaglia-Multicarry", .RNG.seed=3)
  inits4 <- list(m=1, c=1, precision=1,
  .RNG.name="base::Mersenne-Twister", .RNG.seed=4)
  jags <- jags.model(model_spec,
                     data = list(pop = pop,
                                 pop_next = pop.next,
                                 act = act,
  			       #inits = list(inits1,inits2,inits3,inits4),
                                 N = length(pop),
                                 M = length(unique(act))),
                     n.chains=4,
                     n.adapt=100)
  
  # warmup
  #set.seed(seed)
  update(jags, 1000)
  set.seed(seed)
  post_samples <- jags.samples(jags,
               c('mu', 'mu0',  'mu1', 'mu2', 'sigma', 'ext_mu'),
               250)
  #set.seed(seed)
  saveRDS(post_samples, file="postsamples.rds")
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
#k <- 1
#for(i in 1:dim(post_samples$mu0)[2]){
set.seed(1)
#  for(j in 1:dim(post_samples$mu0)[3]){
#    efficiency_sampled[k,] <- post_samples$mu0[1,i,j] + 
#            post_samples$mu1[1,i,j] * population_range   + 
#            post_samples$mu2[1,i,j] * population_range^2
#    k <- k + 1
#  }
#}

#efficiency_sampled.df <- reshape2::melt(efficiency_sampled, value.name = "Rate")
#colnames(efficiency_sampled.df) <- c("Sample", "Population", "Rate")
#efficiency_sampled.df$Population <- efficiency_sampled.df$Population - 1




### ------ Solve nominal problem ---------------------

# this is very much a model-based solution

# fist action: no control, second action: pesticide
exp.growth.rate <- rbind(rep(2.0, max.population+1), growth.app,
                         growth.app, growth.app, growth.app)
sd.growth.rate <- rbind(rep(0.6, max.population+1), rep(0.6, max.population+1), 
                        rep(0.5, max.population+1), rep(0.4, max.population+1),
                        rep(0.3, max.population+1))


mdp.bayesian <- rcraam::mdp_population(max.population, init.population,
                                        exp.growth.rate, sd.growth.rate,
                                        rewards, external.pop, external.pop/2, "logistic")
warnings()
mdp.bayesian['idoutcome']=0
for( model in 1:dim(post_samples$mu)[2]) {
      rates <- matrix(0,ncol=dim(exp.growth.rate)[2],nrow=dim(exp.growth.rate)[1])
      #ext_pop = rnorm(1,post_samples$ext_mu[1,model,1],post_samples$ext_std[model])
      for(i in 1:dim(exp.growth.rate)[1]){
            
	   for(j in 1:dim(exp.growth.rate)[2]){
		if(i==1){
                     rates[i,j] <- post_samples$mu[1,model,i]
		} else {
		     rates[i,j] <- post_samples$mu0[1,model,i-1] * j + post_samples$mu1[1,model,i-1] * j^2 + post_samples$mu2[1,model,i-1] * j^3;
		}
	}

   }

   pop.model.mdp <- rcraam::mdp_population(max.population, init.population,
                                        rates, sd.growth.rate,
                                        rewards, external.pop, external.pop/2, "logistic")
   pop.model.mdp['idoutcome']=model
   mdp.bayesian <- rbind(mdp.bayesian,pop.model.mdp)
}

extra=FALSE

init.dist <- rep(0, 51)
init.dist[1] <- 1
confidence=0.7


risk_weights <- seq(0, 1, length.out = 5)

results <- list(
  method = c(), risk_weight = c(), predicted = c(),
  expected = c(), var = c(), avar = c()
)

#sol.norbu.s <- rsolve_mdpo_s(k, discount, "eavaru",
#    list(alpha = 1 - confidence, beta = 1), show_progress = 0, algorithm = "vi")
  #report_solution("s-NORBU", risk_weight, mdp.bayesian, sol.norbu.s)

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

bayes.returns <- function(mdp.bayesian, policy, maxcount = 100,flag=0) {
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
      k <- init.dist
      if (flag==0){
          k <- state0_dist
      }
      else if(flag==1){
        k <- state1_dist
      }
      sol$valuefunction$value %*% k

    }
  )
}

#' Prints experiment result statistics.
#'
#' It also prints its guarantees, solution quality and
#' posterior expectation of how well it is likely to work
#'
#' @param name Name of the algorithm that produced the results
#' @param mdp.bayesian MDP with outcomes representing bayesian samples
#' @param solution Output from the algorithm's solution
report_solution <- function(name, risk_weight, mdp.bayesian, solution) {
  results$method <<- c(results$method, name)
  results$risk_weight <<- c(results$risk_weight, risk_weight)
  results$predicted <<- c(results$predicted, solution$valuefunction$value %*% init.dist)

  posterior.returns <- bayes.returns(mdp.bayesian, solution$policy,flag=2)
  posterior.returns0 <- bayes.returns(mdp.bayesian, solution$policy,flag=2)
  posterior.returns1 <- bayes.returns(mdp.bayesian, solution$policy,flag=2)
  dst <- rep(1 / length(posterior.returns), length(posterior.returns))
  results$expected <<- c(results$expected, mean(posterior.returns))

  results$var <<- c(results$var, quantile(posterior.returns, 1 - confidence))
  results$avar <<- c(results$avar, avar(posterior.returns, dst, 1 - confidence)$value)
  results$policy <<- c(results$policy, toString(solution$policy))
  results$returns0 <<- c(results$returns0, mean(posterior.returns0))
  results$returns1 <<- c(results$returns1, mean(posterior.returns1))
  results$valuefunction <<- c(results$valuefunction, toString(solution$valuefunction.value))

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
#TODO: Whys is this even needed?
model.bayes.loc$mdp.mean <- normalize_transition_probs(model.bayes.loc$mdp.mean)
sol.bcr <- rsolve_mdp_sa(model.bayes.loc$mdp.mean, discount, "l1",
  model.bayes.loc$budgets,
  show_progress = FALSE
)
report_solution("BCR-l", 1, mdp.bayesian, sol.bcr)

## ---- RSVF-like (simplified) -------

sa.count <- nrow(unique(mdp.bayesian %>% select(idstatefrom, idaction)))
confidence.rect <- (1 - confidence) / sa.count
sol.rsvf <- rsolve_mdpo_sa(mdp.bayesian, discount, "evaru",
  list(alpha = confidence.rect, beta = 1.0),
  show_progress = FALSE
)
report_solution("RSVF", 1.0, mdp.bayesian, sol.rsvf)


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
      list(alpha = 1 - confidence, beta = risk_weight), show_progress = 0, algorithm = "vi")
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

gurobi_set_param("OutputFlag", "1")
gurobi_set_param("LogFile", "/tmp/gurobi.log")
gurobi_set_param("LogToConsole", "1");

risk_weight = 0.8
sol.torbu.milp <- srsolve_mdpo(mdp.bayesian, init.dist.df, discount, 
                               alpha = 1-confidence, beta = risk_weight)
report_solution("TORBU-m: ", mdp.bayesian, sol.torbu.milp)


## -------- Report results ------------

#x <- "test_population.csv"
#write.csv2(results,x)
results <- as.data.frame(results)
results$method <- factor(results$method, levels=c("s-NORBU", "sa-NORBU", "BCR-l", "RSVF"))
results$risk_weight <- round(results$risk_weight,2)
plot <-
    ggplot(results,
           aes(x = avar, y = expected, color = method)) +
    geom_point() +
    geom_path(linetype="dashed") +
    geom_text(aes(label=risk_weight), position="jitter")
    labs(x = paste0("AVaR(",confidence,")"), y = "Expected")

print(plot)
#x <- paste("","test_population.pdf",sep="")
#ggsave(x, plot, width = 5, height = 3)

#results$method <- factor(results$method, levels=c("s-NORBU", "NORBU"))
# plot <-
#     ggplot(results,
#            aes(x = risk_weight, y = valuefunction, color = method)) +
#     geom_point() +
#     geom_path(linetype="dashed") +
#     labs(x = "Lambda", y = "Soft Robust Value Function")

# print(plot)

# ggsave("exanmple_value_function.pdf", plot, width = 5, height = 3)


