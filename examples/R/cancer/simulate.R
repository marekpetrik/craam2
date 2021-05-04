library(Rcpp)
library(rcraam)
library(ggplot2)
library(dplyr)
library(tidyr)

rm(list=ls())

theme_set(theme_light())
sourceCpp("cancer_sim.cpp")

# --- Params ----
state_count <- 2000
discount <- 0.9
sim_runs <- 1000

def_config <- default_config()
def_config$noise <- "gamma"
def_config$noise_std <- 0.2
def_config$dose_penalty <- 1.6

# --- Initialize -----

state <- init_state()

#print(cancer_transition(state, FALSE,  def_config))
#print(cancer_transition(state, TRUE, def_config))

# *** generate a bunch of random states (to serve as centers of the aggregate states) ***

cat("Gathering samples ... \n")
samples_all <- simulate_random(def_config, state_count, 200)

#plot distributions from random policy

plot_density <- function() {
    ggplot(pivot_longer(samples_all$states_from, c(C,P,Q,Q_p), "predictor"), 
           aes(x = value)) + stat_density() + facet_wrap(vars(predictor)) +
            lims( x = c(-10,10))
}

samples_rep <- sample_n(samples_all$states_from, state_count)
samples_rep_state <- mutate(samples_rep, idstate = row_number())

# assumption: the first sample is the initial state
# # note: the state indexes are 1-based because state 0 is assumed to be the terminal state
init_state_num <- as.integer(class::knn1(samples_rep, samples_all$states_from[1,], 1:nrow(samples_rep)))

# *** construct the MDP of the simulator

cat("Building the MDP ... \n")
mdp <- cancer_mdp(def_config, samples_rep, 1000, TRUE)
cat("MDP building complete. \n")
max_state <- max(max(mdp$idstateto), max(mdp$idstatefrom))

cat("*******\n")
cat("\nPredicted return: \n")
sol <- solve_mdp(mdp, discount, show_progress = 0)
# sol <- rsolve_mdp_sa(mdp, discount, nature = "l1u", nature_par = 0.05)
ret_value <- sol$valuefunction %>% filter(idstate == init_state_num)
cat("Optimized: ", ret_value$value, "\n")

policy_never <- data.frame(idstate = seq(0, max_state), idaction = 0)
sol_never <- solve_mdp(mdp, discount, show_progress = FALSE, policy_fixed = policy_never)
cat("Never: ", (sol_never$valuefunction %>% filter(idstate == init_state_num))$value, "\n" )

policy_always <- data.frame(idstate = seq(0, max_state), idaction = 1)
sol_always <- solve_mdp(mdp, discount, show_progress = FALSE, policy_fixed = policy_always)
cat("Always: ", (sol_always$valuefunction %>% filter(idstate == init_state_num))$value, "\n" )
cat("*******\n\n")

# **** Simulate the return *****

# join state features and actions and strips the terminal state idstate = 0
state_policy <- inner_join(sol$policy, samples_rep_state, by = "idstate") %>%
    mutate(idaction = idaction) %>% filter(idaction >= 0)

# simulates the policy
simulated_policy <- simulate_proximity(def_config, select(state_policy, C, P, Q, Q_p), 
                   as.integer(state_policy$idaction), sim_runs, 200)

simulated_random <- simulate_random(def_config, sim_runs, 200)
simulated_never <- simulate_trivial(def_config, 0, sim_runs, 200)
simulated_always <- simulate_trivial(def_config, 1, sim_runs, 200)

compute_return <- function(simulation){
    sum(simulation$rewards * discount^simulation$steps) / simulation$runs
}

cat("*******\n")
cat("Simulated returns:\n")
cat("Optimized: ", compute_return(simulated_policy), "\n")
cat("Random: ", compute_return(simulated_random), "\n")
cat("Never: ", compute_return(simulated_never), "\n")
cat("Always: ", compute_return(simulated_always), "\n")
cat("*******\n\n")

library(C50)
library(rpart)
library(rpart.plot)

rpart_c <- rpart(idaction ~ C + P + Q + Q_p, data = mutate(state_policy, idaction = as.factor(idaction)))
#print(tidyrules::tidyRules(rpart_c))
prp(rpart_c)

# ---- Generate simulated data -------

sim_data <- with(simulated_policy, {
  cbind(
      states_from %>% rename_all(function(x){paste0("from_",x)}),
      states_to %>% rename_all(function(x){paste0("to_",x)}),
      idaction = actions)
})

# ---- Fit linear regression --------

cat("*******\n")
cat("Linear model fitting statistics: \n")
for(idaction_fit in c(0,1)){
    sim_data2 <- filter(sim_data, idaction == idaction_fit)
    lr <- lm(to_P ~ from_C + from_P + from_Q + from_Q_p, data = sim_data2)
    cat("Action", idaction_fit, "P R2:", summary(lr)$r.squared, "\n")
    lr <- lm(to_Q ~ from_C + from_P + from_Q + from_Q_p, data = sim_data2)
    cat("Action", idaction_fit, "Q R2:", summary(lr)$r.squared, "\n")
    lr <- lm(to_Q_p ~ from_C + from_P + from_Q + from_Q_p, data = sim_data2)
    cat("Action", idaction_fit, "Q_p R2:", summary(lr)$r.squared, "\n")
    # This gives a funny R2 just because TSS is so small
    #lr <- lm(to_C ~ from_C + from_P + from_Q + from_Q_p, data = sim_data2)
    #cat("Action", idaction_fit, "C R2:", summary(lr)$r.squared, "\n")
}
cat("*******\n\n")

# ------ Fit a Stan model --------

cat("*******\n")
cat("Stan results:\n")

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

X <- model.matrix(to_P ~ from_C + from_P + from_Q + from_Q_p,
             data = sim_data %>% filter(idaction == 1, to_P > 0.1))
y <- (sim_data %>% filter(idaction == 1, to_P > 0.1) )$to_P

X <- X[1:200,]
y <- y[1:200]

standata <- list( N = nrow(X), K = ncol(X), X = X, y = y)

fit <- stan(file = 'fit_gamma.stan', data = standata, chains = 1, iter = 2000)
#print(fit)
#plot(fit, pars = c('alpha'))
#plot(fit, pars = c('w'))

alpha_samples <- extract(fit, "alpha")
w_samples <- extract(fit, "w")

# the gamma standard deviation is sqrt(1/alpha)
cat("True gamma sd:", def_config$noise_std, 
    "predicted gamma sd:", mean(sqrt(1/alpha_samples$alpha)), "\n" )
cat("*******\n")