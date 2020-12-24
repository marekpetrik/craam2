library(Rcpp)
library(rcraam)
library(ggplot2)
library(dplyr)
library(tidyr)

theme_set(theme_light())

state_count <- 5000
discount <- 0.8

sourceCpp("cancer_sim.cpp")

def_config <- default_config()

def_config$transition_noise <- 0.0 #8

state <- init_state()

#print(cancer_transition(state, FALSE,  def_config))
#print(cancer_transition(state, TRUE, def_config))

# *** generate a bunch of random states (to serve as centers of the aggregate states) ***

cat("Gathering samples ... \n")
samples_all <- simulate_random(def_config, state_count, 50)

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
sol <- solve_mdp(mdp, discount, show_progress = 0)

ret_value <- sol$valuefunction %>% filter(idstate == init_state_num)

cat("*******\n")
cat("Computed return: ", ret_value$value, "\n")
cat("*******\n\n")

# **** Simulate the return

# join state features and actions and strips the terminal state idstate = 0
state_policy <- inner_join(sol$policy, samples_rep_state, by = "idstate") %>%
    mutate(idaction = idaction) %>% filter(idaction >= 0)

sim_runs <- 1000

simulated_policy <- simulate_proximity(def_config, select(state_policy, C, P, Q, Q_p), 
                   as.integer(state_policy$idaction), sim_runs, 100)

cat("*******\n")
cat("Simulated return: ", sum(simulated_policy$rewards * discount^simulated_policy$steps) / sim_runs, "\n")
cat("*******\n\n")


# print(sol$policy %>% filter(idaction == 0))


ff <- function(){
    library(C50)
    library(tidyrules)
    library(rpart)
    library(rpart.plot)
    
    c5_c <- C5.0(idaction ~ C + P + Q + Q_p, data = state_policy, rules = TRUE)
    print(tidyRules(c5_c))

    rpart_c <- rpart(idaction ~ C + P + Q + Q_p, data = state_policy)
    prp(rpart_c)
}


#mdp %>% filter(idstatefrom == init_state_num)
#unlist(cancer_transition(samples_rep[init_state_num,], T, def_config)$state)
#unlist(cancer_transition(samples_rep[init_state_num,], F, def_config)$state)



sim_data <- with(simulated_policy, {
  cbind(
      states_from %>% rename_all(function(x){paste0("from_",x)}),
      states_to %>% rename_all(function(x){paste0("to_",x)}),
      idaction = actions)
})

cat("Linear model fitting statistics: \n")

lr <- lm(to_P ~ from_C + from_P + from_Q + from_Q_p, 
         data = sim_data %>% filter(idaction == 1))
cat("P R2:", summary(lr)$r.squared, "\n")
cat("Coefficients:", lr$coefficients, "\n\n")

# Coefficients: 0.002791915 0.01800951 0.9654105 -0.01137875 0.0007189314

lr <- lm(to_Q ~ from_C + from_P + from_Q + from_Q_p, 
         data = sim_data %>% filter(idaction == 1))
cat("Q R2:", summary(lr)$r.squared, "\n")
cat("Coefficients:", lr$coefficients, "\n\n")

lr <- lm(to_Q_p ~ from_C + from_P + from_Q + from_Q_p, 
         data = sim_data %>% filter(idaction == 1))
cat("Q_p R2:", summary(lr)$r.squared, "\n")
cat("Coefficients:", lr$coefficients, "\n\n")


stop("all done")

# stan linear model

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

X <- model.matrix(to_P ~ from_C + from_P + from_Q + from_Q_p - 1,
             data = sim_data %>% filter(idaction == 1))
X <- X[1:200,]

y <- (sim_data %>% filter(idaction == 1))$to_P
y <- y[1:200]

standata <- list( N = nrow(X), K = ncol(X), x = X, y = y)

inits <- lapply(1:1, function(i){list(true_y = y, sigma = 20, sigma2 = 20, alpha = mean(y),
      beta = rep(0, ncol(X)), noise = rep(1, nrow(X) )) } )
fit <- stan(file = 'linear_model.stan', data = standata, init = inits, chains = 1)
print(fit)


