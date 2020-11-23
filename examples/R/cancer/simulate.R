library(Rcpp)
library(rcraam)
library(ggplot2)
library(dplyr)
library(tidyr)

theme_set(theme_light())

state_count <- 10000

sourceCpp("cancer_sim.cpp")

def_config <- default_config()

def_config$transition_noise <- 2

state <- init_state()

#print(cancer_transition(state, FALSE,  def_config))
#print(cancer_transition(state, TRUE, def_config))

# *** generate a bunch of random states (to serve as centers of the aggregate states) ***

cat("Gathering samples ... \n")
samples_all <- simulate_random(def_config, state_count, 50)

#plot distributions from random policy

print(ggplot(pivot_longer(samples_all$states_from, c(C,P,Q,Q_p), "predictor"), 
       aes(x = value)) + stat_density() + facet_wrap(vars(predictor)) +
        lims( x = c(-10,10)))

samples_rep <- sample_n(samples_all$states_from, state_count)
samples_rep_state <- mutate(samples_rep, idstate = row_number())

# assumption: the first sample is the initial state
# # note: the state indexes are 1-based because state 0 is assumed to be the terminal state
init_state_num <- as.integer(class::knn1(samples_rep, samples_all$states_from[1,], 1:nrow(samples_rep)))

# *** construct the MDP of the simulator

cat("Building the MDP ... \n")
mdp <- cancer_mdp(def_config, samples_rep, 500, TRUE)
cat("MDP building complete. ")
sol <- solve_mdp(mdp, 0.8)

ret_value <- sol$valuefunction %>% filter(idstate == init_state_num)

cat("*******\n")
cat("Computed return: ", ret_value$value, "\n")
cat("*******\n\n")


# print(sol$policy %>% filter(idaction == 0))

# join state features and actions and strips the terminal state idstate = 0
state_policy <- inner_join(sol$policy, samples_rep_state, by = "idstate") %>%
    mutate(idaction = as.factor(idaction))


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
ff()


mdp %>% filter(idstatefrom == init_state_num)

unlist(cancer_transition(samples_rep[init_state_num,], T, def_config)$state)
unlist(cancer_transition(samples_rep[init_state_num,], F, def_config)$state)
