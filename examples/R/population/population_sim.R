library(rcraam)
library(dplyr)
library(ggplot2)
library(plot.matrix)
theme_set(theme_light())

l1error <- 0.2

### ***** Problem definition *************
max.population <- 50
init.population <- 10
actions <- 4
discount <- 0.95

corr <- (seq(0,max.population) - (max.population/2))^2 
growth.app <- 0.4 + corr / max(corr) 

# fist action: no control, second action: pesticide
exp.growth.rate <- rbind(rep(2.8, max.population+1), growth.app, growth.app, growth.app)
sd.growth.rate <- rbind(rep(0.5, max.population+1), rep(0.5, max.population+1), 
                        rep(0.25, max.population+1), rep(0.12, max.population+1))

# rewards decrease with increasing population, and there is an extra penalty
# for applying the pesticide
rewards <- matrix(rep(- seq(0,max.population)^2, actions), nrow=actions, byrow=TRUE)
rewards[2,] <- rewards[2,] - 1600
rewards[3,] <- rewards[3,] - 1700
rewards[4,] <- rewards[4,] - 2000

pop.model.mdp <- rcraam::mdp_population(max.population, init.population, 
                                        exp.growth.rate, sd.growth.rate, 
                                        rewards, "logistic")

### ***** Nominal Solution *************

# solve for the optimal policy

mdp_sol <- solve_mdp(pop.model.mdp, discount, list(algorithm = "pi", 
                                                   output_tran = TRUE))
#print(mdp_sol$valuefunction)
cat("MDP policy:", mdp_sol$policy$idaction)

# simulate the model
rpolicy <- mutate(mdp_sol$policy, value = 1.0) # needs a randomized policy for now


sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1)
print(sim.samples$idstatefrom)
#print(discount^sim.samples$step %*% sim.samples$reward)

# simulate the model
rpolicy <- mutate(rmdp_rsol$policy, value = 1.0) # needs a randomized policy for now

#sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1)
#print(sim.samples$idstatefrom)
#cat("Robust Return:", discount^sim.samples$step %*% sim.samples$reward)
