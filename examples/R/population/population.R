library(rcraam)
library(dplyr)
library(ggplot2)
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
sd.growth.rate <- rbind(rep(0.5, max.population+1), rep(0.5,max.population+1), 
                        rep(0.25,max.population+1), rep(0.12,max.population+1))

# rewards decrease with increasing population, and there is an extra penalty
# for applying the pesticide
rewards <- matrix(rep(- seq(0,max.population)^2, actions), nrow=actions, byrow=TRUE)
rewards[2,] <- rewards[2,] - 1600
rewards[3,] <- rewards[3,] - 1700
rewards[4,] <- rewards[4,] - 2000

pop.model.mdp <- rcraam::mdp_population(max.population, init.population, 
                                    exp.growth.rate, sd.growth.rate, 
                                    rewards)

# plots using plot.matrix
#plot(mdp_rsol$mat, col=gray.colors, breaks = seq(0,0.3,by=0.01) )

### ***** Optimal non-robust solution *************

# solve for the optimal policy

mdp_sol <- solve_mdp(pop.model.mdp, discount, list(algorithm = "pi", 
                                                   output_tran = TRUE))
dpolicy <- mdp_sol$policy
print(mdp_sol$valuefunction)
print(dpolicy$idaction)

# simulate the model
rpolicy <- mutate(dpolicy, value = 1.0) # needs a randomized policy for now

sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1)
#print(sim.samples$idstatefrom)
print(discount^sim.samples$step %*% sim.samples$reward)

### **** Robust evaluation of the nonrobust policy ****************

mdp_rsol <- rsolve_mdp_sa(pop.model.mdp, discount, "l1u", l1error, list(policy = mdp_sol$policy,
                                                                    algorithm = "ppi",
                                                                    output_tran = TRUE))

### ***** Robust policy *************
# test a robust value function
rmdp_rsol <- rsolve_mdp_sa(pop.model.mdp, discount, "l1u", l1error)
dpolicy <- rmdp_rsol$policy
print(rmdp_rsol$policy$idaction)
print(rmdp_rsol$valuefunction)

# simulate the model
rpolicy <- mutate(dpolicy, value = 1.0) # needs a randomized policy for now

sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1)
#print(sim.samples$idstatefrom)
print(discount^sim.samples$step %*% sim.samples$reward)

### ***** Robust policy, exp evaluation *************

rmdp_sol <- solve_mdp(pop.model.mdp, discount, list(policy = rmdp_rsol$policy, 
                                                    algorithm = "pi"))

### ***** Plot Values *************

# value functions with expected probabilities
value.functions <- rbind(
      data.frame(state = seq(0,max.population), method = "Exp", 
                 value = mdp_sol$valuefunction),
      data.frame(state = seq(0,max.population), method = "Rob", 
                 value = rmdp_sol$valuefunction))

# value functions with robust probabilities
rvalue.functions <- rbind(
  data.frame(state = seq(0,max.population), method = "Exp", 
             value = mdp_rsol$valuefunction),
  data.frame(state = seq(0,max.population), method = "Rob", 
             value = rmdp_rsol$valuefunction))

print(ggplot(value.functions %>% filter(state > 0), aes(x=state, y=value, color=method)) + 
        geom_line() + ggtitle("Expected Value"))
print(ggplot(rvalue.functions %>% filter(state > 0), aes(x=state, y=value, color=method)) + 
        geom_line() + ggtitle("Robust Value"))
