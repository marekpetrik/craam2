library(rcraam)
library(dplyr)
library(ggplot2)
library(plot.matrix)
theme_set(theme_light())

error_size <- 0.1
error_type <- "l1u"

### ----- Problem definition -------------
max.population <- 50
init.population <- 10
actions <- 5
discount <- 0.98
plot.breaks <- seq(0,0.4,by=0.005)
show.plots <- FALSE
external.pop <- 3

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

### ***** Nominal Solution *************

# solve for the optimal policy

mdp_sol <- solve_mdp(pop.model.mdp, discount, list(algorithm = "pi", 
                                                   output_tran = TRUE))
#print(mdp_sol$valuefunction)
cat("MDP policy:", mdp_sol$policy$idaction, "\n")

# simulate the model
rpolicy <- mutate(mdp_sol$policy, probability = 1.0) # needs a randomized policy for now

sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1)
print(sim.samples$idstatefrom)
print(discount^sim.samples$step %*% sim.samples$reward)


### ****** Fit a model to the solution *********************

library(rjags)

pop <- sim.samples$idstatefrom
act <- sim.samples$idaction

model1.string <-"
model {
  for (i in 2:N){
    ext[i] ~ dnorm(ext_mu, ext_std)
    pop[i] ~ dnorm(mu[act[i] + 1] * pop[i-1]  + ext[i], sigma[act[i] + 1])
  }
  for (j in 1:M){
    mu[j] ~ dnorm(1.0,0.5)
    sigma[j] ~ dunif(0,5)
  }
  ext_mu ~ dnorm(5, 2)
  ext_std ~ dnorm(2, 2) 
}
"
model1.spec<-textConnection(model1.string)

jags <- jags.model(model1.spec,
                   data = list(pop = pop,
                               act = act,
                               N = length(pop),
                               M = length(unique(act))),
                   n.chains=4,
                   n.adapt=100)

update(jags, 1000)

jags.samples(jags,
             c('mu', 'sigma', 'ext_mu'),
             1000)
