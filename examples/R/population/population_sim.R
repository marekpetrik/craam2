library(rcraam)
library(dplyr)
library(ggplot2)
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

### ----- Nominal Solution -------------

# solve for the optimal policy

mdp_sol <- solve_mdp(pop.model.mdp, discount, algorithm = "pi", output_tran = TRUE)
#print(mdp_sol$valuefunction)
cat("MDP policy:", mdp_sol$policy$idaction, "\n")

# simulate the model
rpolicy <- mutate(mdp_sol$policy, probability = 1.0) # needs a randomized policy for now

sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1, -1)
print(sim.samples$idstatefrom)
print(discount^sim.samples$step %*% sim.samples$reward)

### ------ Fit a model to the solution ---------------------

sim.samples <- simulate_mdp(pop.model.mdp, init.population, rpolicy, 300, 1, -1)
print(sim.samples$idstatefrom)
#print(discount^sim.samples$step %*% sim.samples$reward)

library(rjags)

pop <- sim.samples$idstatefrom
pop.next <- sim.samples$idstateto
act <- sim.samples$idaction

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

jags <- jags.model(model_spec,
                   data = list(pop = pop,
                               pop_next = pop.next,
                               act = act,
                               N = length(pop),
                               M = length(unique(act))),
                   n.chains=4,
                   n.adapt=100)

# warmup
update(jags, 1000)

post_samples <- jags.samples(jags,
             c('mu', 'mu0',  'mu1', 'mu2', 'sigma', 'ext_mu'),
             1000)

cat("Estimated parameters:")
print(post_samples)

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

# plot the density of the posterior
plt <- ggplot(efficiency_sampled.df, aes(x = Population, y = Rate)) + 
        geom_hex() + 
        geom_line(mapping=aes(x=population, y=rate), data=efficiency_true, color = "red")

print(plt)

#plt2 <- 
# plot the posterior efficacy
ggplot(efficiency_sampled.df %>% filter(Sample %% 50 == 0), 
               aes(x = Population, y = Rate, group = Sample)) + 
  geom_line(linetype = 3) +
  geom_line(mapping=aes(x=population, y=rate), 
            data=efficiency_true, color = "red", 
            inherit.aes = FALSE)
  


### ------ Solve nominal problem ---------------------

# this is very much a model-based solution

# fist action: no control, second action: pesticide
exp.growth.rate <- rbind(rep(2.0, max.population+1), growth.app,
                         growth.app, growth.app, growth.app)
sd.growth.rate <- rbind(rep(0.6, max.population+1), rep(0.6, max.population+1), 
                        rep(0.5, max.population+1), rep(0.4, max.population+1),
                        rep(0.3, max.population+1))

pop.model.mdp <- rcraam::mdp_population(max.population, init.population, 
                                        exp.growth.rate, sd.growth.rate, 
                                        rewards, external.pop, external.pop/2, "logistic")
