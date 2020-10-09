source('utils.R')
source('experiment_helpers.R')
library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)
library("latex2exp")
library(hash)
library(data.table)

loadNamespace("tidyr")
loadNamespace("reshape2")
loadNamespace("stringr")




description <- "./riverswim/riverswim_mdp.csv"
eps <- 10^-9
num_actions = 2 #indices: 1->left, 2->right
num_states = 6
discount <- 0.95
confidence <- 0.95
bayes.samples <- 1000
samples <- 50
sample.seed <- 2011
episodes <- 10
num_samples_from_truth <- 100
weights_uniform <- rep(1, num_states) # Uniform weightes to be used for unweighted case
# initial dist. over states
init.dist <- rep(1/6,6)
#init.dist <- c(0, 1, 1, 0, 0, 0) / 2
stopifnot(length(init.dist) == num_states)
stopifnot(abs(sum(init.dist) - 1) < 1e-3)


config <- list(iterations = 10000, progress = FALSE, timeout = 300, precision = 0.1)

prior <- rep(1, num_states)


## ----- Initialization ------

mdp.truth <- read_csv(description,
                      col_types = cols(idstatefrom = 'i',
                                       idaction = 'i',
                                       idstateto = 'i',
                                       probability = 'd',
                                       reward = 'd'))
rewards.truth <- mdp.truth %>% select(-probability)



# construct a biased policy to prefer going right
# this is to ensure that the "goal" state is sampled
ur.policy = data.frame(idstate = c(seq(0,5), seq(0,5)),
                       idaction = c(rep(0,6), rep(1,6)),
                       probability = c(rep(0.2, 6), rep(0.8, 6)))

# compute the true value function
sol.true <- solve_mdp(mdp.truth, discount, show_progress = FALSE)
vf.true <- sol.true$valuefunction$value
cat("True optimal return", vf.true %*% init.dist, "policy:", sol.true$policy$idaction, "\n\n")



# generate samples from the swimmer domain
simulation <- simulate_mdp(mdp.truth, 0, ur.policy, episodes = episodes,
                           horizon = samples, seed = sample.seed)


baysian.posterior <- baysian.posterior(simulation, rewards.truth, bayes.samples)
mdp.bayesian <- mdpo_bayes(simulation, rewards.truth, bayes.samples)


mdp.freq <- mdp.frequentist(simulation, rewards.truth)


# *** Initialize experiment models and parameters ***
dir.name = "./riverswim/results/"
file.name = "riverswim_results"
file.ext = ".csv"
file.results <- paste(dir.name,file.name,file.ext,sep="")

col.names="confidence,budget,return,method"
write(col.names, file=file.results, append=FALSE)


sampled.model <-
    sample.model.revised(baysian.posterior, mdp.freq, bayes.samples)




li.value.functions <-
  get.value.functions(mdp.freq$mdp.nominal, mdp.bayesian, discount)





















