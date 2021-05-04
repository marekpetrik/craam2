#rm(list = ls())
options(mc.cores = parallel::detectCores())

library(Rcpp)
library(rcraam)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rstan)
sourceCpp("cancer_sim.cpp")

rstan_options(auto_write = TRUE)
theme_set(theme_light())

count <- 100

# ---- Generate noise ---- 

gamma_noise <- rgamma_test(0.6)

X <- matrix(seq(0.2, 1.5, length.out = count), nrow = count, ncol = 1)
y <- drop(gamma_noise[1:count] * X)

standata <- list(N = count, K = 1, X = X, y = y)

#inits <- lapply(1:1, function(i){list(true_y = y, sigma = 20, sigma2 = 20, alpha = mean(y),
#      beta = rep(0, ncol(X)), noise = rep(1, nrow(X) )) } )
#      init = inits, 
fit <- stan(file = 'fit_gamma.stan', data = standata, chains = 1, iter = 4000)
print(fit)



