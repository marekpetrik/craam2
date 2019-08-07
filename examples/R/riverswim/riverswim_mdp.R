library(rcraam)
library(dplyr)
library(readr)

mdp <- read_csv("riverswim_mdp.csv")

solve_mdp(mdp, 0.99)
