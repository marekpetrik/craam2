library(rcraam)
library(dplyr)
library(readr)

mdp <- read_csv("riverswim_mdp.csv", 
                col_types = cols(idstatefrom = 'i',
                                 idaction = 'i',
                                 idstateto = 'i',
                                 probability = 'd',
                                 reward = 'd'))
sol <- solve_mdp(mdp, 0.99)

print(sol)
