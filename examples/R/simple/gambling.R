# A simple example MDP
# Based on Example 4.2 from 
# Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning : An Introduction. 

# The description reads roughly as:
# A gambler has the opportunity to make bets on the outcomes of a sequence of 
# coin flips. If the coin comes up heads, he wins as many dollars as he has 
# staked on that flip; if it is tails, he loses his stake. The game ends when 
# the gambler wins by reaching his goal of $100, or loses by running out 
# of money. On each flip, the gambler must decide what portion of his capital 
# to stake, in integer numbers of dollars. The reward is zero on all transitions 
# except when the goal is reached, when it is 1. The book shows the result for 
# the probability of win p_h = 0.4. The probability of winning is independent of 
# the capital.

# ---- Configuration -------

library(rcraam)
library(ggplot2)
library(dplyr)

max_capital <- 99     # wins if it goes over
p_win <- 0.4          # probability of winning

# ------ Build MDP -------

fmdp <- new.env()
fmdp$idstatefrom <- c()
fmdp$idaction <- c()
fmdp$idstateto <- c()
fmdp$probability <- c()
fmdp$reward <- c()

for(state in seq(0, max_capital)){
    for(action in seq(0, min(state, max_capital - state + 1))){
        
        # winning transition
        
        with(fmdp, {
            idstatefrom <- c(idstatefrom, state)
            idaction <- c(idaction, action)
            idstateto <- c(idstateto, state + action )
            probability <- c(probability, p_win)
            reward <- c(reward, 
                        ifelse(state + action >= max_capital + 1, 1, 0))
        })
        
        # losing transition
        
        with(fmdp, {
            idstatefrom <- c(idstatefrom, state)
            idaction <- c(idaction, action)
            idstateto <- c(idstateto, state - action )
            probability <- c(probability, 1 - p_win)
            reward <- c(reward, 0)
        })
        
    }
}

mdp.frame <- with(fmdp, {data.frame(idstatefrom = idstatefrom,
                                   idaction = idaction,
                                   idstateto = idstateto,
                                   probability = probability,
                                   reward = reward)})

# ----------- Solve ------

sol <- solve_mdp(mdp.frame, 1.0)
print(sol$policy)

qv <- compute_qvalues(mdp.frame, sol$valuefunction, 0.9)
