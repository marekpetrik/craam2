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

theme_set(theme_light())

max_capital <- 99     # wins if it goes over
p_win <- 0.1          # probability of winning
discount <- 1.0


solve.gambling <- function(max_capital, p_win, discount){
        
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
    
    # ----------- Solve and Compute Optimal Policy ------
    
    sol <- solve_mdp(mdp.frame, discount, maxresidual = 0.00001)
    qv <- compute_qvalues(mdp.frame, discount, sol$valuefunction)
    v <- sol$valuefunction
    
    # compute actions that are optimal (separate code to determine ties)
    epsilon <- 0.0000001
    opt.policy <- inner_join(qv, v, by=("idstate")) %>% 
        filter(qvalue >= value - epsilon, idaction > 0)
    
    return(list(opt.policy = opt.policy, opt.value = v))
}

sol <- solve.gambling(max_capital, p_win, discount)

policy.plot <- ggplot(sol$opt.policy, aes(x=idstate,y=idaction,fill=1)) + 
                geom_abline(intercept = 0, slope = 1, color="red") +
                theme(legend.position = "none")+
                geom_tile() + labs(x="State",y="Action")

print(policy.plot)
