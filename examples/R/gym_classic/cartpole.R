# requires that python reticulate is installed
# need to run pip install 
library(ggplot2)
library(dplyr)
library(readr)
library(reticulate)
library(splines)

source("discretized.R")

# ------ Program parameters ------

# parameters
plot.fits <- FALSE
standardize.features <- TRUE
discount <- 0.999

# prediction accuracy
spline.count <- 2
state.count <- 2000

# number of synthetic samples
syn.sample.count <- 10000
limits.lower <- c(-1, -3, -0.25, -3.1)
limits.upper <- c(1, 3, 0.25, 3.1)

# features
feature.names <- c("CartPos","CartVelocity","PoleAngle", "PoleVelocity")
action.names <- c(0,1)

# ------ Generate samples ------

code_generate_samples <- "
import random
import gym
env = gym.make('CartPole-v1')
random.seed(2018)

laststate = None
samples = []
for k in range(20):
  env.reset()
  done = False
  for i in range(100):
    #env.render()
    action = env.action_space.sample()
    if i > 0:
      samples.append( (i-1,) + tuple(state) + (action,) + (reward,) )
    # stop only after saving the state
    if done:
      break
    [state,reward,done,info] = env.step(action) # take a random action
env.close()
"

res <- py_run_string(code_generate_samples)
samples <- as.data.frame(t(sapply(res$samples, unlist)))
colnames(samples) <- c("Step", feature.names,"Action","Reward")

# alternatively: load samples
#samples <- read_csv("cartpole.csv")

# ------ Prepare samples -----

# connect samples
samples_step <- samples
for(f in feature.names){
  samples_step[[paste0(f, "N")]] <- lead(samples_step[[f]], 1)  
}
samples_step <- samples_step %>% 
  mutate(StepN=lead(Step,1), Final=!(lead(Step,2) == Step + 1)) %>%
  filter(StepN == Step + 1) %>%
  na.omit()

# double samples by mirroring the left and right sides

samples_mirror <- samples_step 
for(f in feature.names){
  samples_mirror[[f]] <- -samples_mirror[[f]]
  samples_mirror[[paste0(f, "N")]] <- -samples_mirror[[paste0(f, "N")]]
  samples_mirror$Action <- 1 - samples_mirror$Action
}
samples_step <- rbind(samples_step, samples_mirror)

# ------ Fit a linear model -------
fit_data <- function(samples_step){
  actions <- unique(samples_step$Action)
  models <- list()
  for(a in actions){
    as <- paste0("A",a)
    models[[as]] <- list()
    
    for(f in feature.names){
      subset.data <- samples_step %>% filter(Action == !!a)
      formula.fit <- paste(sapply(feature.names[f != feature.names], 
                                  function(x){paste0("ns(", x, ", spline.count)")}), 
                           collapse = "+")
      formula.fit <- paste0(f,"N~", formula.fit)
      models[[as]][[f]] <- lm(formula(formula.fit), data=subset.data)
    }
  }
  return(models)
}

models <- fit_data(samples_step)

# ------ Predict next state -------

#' Predicts the next state transition for many states simultaneously
#' @param states.current A data frame with columns:
#'  CartPos, CartVelocity,PoleAngle,PoleVelocity
#' @param action A single action that is taken for all states
predict.next.many <- function(models, states.current, action){
  action_string <- paste0("A",action)
  result <- list()
  for(f in feature.names){
      result[[f]] <- predict(models[[action_string]][[f]], states.current)
  }
  return(as.data.frame(result))
}

predict.reward <- function(stateaction){
  # state[1] = $cart.pos
  # state[3] = $pole.angle
  ifelse(abs(stateaction[1]) < 2.4 && 
         abs(stateaction[3]) < (12 / 180 * pi),1, 0)
}
# ------ Construct synthetic samples -------

# Build samples of possible states. These samples will be later aggregated with random 
# locations to determine the actual states.

cat("Generating", syn.sample.count, "synthetic samples from the model .... \n")
samples.gen <- 
  do.call(cbind, 
          lapply(1:length(feature.names), function(i){
                 runif(syn.sample.count,  
                       min=limits.lower[i], 
                       max=limits.upper[i])}))
# Construct predicted transition samples 
samples.actions <- do.call(
  rbind, lapply(action.names, 
    function(x){
      cbind(samples.gen, matrix(x,syn.sample.count,1))
    }))

# construct a data frame with all states that initiate transitions
samples.first <- as.data.frame(samples.gen)
colnames(samples.first) <- feature.names
                               
samples.nexts <- as.matrix(
  do.call(rbind,
          lapply(action.names,
                  function(act){
                    predict.next.many(models,samples.first,act)               
                  }) ))

samples.rewards <- apply(samples.actions, 1, predict.reward)

# Select the actual states randomly from the samples. 
# Since the samples are randomly generated in the first place, we can simply take
# the first $n$ to get a random selection.
states <- samples.gen[1:state.count,]

# Compute scales for each direction to be used with the nearest sample identification. 
# WARNING: it is important to use the same scales when the policy is implemented.
if(standardize.features){
  scales <- apply(samples, 2, purrr::compose(max, abs))
  scales <- diag(1/scales[c('CartPos', 'CartVelocity','PoleAngle','PoleVelocity')])  
}else{
  scales <- diag(c(1,1,1,1))  
}

#print(scales)
write_csv(as.data.frame(scales), 'scales.csv')

# Construct a data frame with the samples. Warning! KNN1 returns a factor, needs to be turned 
# to an integer. Also, it is necessary to scale the features, the scale of 
# the velocities is very different from the scale of the position and the angle.

# NOTE: State 0 is considered to be a terminal state

# --------Build MDP from samples ---------------------

# repeat once for each action
state.ids.from <- as.integer(rep(class::knn1(states %*% scales, 
                                             samples.gen %*% scales, 
                                             1:state.count), 2)) - 1
state.ids.to <- as.integer(class::knn1(states %*% scales, 
                                       samples.nexts %*% scales, 
                                       1:state.count)) - 1
samples.frame <- data.frame(idstatefrom = state.ids.from,
                            idaction = samples.actions[,5],
                            idstateto = state.ids.to,
                            reward = samples.rewards)
  
# -------Solve MDP and save policy -------

cat ("Solving MDP\n")
mdp <- rcraam::mdp_from_samples(samples.frame)

write_csv(mdp, "cartpole_mdp.csv")
solution <- rcraam::solve_mdp(mdp, discount, list(algorithm="pi"))
rsolution <- rcraam::rsolve_mdp_sa(mdp, discount, "l1u", 0.1, list(algorithm="vi",
                                                                  iterations = 10000))
rsolution <- rcraam::rsolve_mdp_sa(mdp, discount, "l1u", 0.1, list(algorithm="ppi",
                                                                  iterations = 5000))
#rsolution <- rcraam::rsolve_mdp_sa(mdp, discount, "l1u", 0.1, list(algorithm="ppi"))

qvalues <- rcraam::compute_qvalues(mdp, solution$valuefunction, discount)

# Save solution to a data_frame
states.scaled <- states %*% scales
pol.states <- solution$policy$idstate
pol.actions <- solution$policy$idaction
solution.df <- data.frame(CartPos = states.scaled[pol.states+1,1],
                          CartVelocity = states.scaled[pol.states+1,2],
                          PoleAngle = states.scaled[pol.states+1,3],
                          PoleVelocity = states.scaled[pol.states+1,4],
                          State = pol.states,
                          Action = pol.actions,
                          Value = solution$valuefunction[pol.states+1])
# add probabilities to the file if the policy is randomized
if("probability" %in% colnames(solution$policy)){
  solution.df$Probability <- solution$policy$probability
}

# ------------- Save values -----------------------

write_csv(samples, "samples.csv")
write_csv(solution.df, "policy_nn.csv")
write_csv(qvalues, "qvalues.csv")


# ------ Plot Fits ---------

if(plot.fits){
  ggplot(samples_step, 
         aes(x=CartPos, y=CartPosN, color=as.factor(Action))) + 
    geom_point() + theme_light()
  
  ggplot(samples_step %>% filter(Action==0), 
         aes(x=PoleAngle, y=CartVelocityN-CartVelocity, 
             color=PoleVelocity)) +
    geom_point() + theme_light()
  
  ggplot(samples_step %>% filter(Action==1), 
         aes(x=PoleAngle, y=CartVelocityN-CartVelocity, 
             color=PoleVelocity)) +
    geom_point() + theme_light()
  
  ggplot(samples_step, 
         aes(x=CartVelocity, y=(PoleVelocity+CartVelocity),color=PoleAngle)) + 
    geom_point() + theme_light()
  
  ggplot(samples_step %>% filter(Action==0), 
         aes(x=PoleVelocity,y=PoleAngleN-PoleAngle,color=CartVelocity )) + 
    geom_point() + theme_light()
  
  ggplot(samples_step %>% filter(Action==1), 
         aes(x=PoleAngle,y=PoleVelocityN - PoleVelocity, color=CartVelocity )) + 
    geom_point() + theme_light()
}
