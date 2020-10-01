source('utils.R')
source('experiment_helpers.R')
library(gtools)
library(dplyr)
library(rcraam)
library(ggplot2)
require(latex2exp)
library(gridExtra)
library(gtable)
library(grid)
library(data.table)
library(readr)
library(reshape2)
library(assertthat)
loadNamespace("tidyr")
loadNamespace("reshape2")
loadNamespace("stringr")

description <- "simple_mdp.csv"

num.states <- 5
num.actions <- 1
samples <- 50
bayes.samples <- 1000
sample.seed <- 2018
discount <- 0.95
episodes <- 10


create_small_mdp <- function(num.states = 6,
                             num.actions = 1,
                             discount = 0.9,
                             reward.range = c(-10 , 10),
                             mdp_csv_filename = "description.csv"
){

    # drichlet dist for transition probabilities
    alpha <- rep(1, num.states)
    probabilities = rdirichlet(num.states*num.actions, alpha)
    # rewards
    reward.list <- runif(num.states*num.actions , reward.range[1],reward.range[2])
    rewards <- matrix( reward.list, nrow=num.states, ncol=num.actions)
    # MDP table each row contains:
    # c("idstatefrom","idaction","idstateto","probability","reward")
    mdp.df <-  data.frame(matrix(ncol = 5, nrow = 0))
    for(s in 1:num.states){
        for(a in 1:num.actions){
            # craam mdp is 0-based, unlike R arrays
            id.statefrom = s-1
            id.action = a-1


            id.reward = rewards[s,a]
            # Add transition to next states and record corresponding weights
            for(s.next in 1:num.states){
                id.stateto = s.next-1
                id.prob = probabilities[(s-1)*num.actions + a, s.next]

                new.row <- c(id.statefrom,id.action,id.stateto,id.prob,id.reward)
                mdp.df <- rbind(mdp.df, new.row)

            }
        }
    }

    colnames(mdp.df) <- c("idstatefrom","idaction","idstateto","probability","reward")
    write.csv(mdp.df, mdp_csv_filename, row.names = FALSE)
    return(mdp.df)
}

create_small_mdp(num.states ,num.actions, discount ,
                 reward.range = c(-10 , 10),
                 mdp_csv_filename = description)







## ----- Initialization ------

mdp.truth <- read_csv(description,
                      col_types = cols(idstatefrom = 'i',
                                       idaction = 'i',
                                       idstateto = 'i',
                                       probability = 'd',
                                       reward = 'd'))
rewards.truth <- mdp.truth %>% select(-probability)

ur.policy = data.frame(idstate = seq(0,num.states-1),
                       idaction = rep(0,num.states),
                       probability = 1)

# generate samples from the swimmer domain
simulation <- simulate_mdp(mdp.truth, 0, ur.policy, episodes = episodes,
                           horizon = samples, seed = sample.seed)

# compute the true value function
sol.true <- solve_mdp(mdp.truth, discount, show_progress = FALSE)
vf.true <- sol.true$valuefunction$value


## Baysian sampling

priors <- rewards.truth %>% select(-reward) %>% unique()

# compute sampled state and action counts
# add a uniform sample of each state and action to work as the dirichlet prior
sas_post_counts <- simulation %>%
    select(idstatefrom, idaction, idstateto) %>%
    rbind(priors) %>%
    group_by(idstatefrom, idaction, idstateto) %>% summarize(count = n())

# construct dirichlet posteriors
posteriors <- sas_post_counts %>%
    group_by(idstatefrom, idaction) %>%
    arrange(idstateto) %>%
    summarize(posterior = list(count), idstatesto = list(idstateto))




## ----- Frequentist sampling
# count the number of samples for each state and action
sa_counts <- simulation %>% select(idstatefrom, idaction) %>%
    group_by(idstatefrom, idaction) %>% summarize(count = n())

# number of valid state-action pairs
sa.count <- nrow(sa_counts)
# count the number of possible transitions from each state and action
tran.count <- rewards.truth %>% group_by(idstatefrom, idaction) %>%
    summarize(tran_count = n())

mdp.nominal <- mdp_from_samples(simulation)

# add transtions to states that have not been observed, and normalize
# them in order to get some kind of a transition probability
mdp.nominal <- full_join(mdp.nominal %>% select(-reward), rewards.truth,
                         by = c('idstatefrom', 'idaction', 'idstateto')) %>%
    mutate(probability = coalesce(probability, 1.0))
# normalize transition probabilities
mdp.freq <-
    full_join(mdp.nominal %>% group_by(idstatefrom, idaction) %>%
                  summarize(prob.sum = sum(probability)),
              mdp.nominal, by=c('idstatefrom', 'idaction')) %>%
    mutate(probability = probability / prob.sum) %>% select(-prob.sum) %>% na.fail()







# All methods


set.seed(858)
dir.name = "./results/"
file.name = "single_Bellman_update"
file.ext = ".csv"
file.results <- paste(dir.name, file.name, file.ext, sep = "")

method_names = list("L1.Frq",
                    "w.L1.Frq.anlyt",
                    "w.L1.Frq.SOCP",
                    "L1.Bay",
                    "w.L1.Bay.anlyt",
                    "w.L1.Bay.SOCP",
                    "Linf.Frq",
                    "w.Linf.Frq.anlyt",
                    "Linf.Bay",
                    "w.Linf.Bay")


col.names = "confidence,budget,return,method"
write(col.names, file = file.results, append = FALSE)

nStates = num.states
num_confs = 5
confidences <- seq(0.5, 0.999, length.out = num_confs)

weights.uniform <- rep(1, nStates)
for (i in 1:nStates) {
    idstate  <- i-1

    # frequentist count
    num_samples_from_truth <- sa_counts$count[[i]]
    # Bayesian count
    num_bayes_samples <- bayes.samples



    #z <- runif(nStates, min = -10, max = 10)
    #z <- sort(runif(nStates, min = -10, max = 100))
    #z <- seq(1,5,length.out = nStates)*3
    #z <- c(rep(0,nStates - 1), -5)
    z <- vf.true

    posterior.alpha <- posteriors$posterior[[i]]
    posteriori_samples <- rdirichlet(num_bayes_samples , alpha = posterior.alpha)
    bayes_nominal_trp <- colMeans(posteriori_samples)

    fci_nominal_trp <- mdp.freq %>% filter(idstatefrom == idstate) %>% pull(probability)



    weights.l1.anlyt <- compute.weights(z = z, norm = "L1", solution = "analytical")
    weights.linf.anlyt <- compute.weights(z = z, norm = "Linf", solution = "analytical")


    for (confidence in confidences) {
        ####################################################
        ##  L1 - Hoefding
        ####################################################



        ####################################################
        ## Calculate the budget L1 - uniform
        ####################################################
        psi.l1.hff <-
            l1.weighted.size.hoeffding(
                nsamples = num_samples_from_truth,
                nstates = nStates,
                nactions = 1,
                confidence = confidence,
                weights = weights.uniform
            )
        ####################################################
        ## Calculate the budget L1 - weighted
        ####################################################

        psi.w.l1.hff <-
            l1.weighted.size.hoeffding(
                nsamples = num_samples_from_truth,
                nstates = nStates,
                nactions = 1,
                confidence = confidence,
                weights = weights.l1.anlyt
            )


        ####################################################
        ## Calculate the returns L1 - uniform
        ####################################################

        return.l1.hff <- rcraam::worstcase_l1_w(z,
                                                fci_nominal_trp,
                                                weights.uniform,
                                                psi.l1.hff)$value


        ####################################################
        ## Calculate the returns L1 - weighted
        ####################################################

        return.w.l1.hff.anlyt <-rcraam::worstcase_l1_w(z,
                                                       fci_nominal_trp,
                                                       weights.l1.anlyt,
                                                       psi.w.l1.hff)$value

        # NOTE: psi.w.l1.socp depends on weights and weights depends on psi (budget)
        # So we choose uniform weights to find the psi (budget)
        psi.l1.hff.socp <- psi.l1.hff
        weights.l1.socp <- weights.uniform
        #psi.l1.hff.socp <- psi.w.l1.hff
        #weights.l1.socp <- weights.l1.anlyt

        for(i in 1:5){
            weights.l1.socp <- compute.weights(z = z,
                                               norm = "L1",
                                               solution = "socp",
                                               p_bar = fci_nominal_trp,
                                               psi = psi.l1.hff.socp )

            psi.l1.hff.socp <-  l1.weighted.size.hoeffding(
                nsamples = num_samples_from_truth,
                nstates = nStates,
                nactions = 1,
                confidence = confidence,
                weights = weights.l1.socp
            )
        }
        return.w.l1.hff.socp <-rcraam::worstcase_l1_w(z,
                                                      fci_nominal_trp,
                                                      weights.l1.socp,
                                                      psi.l1.hff.socp)$value

        ####################################################
        ##  L1 - Bayesian
        ####################################################


        ####################################################
        ## Calculate the budget L1 - uniform
        ####################################################
        psi.l1.bay <-
            l1.size.bayes(
                bayes_nominal_trp,
                posteriori_samples,
                nactions = 1,
                confidence,
                weights = weights.uniform
            )
        ####################################################
        ## Calculate the budget L1 - weighted
        ####################################################

        psi.w.l1.bay <-
            l1.size.bayes(
                bayes_nominal_trp,
                posteriori_samples,
                nactions = 1,
                confidence,
                weights = weights.l1.anlyt
            )


        ####################################################
        ## Calculate the returns L1 - uniform
        ####################################################

        return.l1.bay <- rcraam::worstcase_l1_w(z,
                                                bayes_nominal_trp,
                                                weights.uniform,
                                                psi.l1.bay)$value


        ####################################################
        ## Calculate the returns L1 - weighted
        ####################################################

        return.w.l1.bay.anlyt <- rcraam::worstcase_l1_w(z,
                                                        bayes_nominal_trp,
                                                        weights.l1.anlyt,
                                                        psi.w.l1.bay)$value

        # NOTE: psi.w.l1.socp depends on weights and weights depends on psi (budget)
        # So we choose uniform weights to find the psi (budget)
        #psi.l1.bay.socp <- psi.l1.bay
        #weights.l1.socp <- weights.uniform
        psi.l1.bay.socp <- psi.w.l1.bay
        weights.l1.socp <- weights.l1.anlyt

        for(i in 1:3){
            weights.l1.socp <- compute.weights(z = z,
                                               norm = "L1",
                                               solution = "socp",
                                               p_bar = bayes_nominal_trp,
                                               psi = psi.l1.bay.socp )

            psi.l1.bay.socp <- l1.size.bayes(
                bayes_nominal_trp,
                posteriori_samples,
                nactions = 1,
                confidence,
                weights = weights.l1.socp
            )

        }
        return.w.l1.bay.socp <-rcraam::worstcase_l1_w(z,
                                                      bayes_nominal_trp,
                                                      weights.l1.socp,
                                                      psi.l1.bay.socp)$value


        ####################################################
        ##  Linf - Hoefding and Bayesian
        ####################################################

        ####################################################
        ## Calculate the budget Linf - uniform
        ####################################################
        psi.linf.hff <-
            linf.weighted.size.hoeffding(
                nsamples = num_samples_from_truth,
                nstates = nStates,
                nactions = 1,
                confidence = confidence,
                weights = weights.uniform
            )

        psi.linf.bay <- linf.size.bayes(
            bayes_nominal_trp,
            posteriori_samples,
            nactions = 1,
            confidence,
            weights = weights.uniform
        )

        ####################################################
        ## Calculate the budget Linf - weighted
        ####################################################

        psi.w.linf.hff <-
            linf.weighted.size.hoeffding(
                nsamples = num_samples_from_truth,
                nstates = nStates,
                nactions = 1,
                confidence = confidence,
                weights = weights.linf.anlyt
            )

        psi.w.linf.bay <- linf.size.bayes(
            bayes_nominal_trp,
            posteriori_samples,
            nactions = 1,
            confidence,
            weights = weights.linf.anlyt
        )


        ####################################################
        ## Calculate the returns Linf - uniform
        ####################################################

        return.linf.hff <- rcraam::worstcase_linf_w_gurobi(z,
                                                           fci_nominal_trp,
                                                           weights.uniform,
                                                           psi.linf.hff)$value

        return.linf.bay <- rcraam::worstcase_linf_w_gurobi(z,
                                                           bayes_nominal_trp,
                                                           weights.uniform,
                                                           psi.linf.bay)$value
        ####################################################
        ## Calculate the returns Linf - weighted
        ####################################################

        return.w.linf.hff.anlyt <-rcraam::worstcase_linf_w_gurobi(z,
                                                                  fci_nominal_trp,
                                                                  weights.linf.anlyt,
                                                                  psi.w.linf.hff)$value

        return.w.linf.bay <- rcraam::worstcase_linf_w_gurobi(z,
                                                             bayes_nominal_trp,
                                                             weights.linf.anlyt,
                                                             psi.w.linf.bay)$value


        budgets <-
            c(
                psi.l1.hff,
                psi.w.l1.hff,
                psi.l1.hff.socp,
                psi.l1.bay,
                psi.w.l1.bay,
                psi.l1.bay.socp,
                psi.linf.hff,
                psi.w.linf.hff,
                psi.linf.bay,
                psi.w.linf.bay
            )
        returns <-
            c(
                return.l1.hff,
                return.w.l1.hff.anlyt,
                return.w.l1.hff.socp,
                return.l1.bay,
                return.w.l1.bay.anlyt,
                return.w.l1.bay.socp,
                return.linf.hff,
                return.w.linf.hff.anlyt,
                return.linf.bay,
                return.w.linf.bay
            )



        d = cbind(confidence, budgets, returns, method_names)
        write.table(
            d,
            file = file.results,
            append = TRUE,
            sep = ",",
            row.names = FALSE,
            col.names = FALSE
        )
    }
}






# plot the result

dir.name = "./results/"
file.name = "single_Bellman_update"
file.ext = ".csv"
file.results <- paste(dir.name, file.name, file.ext, sep = "")


df <- read.csv(file = file.results, header = TRUE, sep = ",")
df <- df %>% group_by(confidence, method) %>% summarise(return = mean(return), budget = mean(budget))

df.sub <- df %>% select(confidence, budget, return, method) %>%
    filter(
        method %in% c(
            "L1.Frq",
            "w.L1.Frq.anlyt",
            "w.L1.Frq.SOCP",
            "L1.Bay",
            "w.L1.Bay.anlyt",
            "w.L1.Bay.SOCP",
            "Linf.Frq",
            "w.Linf.Frq.anlyt",
            "Linf.Bay",
            "w.Linf.Bay"
        )
    )

#col.names = "confidence,budget,return,method"


plot.res.color(df.sub, file.name)


