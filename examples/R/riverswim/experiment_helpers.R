stat.type = list("Frequentist"=1,"Bayesian"=2)
methods = list(
  "L1.Hoeff" = 1,
  "L1.Bern" = 2,
  "L1.BCI" = 3,
  "L1.wHoeff" = 4,
  "L1.wBern" = 5,
  "L1.wBCI" = 6,
  "Linf.Hoeff" = 7,
  "Linf.BCI" = 8,
  "Linf.wHoeff" = 9,
  "Linf.wBCI" = 10,
  "L1.wHoeff.SOCP" = 11,
  "L1.wBCI.SOCP" = 12
)
method_names <-
  list(
    "L1 Hoeffding",
    "L1 Bernstein",
    "L1 BCI",
    "L1 weighted Hoeffding",
    "L1 weighted Bernstein",
    "L1 weighted BCI",
    "Linf Hoeffding",
    "Linf BCI",
    "Linf weighted Hoeffding",
    "Linf weighted BCI",
    "L1 weighted Hoeffding SOCP",
    "L1 weighted BCI SOCP"
  )

num.methods <- length(methods)


init.models.opt <- function(num_methods, num_states, num_actions){
  # List of empty rmpds and budgets, Frequentist and Bayesian RMDPs and list of budgets for each method
  rmdps <- list("fmdp"=NULL, "bmdp"=NULL)#list("fmdp"=rcraam::get_mdp(), "bmdp"=rcraam::get_mdp())
  sa_budgets <- list()
  sas_weights <- list()

  # Initialize rmdps and budget dataframe
  for(i in 1:num_methods){
    sa_budgets[[i]] <- data.table(idstate=integer(), idaction=integer(), value=double())
    sas_weights[[i]] <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), value=double())
  }

  return(list("rmdps" = rmdps, "sa_budgets" = sa_budgets, "sas_weights" = sas_weights))
}

# *** Weights only depend on the current value function and is same for each state-action to next state.
# Compute it once and then use that for each state-action.
# The SOCP method requires Psi and p_bar as well. We initiate it here but compute
# it in "run.experiment" function.  ***
precompute.weights <- function(li.value.functions){
  weights.l1.whoeff = compute.weights(li.value.functions[[methods$L1.wHoeff]], norm = "L1")
  weights.l1.wbern = compute.weights(li.value.functions[[methods$L1.wBern]], norm = "L1")
  weights.l1.wbci = compute.weights(li.value.functions[[methods$L1.wBCI]], norm = "L1")

  weights.linf.whoeff = compute.weights(li.value.functions[[methods$Linf.wHoeff]], norm = "Linf")
  weights.linf.wbci = compute.weights(li.value.functions[[methods$Linf.wBCI]], norm = "Linf")

  return(list("l1.whoeff" = weights.l1.whoeff, "l1.wbern" = weights.l1.wbern,
              "l1.wbci" = weights.l1.wbci,
              "linf.whoeff"=weights.linf.whoeff, "linf.wbci"=weights.linf.wbci))
}

# *** For a specific confidence level and fixed number of samples, frequentist budget value is
# constant for each state and action. So, pre-compute it and then use for each state-action
# The SOCP method requires Psi and p_bar as well. We initiate it here but compute
# it in "run.experiment" function.  ***

precompute.freq.budgets <- function(num_samples, num_states, num_actions, confidence_level, weights_uniform, precomp.weights){
  l1.budget.hoeff <- l1.weighted.size.hoeffding(nsamples = num_samples, nstates = num_states, nactions = num_actions,
                                                confidence = confidence_level, weights = weights_uniform )
  l1.budget.bern <- l1.weighted.size.bernstein(nsamples = num_samples, nstates = num_states, nactions = num_actions,
                                               confidence = confidence_level, weights = weights_uniform )
  
  l1.budget.whoeff <- l1.weighted.size.hoeffding(nsamples = num_samples, nstates = num_states, nactions = num_actions,
                                                 confidence = confidence_level, weights = precomp.weights$l1.whoeff )
  
  l1.budget.wbern <- l1.weighted.size.bernstein(nsamples = num_samples, nstates = num_states, nactions = num_actions,
                                                confidence = confidence_level, weights = precomp.weights$l1.wbern )
  
  linf.budgets.hoeff <- linf.size.hoeffding(nsamples = num_samples, nstates = num_states, nactions = num_actions, confidence = confidence_level)
  linf.budgets.whoeff <- linf.weighted.size.hoeffding(nsamples = num_samples, nstates = num_states, nactions = num_actions,
                                                      confidence = confidence_level, weights = precomp.weights$linf.whoeff)
  return(list("l1.hoeff" = l1.budget.hoeff, "l1.bern" = l1.budget.bern, "l1.whoeff" = l1.budget.whoeff, "l1.wbern" = l1.budget.wbern,
              "linf.hoeff"=linf.budgets.hoeff, "linf.whoeff"=linf.budgets.whoeff))
}


compute.bayes.budget <- function(bayes_nominal_trp, posteriori_samples, num_actions, confidence_level, weights_uniform, precomp.weights){
  #cat("bayes_nominal_trp",length(bayes_nominal_trp),"weights_uniform",length(weights_uniform))
  l1.budget.bci <- l1.size.bayes(bayes_nominal_trp, posteriori_samples, nactions = num_actions, confidence_level, weights = weights_uniform)
  l1.budget.wbci <- l1.size.bayes(bayes_nominal_trp, posteriori_samples, nactions = num_actions, confidence_level, weights = precomp.weights$l1.wbci)

  linf.budget.bci <- linf.size.bayes(bayes_nominal_trp, posteriori_samples, nactions = num_actions, confidence_level, weights = weights_uniform)
  linf.budget.wbci <- linf.size.bayes(bayes_nominal_trp, posteriori_samples, nactions = num_actions, confidence_level, weights = precomp.weights$linf.wbci)

  return(list("l1.bci" = l1.budget.bci, "l1.wbci" = l1.budget.wbci, "linf.bci" = linf.budget.bci, "linf.wbci" = linf.budget.wbci))
}



# *** Construct ambiguity sets tuples ***
construct.sa.ambiguity.opt <- function(state, action, num_states, num_actions, freq.budgets, bayes.budgets, sa_budgets, record.data){
  record.index  <- state*num_actions + action + 1
  #record.index <- id.statefrom*num_actions + (id.action) + 1

  # Record the budget for current state-action. Storage format: state - action - budget.value
  sa_budgets[[methods$L1.Hoeff]] <- rbindlist(list(sa_budgets[[methods$L1.Hoeff]],
                                                   list(idstate=state,idaction=action,value=freq.budgets$l1.hoeff)))
  sa_budgets[[methods$L1.Bern]] <- rbindlist(list(sa_budgets[[methods$L1.Bern]],
                                                  list(idstate=state,idaction=action,value=freq.budgets$l1.bern)))
  sa_budgets[[methods$L1.BCI]] <- rbindlist(list(sa_budgets[[methods$L1.BCI]],
                                                 list(idstate=state,idaction=action,value=bayes.budgets$l1.bci)))
  sa_budgets[[methods$L1.wHoeff]] <- rbindlist(list(sa_budgets[[methods$L1.wHoeff]],
                                                    list(idstate=state,idaction=action,value=freq.budgets$l1.whoeff)))

  # SOCP
  #weights.l1.hff.socp   <- record.data[[record.index]]$w.l1.whoeff.socp
  psi.l1.hff.socp <- record.data[[record.index]]$psi.l1.whoeff.socp

  sa_budgets[[methods$L1.wHoeff.SOCP]] <- rbindlist(list(sa_budgets[[methods$L1.wHoeff.SOCP]],
                                                    list(idstate=state,idaction=action,value=psi.l1.hff.socp)))

  sa_budgets[[methods$L1.wBern]] <- rbindlist(list(sa_budgets[[methods$L1.wBern]],
                                                   list(idstate=state,idaction=action,value=freq.budgets$l1.wbern)))
  sa_budgets[[methods$L1.wBCI]] <- rbindlist(list(sa_budgets[[methods$L1.wBCI]],
                                                  list(idstate=state,idaction=action,value=bayes.budgets$l1.wbci)))

  # SOCP
  #weights.l1.bay.socp <- record.data[[record.index]]$w.l1.wbci.socp
  psi.l1.bay.socp <- record.data[[record.index]]$psi.l1.wbci.socp

  sa_budgets[[methods$L1.wBCI.SOCP]] <- rbindlist(list(sa_budgets[[methods$L1.wBCI.SOCP]],
                                                  list(idstate=state,idaction=action,value=psi.l1.bay.socp)))


  sa_budgets[[methods$Linf.Hoeff]] <- rbindlist(list(sa_budgets[[methods$Linf.Hoeff]],
                                                     list(idstate=state,idaction=action,value=freq.budgets$linf.hoeff)))
  sa_budgets[[methods$Linf.wHoeff]] <- rbindlist(list(sa_budgets[[methods$Linf.wHoeff]],
                                                      list(idstate=state,idaction=action,value=freq.budgets$linf.whoeff)))
  sa_budgets[[methods$Linf.BCI]] <- rbindlist(list(sa_budgets[[methods$Linf.BCI]],
                                                   list(idstate=state,idaction=action,value=bayes.budgets$linf.bci)))
  sa_budgets[[methods$Linf.wBCI]] <- rbindlist(list(sa_budgets[[methods$Linf.wBCI]],
                                                    list(idstate=state,idaction=action,value=bayes.budgets$linf.wbci)))

  return(sa_budgets)
}

# *** Construct weight tuples, assign a weight for each state-action-next_state ***
construct.sas.weights.opt <- function(state, action, id.stateto, num_states, num_actions, precomp.weights, sas_weights, type, record.data){
  
  # 1-based index for R
  next.state <- id.stateto+1
  
  
  id.statefrom <- state
  id.action <- action
  #record.index <- state*num_actions + action + 1
  record.index = id.statefrom*num_actions + (id.action) + 1
  
  # Record the weight for current state-action-next_state. Storage format: state - action - next_state - budget.value
  if(type==stat.type$Frequentist){
    sas_weights[[methods$L1.Hoeff]] <- rbindlist(list(sas_weights[[methods$L1.Hoeff]],
                                                      list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=1.0)))
    sas_weights[[methods$L1.Bern]] <- rbindlist(list(sas_weights[[methods$L1.Bern]],
                                                     list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=1.0)))
    sas_weights[[methods$L1.wHoeff]] <- rbindlist(list(sas_weights[[methods$L1.wHoeff]],
                                                       list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=precomp.weights$l1.whoeff[[next.state]])))
    
    # SOCP
    weights.l1.hff.socp <- record.data[[record.index]]$w.l1.whoeff.socp
    sas_weights[[methods$L1.wHoeff.SOCP]] <- rbindlist(list(sas_weights[[methods$L1.wHoeff.SOCP]],
                                                            list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=weights.l1.hff.socp[[next.state]])))
    
    sas_weights[[methods$L1.wBern]] <- rbindlist(list(sas_weights[[methods$L1.wBern]],
                                                      list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=precomp.weights$l1.wbern[[next.state]])))
    
    sas_weights[[methods$Linf.Hoeff]] <- rbindlist(list(sas_weights[[methods$Linf.Hoeff]],
                                                        list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=1.0)))
    sas_weights[[methods$Linf.wHoeff]] <- rbindlist(list(sas_weights[[methods$Linf.wHoeff]],
                                                         list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=precomp.weights$linf.whoeff[[next.state]])))
  }
  else if(type==stat.type$Bayesian){
    sas_weights[[methods$L1.BCI]] <- rbindlist(list(sas_weights[[methods$L1.BCI]],
                                                    list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=1.0)))
    sas_weights[[methods$L1.wBCI]] <- rbindlist(list(sas_weights[[methods$L1.wBCI]],
                                                     list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=precomp.weights$l1.wbci[[next.state]])))
    
    
    # SOCP
    weights.l1.bay.socp <- record.data[[record.index]]$w.l1.wbci.socp
    sas_weights[[methods$L1.wBCI.SOCP]] <- rbindlist(list(sas_weights[[methods$L1.wBCI.SOCP]],
                                                          list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=weights.l1.bay.socp[[next.state]])))
    
    sas_weights[[methods$Linf.BCI]] <- rbindlist(list(sas_weights[[methods$Linf.BCI]],
                                                      list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=1.0)))
    sas_weights[[methods$Linf.wBCI]] <- rbindlist(list(sas_weights[[methods$Linf.wBCI]],
                                                       list(idstatefrom=state,idaction=action,idstateto=id.stateto, value=precomp.weights$linf.wbci[[next.state]])))
  }
  return(sas_weights)
}

# Generate frequentist and Bayesian sampled models from from the underlying true mdp
sample.model <- function(num_states, num_actions, num_samples, num_bayes_samples, prior, true.mdp){
  record.data <- list()
  record.rewards <- hash()
  for(s in 1:num_states){
    for(a in 1:num_actions){

      #cat("sample.model, state",s,"action",a,"\n")
      # craam mdp is 0-based, unlike R arrays
      id.statefrom = s-1
      id.action = a-1

      # The sample records would be of a 3 dimensional matrix: iter x num_states x num_actions
      # Compute an unique index to convert and store this as a 1 dimensional array.
      record.index <- id.statefrom*num_actions + (id.action) + 1
      transitions <- filter(true.mdp, idstatefrom==id.statefrom, idaction==id.action)

      true.trp <- rep(eps, num_states)
      rewards <- rep(0.0, num_states)

      to.states <- transitions$idstateto

      # Add transitions and rewards for the states present in current transition
      for(i in 1:length(to.states)){
        true.trp[to.states[i]+1] <- transitions$probability[i]
        rewards[to.states[i]+1] <- transitions$reward[i]
        record.rewards[[toString(c(s,a,to.states[i]+1))]] <- transitions$reward[i]
      }

      #cat("sample.model, true.trp: ", true.trp, "\n")
      samples <- rmultinom(1, size = num_samples, prob = true.trp)

      #cat("samples",samples, "\n")
      prior <- rep(eps, num_states)
      for(i in 1:length(to.states)){
        if(true.trp[[i]]>eps)
          prior[[i]] <- 1
      }
      #cat("prior", prior)

      fci_nominal_trp <- samples / sum(samples)
      posteriori_samples <- rdirichlet(num_bayes_samples, samples + prior)
      bayes_nominal_trp <- colMeans(posteriori_samples)

      #cat("fci_nominal_trp", fci_nominal_trp, "\n")
      #cat("bayes_nominal_trp", bayes_nominal_trp, "\n")

      record.data[[record.index]] <- list("posteriori_samples"=posteriori_samples, 
                                          "fci_nominal_trp"=fci_nominal_trp,
                                          "bayes_nominal_trp"=bayes_nominal_trp,
                                          "w.l1.whoeff.socp" = NA,
                                          "w.l1.wbci.socp" = NA,
                                          "psi.l1.whoeff.socp" = NA,
                                          "psi.l1.wbci.socp" = NA
                                          )
    }
  }
  return(list("record.data"=record.data,"record.rewards"=record.rewards))
}


get.nonrobust.sol <- function(num_states, num_actions, discount_factor, sampled.model){
  config <- list(iterations = 50000, progress = FALSE, output_tran = TRUE, timeout = 300, precision = 0.1)
  num_methods <- length(method_names)

  record.data <- sampled.model$record.data
  record.rewards <- sampled.model$record.rewards

  freq.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
  bayes.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())

  for(s in 1:num_states){
    for(a in 1:num_actions){
      # craam mdp is 0-based, unlike R arrays
      id.statefrom = s-1
      id.action = a-1

      # The sample records would be of a 2 dimensional matrix: num_states x num_actions
      # Compute an unique index to convert and store this as a 1 dimensional array.
      record.index = id.statefrom*num_actions + (id.action) + 1

      fci_nominal_trp <- record.data[[record.index]]$fci_nominal_trp
      posteriori_samples <- record.data[[record.index]]$posteriori_samples
      bayes_nominal_trp <- record.data[[record.index]]$bayes_nominal_trp

      freq.sa.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
      bayes.sa.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
      freq.n.times <- list()
      bayes.n.times <- list()

      # Add transition to next states and record corresponding weights
      for(s.next in 1:num_states){
        id.stateto = s.next-1

        reward <- ifelse(has.key(toString(c(s,a,s.next)), record.rewards), record.rewards[[toString(c(s,a,s.next))]], 0.0)

        # There's a transition to id.stateto, so add that transition and weight of id.stateto
        if(fci_nominal_trp[s.next]>0.0){
          # store the transition (s,a,s',r) into df
          freq.sa.df <- rbindlist(list(freq.sa.df, list(id.statefrom, id.action, id.stateto, reward)))

          # store frequency of this transition happening: trp*100
          freq.n.times[[length(freq.n.times)+1]] <- max(as.integer(fci_nominal_trp[s.next]*1000),1)
        }
        if(bayes_nominal_trp[s.next]>0.0){
          # store the transition (s,a,s',r) into df
          bayes.sa.df <- rbindlist(list(bayes.sa.df, list(id.statefrom, id.action, id.stateto, reward)))

          # store frequency of this transition happening: trp*100
          bayes.n.times[[length(bayes.n.times)+1]] <- max(as.integer(bayes_nominal_trp[s.next]*1000),1)
        }
      }

      # Repeat each (s,a,s',r) sample frequency number of times and put into the dataframe and bind
      freq.sa.df <- freq.sa.df[rep(seq_len(nrow(freq.sa.df)), freq.n.times),]
      freq.df <- rbindlist(list(freq.df, freq.sa.df)) #rbind(freq.df, freq.sa.df)

      # Repeat each (s,a,s',r) sample frequency number of times and put into the dataframe and bind
      bayes.sa.df <- bayes.sa.df[rep(seq_len(nrow(bayes.sa.df)), bayes.n.times),]
      bayes.df <- rbindlist(list(bayes.df, bayes.sa.df)) #rbind(bayes.df, bayes.sa.df)
    }
  }

  fmdp <- rcraam::mdp_from_samples(freq.df)
  bmdp <- rcraam::mdp_from_samples(bayes.df)

  sol.fmdp <- solve_mdp(fmdp, discount_factor, append(config, list(algorithm="pi")))$valuefunction
  sol.bmdp <- solve_mdp(bmdp, discount_factor, append(config, list(algorithm="pi")))$valuefunction

  li.value.functions <- list()
  for(m in methods){
    if(grepl("BCI", method_names[m]))
      li.value.functions[m] <- list( sol.bmdp )
    else
      li.value.functions[m] <- list( sol.fmdp )
  }
  return(li.value.functions)
}



dot.product <- function(x, y){
  return(x%*%y)
}




run.experiment <- function(run, num_states, num_actions, discount_factor, confidence_level, initial.distrubution, sampled.model, li.value.functions, file.name){

  num_iterations = 1
  weights_uniform <- rep(1, num_states) # Uniform weightes to be used for unweighted case

  config <- list(iterations = 50000, progress = FALSE, output_tran = TRUE, timeout = 300, precision = 0.1)
  num_methods <- length(method_names)

  record.data <- sampled.model$record.data
  record.rewards <- sampled.model$record.rewards
  
  for(iter in 1:num_iterations){
    cat("------------- run# ", run, ", iteration# ",iter," -------------\n")

    init <- init.models.opt(num_methods, num_states, num_actions)
    
    
    rmdps <- init$rmdps
    sa_budgets <- init$sa_budgets
    sas_weights <- init$sas_weights
    precomp.weights <- precompute.weights(li.value.functions)
    freq.budgets <- precompute.freq.budgets(num_samples_from_truth, num_states, num_actions, confidence_level, weights_uniform, precomp.weights)

    freq.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
    bayes.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
  

    for(s in 1:num_states){
      for(a in 1:num_actions){
        #cat("state:",s,"action:",a,"\n")

        # craam mdp is 0-based, unlike R arrays
        id.statefrom = s-1
        id.action = a-1

        # The sample records would be of a 2 dimensional matrix: num_states x num_actions
        # Compute an unique index to convert and store this as a 1 dimensional array.
        record.index = id.statefrom*num_actions + (id.action) + 1

        fci_nominal_trp <- record.data[[record.index]]$fci_nominal_trp
        posteriori_samples <- record.data[[record.index]]$posteriori_samples
        bayes_nominal_trp <- record.data[[record.index]]$bayes_nominal_trp

        bayes.budget <- compute.bayes.budget(bayes_nominal_trp, posteriori_samples, num_actions, confidence_level, weights_uniform, precomp.weights)
        
        
        # TODO: ADD SOCP weights
        # initiate psi from weighted L1 analytical solution

        weights.l1.hff.socp <- precomp.weights$l1.whoeff
        psi.l1.hff.socp <- freq.budgets$l1.whoeff

        weights.l1.bay.socp <- precomp.weights$l1.wbci
        psi.l1.bay.socp <- bayes.budget$l1.wbci
        # A few iteration
        for (i in 1:3){
          weights.l1.hff.socp <- compute.weights(z = li.value.functions[[methods$L1.wHoeff.SOCP]],
                                             norm = "L1",
                                             solution = "socp",
                                             p_bar = fci_nominal_trp,
                                             psi = psi.l1.hff.socp )
          psi.l1.hff.socp <- l1.weighted.size.hoeffding(nsamples = num_samples_from_truth, nstates = num_states, nactions = num_actions,
                                     confidence = confidence_level, weights = weights.l1.hff.socp )


          weights.l1.bay.socp <- compute.weights(z = li.value.functions[[methods$L1.wBCI.SOCP]],
                                                 norm = "L1",
                                                 solution = "socp",
                                                 p_bar = bayes_nominal_trp,
                                                 psi = psi.l1.bay.socp)
          psi.l1.bay.socp <- l1.size.bayes(bayes_nominal_trp,
                                           posteriori_samples,
                                           nactions = num_actions,
                                           confidence = confidence_level,
                                           weights = weights.l1.bay.socp )

        }
     

        record.data[[record.index]]$w.l1.whoeff.socp <- weights.l1.hff.socp
        record.data[[record.index]]$psi.l1.whoeff.socp <- psi.l1.hff.socp
        record.data[[record.index]]$w.l1.wbci.socp <- weights.l1.bay.socp
        record.data[[record.index]]$psi.l1.wbci.socp <- psi.l1.bay.socp




        freq.sa.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
        bayes.sa.df <- data.table(idstatefrom=integer(), idaction=integer(), idstateto=integer(), reward=double())
        freq.n.times <- list()
        bayes.n.times <- list()

        # Add transition to next states and record corresponding weights
        for(s.next in 1:num_states){
          id.stateto = s.next-1

          reward <- ifelse(has.key(toString(c(s,a,s.next)), record.rewards), record.rewards[[toString(c(s,a,s.next))]], 0.0)

          # There's a transition to id.stateto, so add that transition and weight of id.stateto
          if(fci_nominal_trp[s.next]>=0.0){
            # store the transition (s,a,s',r) into df
            freq.sa.df <- rbindlist(list(freq.sa.df, list(id.statefrom, id.action, id.stateto, reward)))

            # store frequency of this transition happening: trp*100
            freq.n.times[[length(freq.n.times)+1]] <- max(as.integer(fci_nominal_trp[s.next]*1000),1)

            sas_weights <-
              construct.sas.weights.opt(
                id.statefrom,
                id.action,
                id.stateto,
                num_states,
                num_actions,
                precomp.weights,
                sas_weights,
                stat.type$Frequentist,
                record.data
              )
            
          }
          
          if(bayes_nominal_trp[s.next]>=0.0){
            # store the transition (s,a,s',r) into df
            bayes.sa.df <- rbindlist(list(bayes.sa.df, list(id.statefrom, id.action, id.stateto, reward)))

            # store frequency of this transition happening: trp*100
            bayes.n.times[[length(bayes.n.times)+1]] <- max(as.integer(bayes_nominal_trp[s.next]*1000),1)

            sas_weights <-
              construct.sas.weights.opt(
                id.statefrom,
                id.action,
                id.stateto,
                num_states,
                num_actions,
                precomp.weights,
                sas_weights,
                stat.type$Bayesian,
                record.data
              )
          }
        }

        # Repeat each (s,a,s',r) sample frequency number of times and put into the dataframe and bind
        freq.sa.df <- freq.sa.df[rep(seq_len(nrow(freq.sa.df)), freq.n.times),]
        freq.df <- rbindlist(list(freq.df, freq.sa.df)) #rbind(freq.df, freq.sa.df)

        # Repeat each (s,a,s',r) sample frequency number of times and put into the dataframe and bind
        bayes.sa.df <- bayes.sa.df[rep(seq_len(nrow(bayes.sa.df)), bayes.n.times),]
        bayes.df <- rbindlist(list(bayes.df, bayes.sa.df)) #rbind(bayes.df, bayes.sa.df)

        sa_budgets <- construct.sa.ambiguity.opt(id.statefrom, id.action, num_states, num_actions, freq.budgets, bayes.budget, sa_budgets, record.data)
      }
    }

    rmdps$fmdp <- rcraam::mdp_from_samples(freq.df)
    rmdps$bmdp <- rcraam::mdp_from_samples(bayes.df)


    for(m in methods){

      #Don't spend time on Bernstein methods
      # if(grepl("Bernstein", method_names[m])){
      #   next
      # }

      rmdp = rmdps$fmdp

      # Only bayesian methods use the mdp constructed with posterior samples
      if(grepl("BCI", method_names[m]))
        rmdp = rmdps$bmdp

      b <- sa_budgets[[m]]
      w <- sas_weights[[m]]
      
      browser()

      # Not all states are reachable from a state and weights are present only for non-zero transitions
      # So, normalize the weights for present states.
      if(grepl("weighted", method_names[m]))
        w <- w %>% group_by(idstatefrom, idaction) %>% mutate(value = normalize(value))
      bw <- list("budgets"=b, "weights"=w)
      if(grepl("L1", method_names[m]))
        li.value.functions[m] <- list( rsolve_mdp_sa(rmdp, discount_factor, "l1w", bw, append(config, list(algorithm="vi")))$valuefunction )
      else
        li.value.functions[m] <- list( rsolve_mdp_sa(rmdp, discount_factor, "linfw_g", bw, append(config, list(algorithm="vi")))$valuefunction )

    }
  }

  budgets <- matrix(0.0, nrow=num_methods, ncol=num_iterations)

  returns <- lapply(li.value.functions, dot.product, y = initial.distrubution)

  d = cbind(confidence_level, rowMeans(budgets), returns, method_names)
  write.table(d, file=file.name, append=TRUE, sep = ",", row.names = FALSE, col.names = FALSE)
}



