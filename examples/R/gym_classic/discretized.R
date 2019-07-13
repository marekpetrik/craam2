#' Builds an MDP from the discretized states, using the 
#' the closes state to each sample as an approximation
#' 
#' The function also removes states with missing actions (to prevent
#' invalid state-action pairs)
#' 
#' @param states Dataframe with the states
#' @param samples.start Samples that contain the start of each transition. 
#' @param actionreward Action and reward for each transition (should have columns
#'                       `action`, `reward`). The action should be a factor
#' @param samples.next Samples that correspond to the transition from
#'   the previous sample. Should have the same number of rows as
#'   samples.start and the same columns as samples.start 
#' @param min.actions Removes states that do not have the minimum number of actions
#'   specified. Default is 0.
#' @param standardize Whether to standardize features to have a lower bound 0
#'   and upper bound
#'   
#' @return State numbers, scaling coefficients, mdp definition,
#'   and a function that maps states to their ids
build_discretized <- function(states, samples.start, actionreward, samples.next,
                              min.actions = 0, standardize = TRUE){
    library(dplyr)
    library(rcraam)
    
    if(require(assertthat)){
        # check the number of samples
        assert_that(nrow(samples.start) == nrow(samples.next))
        assert_that(nrow(actionreward) == nrow(samples.start))
        # check the features
        assert_that(setequal(colnames(states), colnames(samples.next)))
        assert_that(setequal(colnames(states), colnames(samples.start)))  
    }
    
    # standardize features
    if(standardize){
        scales <- apply(samples.start, 2, function(x){max(x) - min(x)})
        
        for(sn in names(scales)){
            states[sn] <- states[sn] / scales[sn]
            samples.start[sn] <- samples.start[sn] / scales[sn]
            samples.next[sn] <- samples.next[sn] / scales[sn]
        }
        
    }else{
        scales <- apply(samples.start, 2, function(x){1.0})
    }
    
    repeat{
        state.count <- nrow(states)  
        
        # construct samples with ids (the ids are 0-based)
        state.ids.from <- as.integer(
            class::knn1(states,
                        samples.start,
                        1:state.count)) - 1
        state.ids.to <- as.integer(
            class::knn1(states,
                        samples.next,
                        1:state.count)) - 1
        action.ids <- as.numeric(as.factor(actionreward$action)) - 1
        
        samples.frame <- data.frame(idstatefrom = state.ids.from,
                                    idaction = action.ids,
                                    idstateto = state.ids.to,
                                    reward = actionreward$reward)
        
        if(min.actions == 0)
            break
        # check for states that are missing actions, and remove them
        keep_states <- samples.frame %>% group_by(idstatefrom) %>% 
            summarize(action_count = n_distinct(idaction)) %>% 
            filter(action_count >= min.actions)
        
        if(nrow(keep_states) == state.count)
            break
        
        # now remove the states that do not have enough actions
        states <- states[keep_states$idstatefrom + 1,]
    }
    
    find.state <- function(state.features){
        sf <- state.features[names(scales)] / scales
        as.integer(class::knn1(states, sf, 1:nrow(states))) - 1
    }
    return(list(states = states, 
                scales = scales,
                mdp = rcraam::mdp_from_samples(samples.frame),
                find.state = find.state))
}
