library(Rcpp)
library(rcraam)

sourceCpp("cancer_sim.cpp")

def_config <- default_config()

state <- init_state()

print(cancer_transition(state, FALSE,  def_config))
print(cancer_transition(state, TRUE, def_config))


# *** generate a bunch of random states (to serve as centers of the aggregate states) ***


# *** construct the MDP of the simulator

#cmdp <- cancer_mdp(def_config, );
