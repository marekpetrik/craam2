# Intent and Motivation #

The scripts in this folder generate samples from a posterior distribution over MDP transition (and possibly) reward parameters. The posterior samples should contain all information that is necessary to compute robust solutions and evaluate and compare different algorithms.

Currently, only tabular MDPs are supported. Each state and each action have 0-based id.

# Content #

Each script should create the following files:

1. `parameters.csv`: parameters (like discount rate)
2. `initial.csv.xz`: initial distribution
3. `true.csv.xz`: true MDP model
4. `training.csv.xz`: training data meant to compute the policy
5. `test.csv.xz`: test data meant to evaluate the quality of the policy


## parameters.csv ##

A CSV file with two columns: `parameter` and `value`. The only parameter currently required is the discount rate: `discount`.

## initial.csv.xz ##

The initial distribution. When the model is uncertain, the optimal policy may depend on the initial distribution. This is different from plain MDP models.

The format of the file is a follows. It is an xzipped CSV file with columns: `idstate`, `probability`. Each row represents the initial probability of each state. If a state is missing, that means that the initial probability in that state is 0.

## true.csv.xz ##

This file represents the TRUE (or base) MDP model. This model may not be the posterior mean (and in fact may have a small posterior probability). The return of the policy on the true model is, in general, not very informative.

The format of the file is a follows. It is an xzipped CSV file with columns: `idstatefrom`, `idaction`, `idstateto`, `probability`, `reward`. Each row represents a single transition from a state, following an action, and transitioning to another state.

## training.csv.xz ##

This file represents the training posterior samples from the uncertain MDP model. This dataset is meant to be used to optimize the policy. There is another (test) file that is meant to be used to evaluate the solution quality.

The format of the file is a follows. It is an xzipped CSV file with columns: `idstatefrom`, `idaction`, `idoutcome`, `idstateto`, `probability`, `reward`. Each row represents a single transition from a state, following an action, and transitioning to another state for a *single* posterior sample. The particular posterior sample is identified by the 0-based index in `idoutcome`.

## test.csv.xz ##

This file represents the training posterior samples from the uncertain MDP model. This dataset is meant to be used to evaluate and compare the performance of policies. There is another (training) file that is meant to be used to optimize the policy.

The format of the file is a follows. It is an xzipped CSV file with columns: `idstatefrom`, `idaction`, `idoutcome`, `idstateto`, `probability`, `reward`. Each row represents a single transition from a state, following an action, and transitioning to another state for a *single* posterior sample. The particular posterior sample is identified by the 0-based index in `idoutcome`.
