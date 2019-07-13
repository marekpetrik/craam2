
## Generate samples

import random
import gym
import tqdm
import csv
env = gym.make('CartPole-v1')

generate_samples = False

random.seed(2018)

if(generate_samples):
    print("Generating samples ...")

    with open('cartpole.csv', 'w', newline='') as csvfile:
        samplewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        samplewriter.writerow(["Step", "CartPos", "CartVelocity", "PoleAngle", \
                                "PoleVelocity", "Action", "Reward"])
        
        laststate = None
        for k in tqdm.trange(20):
            env.reset()
            done = False
            for i in range(100):
                env.render()
                action = env.action_space.sample()
                if i > 0:
                    samplewriter.writerow((i-1,) + tuple(state) + (action,) + (reward,))
                
                # stop only after saving the state
                if done:
                    #time.sleep(0.5)
                    break
                
                [state,reward,done,info] = env.step(action) # take a random action
                
    env.close()

## Simulate a policy NN

import gym
import tqdm
import pandas as pa
import numpy as np
import time

from gym.envs.registration import register

#register(
#    id='CartPole-v2',
#    entry_point='gym.envs.classic_control:CartPoleEnv',
#    max_episode_steps=5000,
#    reward_threshold=4750.0,
#)
# env = gym.make('CartPole-v2')


# number of runs to determine how good is the policy
trials = 1

env = gym.make('CartPole-v1')
policy = pa.read_csv("policy_nn.csv")
scales = pa.read_csv("scales.csv").values

statefeatures = policy.values[:,0:4]
stateids = np.array(policy["State"])
actions = np.array(policy["Action"])
values = np.array(policy["Value"])


if "Probability" in policy.columns:
    print("Using a randomized policy")
    policy_randomized = True
    probabilities = np.array(policy["Probability"])
else:
    print("Using a deterministic policy")
    policy_randomized = False
    probabilities = None

totalreward = 0

for trial in tqdm.trange(trials):
    env.reset()
    done = False
    for i in range(500):
        env.render()
        time.sleep(0.05)
        if i > 0:
            # find the closest state
            statescaled = state @ scales
            dst = np.linalg.norm(statefeatures - np.repeat(np.atleast_2d(statescaled), statefeatures.shape[0], 0), axis=1)    
            
            statei = np.argmin(dst) # state index in the file, not the number of the state
            
            if policy_randomized:
                idstate = stateids[statei]
                all_statei = np.where(stateids == idstate)[0] # find all relevant state ids
                all_probs = probabilities[all_statei]
                all_acts = actions[all_statei]
                assert(abs(1-sum(all_probs)) < 0.01)
                action = int(np.random.choice(all_acts, p = all_probs))
            else:
                # assume that there is a single action for each state
                action = int(actions[statei])
            
            print(i, stateids[statei], action, values[statei], np.linalg.norm(statescaled - statefeatures[statei]) )
        else:
            action = env.action_space.sample()
        
        # stop only after saving the state
        if done:
            break
        
        [state,reward,done,info] = env.step(action) # take a random action
        totalreward += reward

env.close()
print("Average reward per trial", totalreward / trials)
