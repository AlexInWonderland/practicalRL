# most of the code is taken form Practical RL week0 assignment (https://github.com/yandexdataschool/Practical_RL/tree/master/week0)

# I'm just trying out OpenAI submission interface

import numpy as np
import os

import gym
from gym import wrappers

env = gym.make("FrozenLake-v0")
filespath = 'frozenlake-v0-experiment-1/'
os.system('rm -rf /tmp/gym-results/' + filespath)
env = wrappers.Monitor(env, '/tmp/gym-results/' + filespath)

n_states = env.observation_space.n
n_actions = env.action_space.n

def get_random_policy():
    P = np.random.randint(0, 4, 16)
    return P
    
def print_policy(policy):
    lake = "SFFFFHFHFFFHHFFG"
    arrows = ['<V>^'[a] for a in policy]
    signs = [arrow if tile in "SF" else tile for arrow, tile in zip(arrows, lake)]
    for i in range(0, 16, 4):
        print(' '.join(signs[i:i+4]))
        
def sample_reward(env, policy, t_max=100):
    s = env.reset()
    total_reward = 0
    is_game_over = False
    current_state = 0
    while not is_game_over:
        current_state, reward, is_game_over, _ = env.step(policy[current_state])
        total_reward += reward
    return total_reward
    
def evaluate(policy, n_times=100):
    """Run several evaluations and average the score the policy gets."""
    rewards = [sample_reward(env, policy) for _ in range(n_times)]
    return float(np.mean(rewards))
    
def crossover(policy1, policy2, p=0.5):
    """for each state, with probability p take action from policy1, else policy2"""
    mask = np.random.choice(2, size=16, p=[1-p, p])
    new_policy = policy1 * mask + policy2 * (1 - mask)
    return new_policy
    
def mutation(policy, p=0.1):
    '''
    for each state, with probability p replace action with random action
    Tip: mutation can be written as crossover with random policy
    '''
    mask = np.random.choice(2, size=16, p=[1-p, p])
    new_policy = get_random_policy() * mask + policy * (1 - mask)
    return new_policy
    
# hyperparams
n_epochs = 100 #how many cycles to make
pool_size = 100 #how many policies to maintain
n_crossovers = 50 #how many crossovers to make on each step
n_mutations = 50 #how many mutations to make on each tick

pool = [get_random_policy() for _ in range(pool_size)]
pool_scores = [evaluate(policy) for policy in pool]

best_evals = []
for epoch in range(20):
    print("Epoch %s:"%epoch)
    
    crossovered = [crossover(pool[np.random.randint(low=0, high=pool_size)], 
                             pool[np.random.randint(low=0, high=pool_size)])  
                   for _ in range(n_crossovers)]
    
    mutated = [mutation(pool[np.random.randint(low=0, high=pool_size)]) for _ in range(n_mutations)]
    
    assert type(crossovered) == type(mutated) == list
    
    #add new policies to the pool
    pool += crossovered + mutated 
    pool_scores = [evaluate(policy, n_times=100) for policy in pool]
    
    #select pool_size best policies
    selected_indices = np.argsort(pool_scores)[-pool_size:]
    pool = [pool[i] for i in selected_indices] # survive of the fittest; take the best from population
    pool_scores = [pool_scores[i] for i in selected_indices]

    #print the best policy so far (last in ascending score order)
    print("best score:", pool_scores[-1])
    best_evals.append(pool_scores[-1])
    print_policy(pool[-1])
    
env.close()
