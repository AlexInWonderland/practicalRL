import numpy as np
import os

import gym
from gym import wrappers

env = gym.make("FrozenLake-v0")
#filespath = 'frozenlake-v0-experiment-1/'
#os.system('rm -rf /tmp/gym-results/' + filespath)
#env = wrappers.Monitor(env, '/tmp/gym-results/' + filespath)

n_states = env.observation_space.n
n_actions = env.action_space.n

def get_random_policy():
    return np.random.randint(0,4,16)
    
def sample_reward(env, policy, t_max=100):
    s = env.reset()
    total_reward = 0
    t = 0
    game_over = False
    while not game_over and t <t_max: 
        s, reward, is_game_over, _ = env.step(policy[s])
        total_reward += reward
        t += 1
    return total_reward

def evaluate(policy, n_times=100):
    """Run several evaluations and average the score the policy gets."""
    rewards = [sample_reward(env, policy) for _ in range(n_times)]
    return float(np.mean(rewards))

def print_policy(policy):
    lake = "SFFFFHFHFFFHHFFG"
    arrows = ['<V>^'[a] for a in policy]
    signs = [arrow if tile in "SF" else tile for arrow, tile in zip(arrows, lake)]
    for i in range(0, 16, 4):
        print(' '.join(signs[i:i+4]))

#print("random policy:")
#print_policy(get_random_policy())

def crossover(policy1, policy2, p=0.5):
    """for each state, with probability p take action from policy1, else policy2"""
    mask = np.random.choice(2, size=16, p=[1-p, p])
    cross_policy = policy1 * mask + policy2* (1-mask)
    #cross_policy = []
    #for j in range(len(policy1)):  
    #   i = np.random.uniform(0,1)
    #   if i > p:
    #     cross_policy.append(policy2[j])
    #   else:
    #     cross_policy.append(policy1[j])
    return cross_policy
    
def mutation(policy, p=0.1):
    '''
    for each state, with probability p replace action with random action
    Tip: mutation can be written as crossover with random policy
    '''
    m_policy = crossover(policy, get_random_policy(), p)
    return m_policy


n_epochs = 100 #how many cycles to make
pool_size = 100 #how many policies to maintain
n_crossovers = 50 #how many crossovers to make on each step
n_mutations = 50 #how many mutations to make on each tick
print("initializing...")
pool = [get_random_policy() for _ in range(pool_size)]
pool_scores = [evaluate(p) for p in pool]
#main loop
for epoch in range(n_epochs):
    print("Epoch %s:"%epoch)
    
    
    crossovered = [crossover(pool[np.random.randint(low=0, high=pool_size)], 
                             pool[np.random.randint(low=0, high=pool_size)]) 
                             for _ in range(n_crossovers)]
    
    mutated = [mutation(pool[np.random.randint(low=0, high=pool_size)]) for _ in range(n_mutations)]
    assert type(crossovered) == type(mutated) == list
    
    #add new policies to the pool
    pool = pool + crossovered + mutated
    pool_scores = [evaluate(p) for p in pool]
    
    #select pool_size best policies
    selected_indices = np.argsort(pool_scores)[-pool_size:]
    pool = [pool[i] for i in selected_indices]
    pool_scores = [pool_scores[i] for i in selected_indices]

    #print the best policy so far (last in ascending score order)
    print("best score:", pool_scores[-1])
    print_policy(pool[-1])
