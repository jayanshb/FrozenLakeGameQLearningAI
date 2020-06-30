import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# creating the enviroment
env = gym.make("FrozenLake-v0")

action_set_size = env.action_space.n
state_set_size = env.observation_space.n

q_table = np.zeros((state_set_size, action_set_size))

# setting up hyperparameters
n_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Q-Learning algo.
rewards_all_episodes = []

for episode in range(n_episodes):
    state = env.reset()

    # wether episode is over or not
    done = False
    rewards_current_episode = 0
    # until current episode ends
    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        # exploitation condition
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        # exploration condition
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # Update Q(s,a) in Q Table
        # q(new)(s,a) = (1-a)q(s,a) + a(R_t + r*max(q(s',a')))
        q_table[state, action] = q_table[state, action]*(1 - learning_rate) + \
            learning_rate * (reward + discount_rate *
                             np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward
        if done == True:
            break

    # Exploration rate decays after one episode gets over
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)

# training results
rewards_per_thousand_episodes = np.split(
    np.array(rewards_all_episodes), n_episodes / 1000)
count = 1000
print("********* Average reward per thousand episodes *********")
for reward in rewards_per_thousand_episodes:
    print(count, ":", str(sum(reward/1000)))
    count += 1000

# Print final Q Table
print("\n\n***** Q table *****\n")
print(q_table)

# Game UI
for episode in range(3):
    state = env.reset()
    done = False
    print("***** EPISODE ", episode+1, "***** \n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(3)

        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("**** You reached the goal ****")
                time.sleep(3)

            else:
                print("**** You fell through a hole ****")
                time.sleep(3)
            clear_output(wait=True)
            break

        state = new_state

env.close()
