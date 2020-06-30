# Overview
This project implements a Gaming Bot for the game Frozen Lake from OpenAI Gym, trained using Q Learning. 

## Introduction 
Generally, Reinforcement Learning is a family of machine learning techniques that allows us to create intelligent agents that learn from the environment by interacting with it. In doing so, they learn the *optimal policy* which would grant them the maximum future discounted rewards. This is useful in many real world applications where supervised learning might not be the best approach due to various reasons like nature of task itself, lack of appropriate labelled data, etc.
The important idea here is that this technique can be applied to any real world task that can be described loosely as a Markovian process.


## Approach - how it works. 
It is an AI based on Q learning from Reinforcement Learning. The Agent (game player) performs a certain *set of actions* in an Environment. The agent takes an *action (Moves up, down, left, or right)*. The environment then samples a *reward (points recieved by the player for moving in a specific direction)* and *next state (described location of the player on the grid)*. The player then receives the award and proceeds to the next state sampled by the environment. This process repeats itself until the player reaches a *terminal state*. The **greedy epsilon algorithm** has been used in order to take care of the exploration - exploitation tradeoff. 

## Results 
After training on 10000 episodes of games, the reward kept on increasing, on an average, after each episode as the Q table kept on filling up and converged to q* ie. our optimal reward-maximizing policy. 
At the end of 10000 episodes of exploration and exploitation, the algorithm concluded with a **reward of 0.71**. This meant that out of the 100 times that the player tried to play the game, he was able to win 71 times. 

## Mathematic Intuition 
- *Set of possible states* - S <br />
- *Set of possible actions* - A <br />
- *Distribution of award given (state,action) pair* - R <br />
- *Transition probability - distribution over next state given (state,action) pair* - P <br />
- *Discount facctor* - r <br />

We initialize the initial q table to 0. For every episode, we perform either exploration or exploitation according to the following equations :- 

For **Exploitation**, we use the *Bellman* equation :- <br /> <br />
<img src="https://render.githubusercontent.com/render/math?math=q^{*}(s,a) = R + \gamma max(q^{*}(s',a'))">

<br /> where *s'* is the state reached by the player after performing action *a* in state *s*



![Training results](/Images/rewards.png)

## Getting started 
In order to view this project, you will need to install the required python packages

```
pip install -r requirements.txt
```

Now you can open up the terminal and see how the player trains itself using : 

```
python frozenLakeGameQLearning.py
```

