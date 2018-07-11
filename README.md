# Mountain Car with SARSA Function Approximation

This repository contains two projects:

1. Using SARSA with linear function approximation to solve the Mountain Car problem.
2. Using Actor Critic to solve the Continuous Mountain Car problem. 

<img src="https://i.imgur.com/NYMsQqX.png" width="500"/>

Mountain car is one of the most popular reinforcement learning test environemts. The agent must learn use the momentum gained by rolling down the hills to reach the goal. It has a continous state space with a discrete set of actions (left, right, and do nothing). I used a linear combination of a feature vector and set of weights to approximate the state-action value function Q. The state samples were transformed into a higher dimensional space using an approximation of an RBF kernel allowing for a non-linear value function.

Simply run mountaincar.py and comment out the env.render() as necessary to toggle visualization. I have also included optional methods for gradient checking and plots of action choices, value function, and rewards. 

It would take little effort to turn this into a Q-Learning solution.

For the continuous environment, the agent tends to converge to the local optima of choosing not to move. This is beacuse the agent is given negative rewards for each action taking and a reward of 0 is better than a reward of -99. The agent only works when it discovers the optimal policy of reaching the flag in the first episode. I will have to come back to this project when I have a better exploration strategy.


