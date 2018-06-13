# Mountain Car with SARSA Function Approximation

Using the SARSA algorithm linear function approximation to solve the Mountain Car problem.  

<img src="https://i.imgur.com/NYMsQqX.png" width="500"/>

Mountain car is one of the most popular reinforcement learning test environemts. It has a continous state space with a finite set of actions (left, right, and do nothing). I used a linear combination between a feature vector and set of weights to approximate the state-action value function Q. The state samples were transformed into a higher dimensional space using an approximation of an RBF kernel allowing for a non-linear value function. 

Simply run mountaincar.py and comment out the env.render() as necessary to toggle visualization. I have also included optional methods for gradient checking and plots of action choices, value function, and rewards. 
