---
category: "post"
---
# Muzero
An algorithm I spent a significant time implementing was Muzero and I want to explain it here.


# What is Muzero?
Muzero is in the AlphaGo and AlphaZero family of algorithms. It adepts the monte carlo tree search (MCTS) approach of the previous algorithms in its family and adds in a learned model of the environment. This means it can act in environments where a known model is not present. The previous iteration, AlphaZero,  worked specifically with games where we had a known model of the environment. So it would work in environments such as chess, shogi, and go. But something more ill-defined such as Atari games would never work. Muzero removes this limitation. 

## Monte Carlo Search Tree

The basic idea of muzero and any of the MCTS methods is to simulate a search in the environment n times. Where n is a parameter chosen depending on the game. Utilizing what is known or predicted about the model we visit possible future nodes and build a visit policy based upon this information.

In muzero specifically we explore possible next nodes based upon the following formula:
$$

\DeclareMathOperator*{\argmax}{argmax}

a^k = \argmax_{a}\left[ Q(s,a) + P(s,a) \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)} \left( c_{1} + \log \frac{\left( \textstyle\sum_{b} N(s,b) + c_{2} + 1 \right)}{c_{2}} \right)\right]
$$

with c1 and c2 being hyper parameters set to c1 = 1.25 and c2 = 19652. Q is the state action value function predicted by our network, and p is some policy also predicted by our network.

In previous iterations of the AlphaGo family we could do this with perfect information and with less need to predict since the dynamics of the game were known.

## Learned Model

Muzero is unique because it is predicting the model of the environment using three learned networks usually with a shared base network. These three networks are called representation, dynamics, and prediction.

### Representation
Receives past observations and actions 1..t and returns the hidden state, an encoded representation of the game state.

### Dynamics
Receives the previous hidden state and the action we're are taking next and produces the immediate reward and a new hidden state.

### Prediction
Takes a hidden state and produces a policy and a value function.
