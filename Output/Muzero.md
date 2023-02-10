---
category: "post"
---
# Muzero
In this article I will give an overview of Muzero. An algorithm used for performing well at Go, Chess, Shogi, and Atari. It was the state of the art in each of these domains for a while and has some very interesting ideas worth exploring.

## What is Muzero?

Muzero is in the AlphaGo and AlphaZero family of algorithms. It adopts the Monte Carlo Tree Search (MCTS) approach of the previous algorithms in its family and adds in a learned model of the environment. This means it can act in environments where a known model is not present. The previous iteration, AlphaZero,  did not work in environments without a learned model. So it could work in environments such as chess, shogi, and go. But something with a less defined dynamics model such as Atari games would not work. Muzero removes this limitation. 


To fully grasp Muzero there are a few things we need to understand:
1. AlphaGo family history
2. Monte Carlo Tree Search
	1. Action Selection Formula
	2. Visit count and Q value updates
	3. Visit Policy Formula
3. How Muzero learns it's environment
	1. Representation Network
	2. Dynamics Network
	3. Prediction Network
4. Muzero loss calculation


## The AlphaGo Algorithm Family
So what is the AlphaGo family of algorithms? In short it was created to create a computer program that can beat grand masters at the game of Go. Go was long considered to be a game unsolvable by computers. Chess itself is very difficult for computers due to the large search space of it's moves. For each move for each piece in the game of chess that is a possibility you would have to search. 

The search space of chess is estimated to be 10^50. Go on the other hand has a a branching factor of 250. The search space of Go is estimated to be 10^170. A whole googol more complex than the game of chess. The naive approach to such a problem would be an exhaustive search but such a large search space makes this impossible. More traditional methods relied heavily upon pruning unpromising parts of the search tree to reduce the search tree size. However knowing what to prune is not easy. The combination of an algorithm called Monte Carlo Search Tree algorithm with neural nets made AlphaGo a reality. So what is Monte Carlo Tree Search (MCTS)?


## Monte Carlo Tree Search

MCTS follows a Monte Carlo simulation. Where it chooses actions to take according to an upper confidence bound. This upper confidence bound characterizes the tradeoff between exploration and exploitation. Or the need to look at something new compared to the desire to take advantage of what we already know.

The basic idea of muzero and any of the MCTS methods is to simulate a search in the environment n times. Where n is a parameter chosen depending on the game. Utilizing what is known or predicted about the model we visit possible future nodes and build a visit policy based upon this information. We build up knowledge of how well explored an action is and how valuable it is over each subsequent simulation.

In muzero specifically we choose possible next state to explore with the following formula
$$
a^k = \argmax_{a}\left[ Q(s,a) + P(s,a) \cdot\frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)}\cdot \left( c_{1} + \log \left(\frac{\left( \textstyle\sum_{b} N(s,b) + c_{2} + 1 \right)}{c_{2}}\right) \right)\right]
$$

Where the first part $Q(s,a)$ controls exploitation and the rest controls exploration.
*Factored out this would be:*
$$

a^k = \argmax_{a}\left[ Q(s,a) + \left( P(s,a) \cdot \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)} \cdot c_{1} + P(s,a) \cdot \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)} \cdot \log \left( \frac{\left( \textstyle\sum_{b} N(s,b) + c_{2} + 1 \right)}{c_{2}} \right) \right)\right]
$$


With $c_{1}$ and $c_{2}$ being hyper parameters set to $c_{1} = 1.25$ and $c_{2} = 19652$. $Q(s,a)$ is the state action value function predicted by our network, and $P(s,a)$ is some policy also predicted by our network.

In this formula $c_{1}$ controls the tradeoff between exploiting the value $Q(s,a)$ and further exploration. While $c_{2}$ controls a slowly increasing ratio that increases exploration as more nodes are visited.

I feel like this formula was difficult for me to wrap my head around. So let's break it down a little bit. A lot of the added complication in this formula is due to scaling the policy times the visit count ratio $P(s,a) \cdot \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)}$

## Action Selection Formula Broken Down

I think it's instructive to work through what this would look like in practice so lets pretend we are solving this formula in the full atari environment where there are 18 action choices. Let's assume we have yet to visit any actions in this case the formula, ignoring for now the policy and Q value, for any action a would look like this.

$$
  P(s,a) \cdot \frac{0}{1 + 0} \cdot 1.25 + P(s,a) \cdot \frac{0}{1 + 0} \cdot \log \left( \frac{\left( 0 + 19652 + 1 \right)}{19652} \right)
$$
Certainly not very interesting yet the formula equals zero plus the Q value of each node. That q value will also be initialized to zero so the action chosen to explore next will be entirely random. Now let's see what this formula looks like upon visiting the first node.

*For that first node already chosen:*
$$
  Q(s,a) + \left(P(s,a) \cdot \frac{\sqrt{ 1 }}{1 + 1} \cdot 1.25 + P(s,a) \cdot \frac{\sqrt{ 1 }}{1 + 1} \cdot \log \left( \frac{\left( 1 + 19652 + 1 \right)}{19652} \right)\right)
$$
*Reduced to:*
$$
Q(s,a) + \left(P(s,a) \cdot 0.625 + 0.5 \cdot P(s,a) \cdot 0.000101765633 \right)
$$
*For other nodes:*
$$
  Q(s,a) + \left(P(s,a) \cdot \frac{\sqrt{ 1 }}{1 + 0} \cdot 1.25 + P(s,a) \cdot \frac{\sqrt{ 1 }}{1 + 0} \cdot \log \left( \frac{\left( 1 + 19652 + 1 \right)}{19652} \right)\right)
$$
*Reduced to:*
$$
Q(s,a) + \left(P(s,a) \cdot 1.25 + 1 \cdot P(s,a) \cdot 0.000101765633 \right)
$$

So as you can see nodes that are chosen less have their P value weigh more highly making them more likely to be chosen.

Now lets look at the case where each node has been chosen once except the final node. 
*The final node formula would be:*
$$
  Q(s,a) + \left(P(s,a) \cdot \frac{\sqrt{ 15 }}{1 + 0} \cdot 1.25 + P(s,a) \cdot \frac{\sqrt{ 15 }}{1 + 0} \cdot \log \left( \frac{\left( 15 + 19652 + 1 \right)}{19652} \right)\right)
$$

*Reduced to:*
$$
Q(s,a) + \left(P(s,a) \cdot 4.84123 + 3.872983346207417 \cdot P(s,a) \cdot 0.000813835243 \right)
$$

*Others would be:*
$$
  Q(s,a) + \left(P(s,a) \cdot \frac{\sqrt{ 15 }}{1 + 1} \cdot 1.25 + P(s,a) \cdot \frac{\sqrt{ 15 }}{1 + 1} \cdot \log \left( \frac{\left( 15 + 19652 + 1 \right)}{19652} \right)\right)
$$

*Reduced to:*
$$
Q(s,a) + \left(P(s,a) \cdot 2.4206145914 + 1.9364916731 \cdot P(s,a) \cdot 0.000813835243 \right)
$$

Note that as other actions are chosen more we value the policy of the unchosen action more and more. Also note that as the total child action count increases more and more we value more the log term including the added policy more. Meaning in practice we value the $P(s,a)$ a bit more compared to $Q(s,a)$. We can even see this taken to an extreme. Let's assume each action has been chosen 30 times except one which has been chosen 29 times. 

*The unchosen action:*

$$
  Q(s,a) + \left(P(s,a) \cdot \frac{\sqrt{ 539 }}{1 + 29} \cdot 1.25 + P(s,a) \cdot \frac{\sqrt{ 539 }}{1 + 29} \cdot \log \left( \frac{\left( 539 + 19652 + 1 \right)}{19652} \right)\right)
$$

*Reduced to:*
$$
Q(s,a) + \left(P(s,a) \cdot 0.967349 + 0.77387912 \cdot P(s,a) \cdot 0.02710737 \right)
$$

As you can see the log term has now gotten much larger and plays a much larger part of the algorithm.

As stated above most of the complication from this formula is around scaling the various terms. In fact I have seen a simpler version of the formula suggested [in a related paper](https://arxiv.org/pdf/2007.12509.pdf) without this scaling as:

$$
a^k = \argmax_{a}\left[ Q(s,a) + c_{1} \cdot P(s,a) \cdot\frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)}\right]
$$

This is very close in practice to the formula used for Muzero. Just without the gradual scaling that occurs as more actions take place.

We end each simulation when an unexplored leaf node has been found. We add this node to the search tree with the computed reward and state. We then update the visit count and Q value of nodes visited within the simulation.

$$
Q(s^{k-1},a^k) := \frac{N(s^{k-1},a^k) \cdot Q(s^{k-1}, a^k) + G^k}{N(s^{k-1},a^k) + 1}
$$
$$
N(s^{k-1},a^k) := N(s^{k - 1}, a^k) + 1
$$

Where $G^k$ is an estimate from $l$ - $k$ estimate of the cumulative discounted reward boostrapped from the value function:

$$
G^k = \sum^{l - 1 - k}_{\tau = 0}\gamma^{\tau}r_{k+1+\tau} + \gamma^{l - k}v^l
$$

In previous iterations of the AlphaGo family we would do all this with in an environment where we would always know exactly how the environment would change due to our actions. The central innovation of Muzero is to adapt this algorithm to environments, like atari, where this is not true.

## Learned Model

Muzero is unique because it is not just building a search tree to find optimal actions but also predicting the dynamics of it's environment. As such it has three learned networks with a shared base network. These three networks are called representation, dynamics, and prediction.

### Representation

Receives past observations and actions 1..t and returns the hidden state, an encoded representation of the game state.

### Dynamics

Receives the previous hidden state and the action we're are taking next and produces the immediate reward and a new hidden state.

### Prediction
Takes a hidden state and produces a policy and a value function.
