---
category: "post"
---
In recent months, rumors of a new multimodal LLM from Google DeepMind called Gemini have been circulating. According to an interview in Wired, the head of DeepMind said this new LLM will use some of the fundamental "techniques used in AlphaGo, aiming to give the system new capabilities such as planning or the ability to solve problems." But what is AlphaGo? What does it actually do? In this post I want to go into detail on the different iterations of the algorithm, with particular emphasis on a more general version, [MuZero](https://www.deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules).

In 2017, DeepMind garnered significant attention in the artificial intelligence community when its algorithm, AlphaGo, defeated the world champion in Go, Ke Jie. This achievement was especially notable as the intricacies of Go had long made such a feat seem nearly insurmountable. In the ensuing years, DeepMind refined its initial design, incorporating self-play learning, expanding its proficiency to Chess and Shogi, and mastering visually intricate Atari games. The most recent iteration, MuZero, transcends previous constraints by eliminating the need for known models, empowering it to operate in environments with unknown transition dynamics (such as Atari).

Looking for a place to hone my research skills and continue to improve my grasp of Reinforcement Learning, I made it a goal of mine to understand and implement the MuZero algorithm. As I began to read and implement the [MuZero paper](https://arxiv.org/abs/1911.08265), it became evident that there was a significant amount of background knowledge the authors had that I lacked. There are a lot of moving pieces and ideas that are assumed to be understood. What follows is what I wish I knew before attempting to reimplement MuZero. I have also tried to explain some of the history that led to the development of MuZero. This will help clarify some of the terms used and help bring understanding to the AlphaGo family's central innovations.

## What is MuZero?

MuZero is in the same family of algorithms as AlphaGo and AlphaZero. It adopts the Monte Carlo Tree Search (MCTS) approach of the previous algorithms in its family and adds a learned model of its environment. Meaning it predicts how the environment will respond to actions through a representation it learns over time.

The previous iterations required environments where the transition dynamics were known. This worked well in environments such as Chess, Shogi, and Go, where the exact transitions are known. If you move a piece in chess you know exactly what the board looks like afterward. Any variability or noise in how the state would change post-action (as in Atari games) made an environment a poor fit for these algorithms. MuZero removed this limitation and, at the time, achieved state-of-the-art on the Atari suite.

To gain a full understanding of MuZero I have broken down the algorithm into smaller more digestible pieces:
1. AlphaGo family history
2. Monte Carlo Tree Search
	1. Selection
	2. Expansion
	3. Backup
3. How MuZero learns it's environment
	1. Representation Network
	2. Dynamics Network
	3. Prediction Network
4. Data representation specifics 
	1. Action support representation
	2. Invertible support transformation
	3. Rollouts
5. MuZero training specifics 
	1. Prioritized replay sampling
	2. MuZero loss formula

In addition to some advances since then that I hope to explore in the future:
1. MuZero Reanalyze
2. EfficientZero
3. MCTS as regularized policy optimization
4. Muesli

## The AlphaGo Algorithm Family

So what is the AlphaGo family of algorithms? As stated above they were developed to create a computer program that can beat grandmasters at the game of Go. Go was long considered a game that was intractably hard for computers to win at while playing experts.

Chess itself is very difficult for computers due to its large search space of moves. For each move for each piece in the game of chess that is a possibility you would have to search. 

The search space of chess is estimated to be 10^50. Go has a much larger set of valid moves and thus a much larger branching factor coming in at 250. The search space of Go is estimated to be 10^170. This comes out to a whole googol more complex than the game of chess. 

The naive approach to such a problem would be an exhaustive search but such a large search space makes this impossible. More traditional methods relied heavily on pruning unprosmising parts of ithe search tree zto reduce the e. However knowing what to prune is not easy.

AlphaGo's genius was combining a Monte Carlo Search Tree (MCTS) algorithm with value and policy functions estimated by neural nets. So what is Monte Carlo Tree Search (MCTS)?

## Monte Carlo Tree Search

In Monte Carlo Tree Search we simulate taking actions from the current point in the environment $n$ times. We will use some action selection criteria and continue the simulation choosing actions down the tree until we reach an unexplored leaf node. As we take more simulations information about the environment such as visit count and value of each state accumulates biasing which action we will visit next. At the end of $n$ simulations we use this information to create a search policy.

MCTS follows a Monte-Carlo rollout in a non-uniform way. Instead of uniformly sampling actions as done in vanilla Monte-Carlo planning, it uses an upper confidence bound to bias which action to simulate next. The MCTS algorithm used in the AlphaGo family is based off of the UCT algorithm (or Upper Confidence Bound for Trees) and the PUCB algorithm (Predictor + UCB). It is considered to be a modified (now renamed from the original paper) pUCT (Predictor + Upper Confidence Bound for Trees) rule.

### UCT Algorithm
For the simplest example of the [UCT algorithm](http://ggp.stanford.edu/readings/uct.pdf) practice we will look towards the setting of One Armed Bandits. For the Bandit Problem imagine a bunch of slot machines, each with their own reward distribution. Each time we pull the arm we get a chance of rewards based on the reward distribution. From timestep to timestep this distribution is the same so we simplify the problem and remove the need to consider how timesteps effect each other. Then we can focus instead on how we balance exploring the environment with exploiting the high value machines we have already explored. This is where an upper confidence bound becomes useful.

UCT uses the biased Monte Carlo rollout to plan which slot machines to exploit and explore and uses a very simple upper confidence bound called UCB1 to choose which one to try next. The simple algorithm follows the following steps.
* Examine current state
* If meets some terminal condition return 0
* If is an unexplored leaf return state action value.
* Otherwise select next arm to pull $X_{it}$ with the selection formula
* Simulate action
* Update average value with the reward times a discounted recursive call on the next action search

#### Selection Formula
$$
I_{t} = \argmax_{i \in \{ 1,\dots, K \}} [\overline{X}_{i, T_{i}(t-1)} + c_{t-1, T_{i}(t-1)}]
$$
Where $\overline{X}_{i, T_{i}(t-1)}$ is the average reward for bandit up to timestep $t - 1$,  $T_{i}(t-1)$ is the number of times an arm has been chosen so far, and $c_{t}$ is a bias sequence that weighs which action to choose based on how many times it has been chosen:
$$
c_{t,s} = \frac{\sqrt{ 2\ln t }}{s}
$$

So the formula in a more plainly named coding setting would be:

```python
argmax(lambda i: average_reward[i] + sqrt(2 * ln(timestep)) / times_chosen[i])
```

So in short we are doing a simple tradeoff between total number of actions taken and how many times our action has been taken. 

### PUCB

[PUCB](http://gauss.ececs.uc.edu/Workshops/isaim2010/papers/rosin.pdf) is an evolution of the UCB algorithms that uses contextual information as a predictor during action selection specifically in the environment of Go. The purpose of this change is to reduce the worst case regret of action selection. Regret is a measure of how costly it is to choose a suboptimal action and is calculated by taking the value of the optimal action minus the value of the action taken. If you want the exact formula for UCT regret and PUCB regret I suggest you check out their [respective](http://ggp.stanford.edu/readings/uct.pdf) [papers](http://gauss.ececs.uc.edu/Workshops/isaim2010/papers/rosin.pdf).

There are a lot of proofs in this paper that are not too relevant for our interests but the main idea we do care about that is introduced is the idea of using a probablity distribution as a predictor to bias which actions we take. This becomes directly relevant in the modified (and now renamed) pUCT rule in MuZero.

### MuZero Selection

In MuZero we choose the next action to simulate with the following pUCT rule.
$$
a^k = \argmax_{a}\left[ Q(s,a) + P(s,a) \cdot\frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)}\cdot \left( c_{1} + \log \left(\frac{\left( \textstyle\sum_{b} N(s,b) + c_{2} + 1 \right)}{c_{2}}\right) \right)\right]
$$
As you can see we are now utilizing a probability distribution $P(s,a)$ as a predictor to bias which action we take. We start by preferring actions with lower visit counts and high probabilities but over time the $Q(s,a)$, or the state action value, will have more weight. This means we first explore proportional to the probability of the policy but as simulations continue we begin to exploit the value $Q(s,a)$ more and more.


*Factored out this would be:*
$$

a^k = \argmax_{a}\left[ Q(s,a) + \left( P(s,a) \cdot \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)} \cdot c_{1} + P(s,a) \cdot \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)} \cdot \log \left( \frac{\left( \textstyle\sum_{b} N(s,b) + c_{2} + 1 \right)}{c_{2}} \right) \right)\right]
$$

With $c_{1}$ and $c_{2}$ being hyper parameters set to $c_{1} = 1.25$ and $c_{2} = 19652$. $Q(s,a)$ is the state action value function predicted by our network, and $P(s,a)$ is some policy also predicted by our network.

In this formula $c_{1}$ controls the tradeoff between exploiting the value $Q(s,a)$ and further exploration. While $c_{2}$ controls a slowly increasing ratio that increases exploration as more nodes are visited.

I feel like this formula was difficult for me to wrap my head around. So let's break it down a little bit. A lot of the added complication in this formula is due to scaling the policy times the visit count ratio $P(s,a) \cdot \frac{\sqrt{ \textstyle\sum_{b} N(s,b)}}{1 + N(s,a)}$

### Action Selection Formula Broken Down

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

This is very close in practice to the formula used for MuZero. Just without the gradual scaling that occurs as more actions take place. 

### Expansion

Once we reach a state not yet added to the tree we add compute the reward and state from the dynamics function and the policy and value from the prediction function. We store all of this information in a new node in the search tree. The node is initialize with a visit count $N(s^{l},a)$ of 0, a state action value $Q(s^{l},a)$, and a policy $P(s^l, a)$ from the policy $p^l$. This means that the dynamics and prediction function are only called once per simulation upon expansion.

### Backup
Upon adding the new node we then backup the nodes along the path we traversed in the simulation. We update the visit count and Q value of nodes visited within the simulation using the following formula.

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

I felt like from the paper I was unclear on what this select, expand, backup process actually looks like in practice. Below is some pseudocode mostly taken from the excellent [MuZero General](https://github.com/werner-duvaud/muzero-general) to hopefully make it a bit more clear.

```python
root_node = initialize_root()
for _ in range(num_simulations):
	node = root
	search_path = [node]
	# NODE SELECTION STEP
	while node.expanded():
		action, node = self.select_child(node)
		search_path.append(node)
	parent = search_path[-2]

	# Call our neural net Dynamics and Prediction functions
	new_reward, new_hidden_state = dynamics_function(
		parent.hidden_state,
		action
	)
	(value, policy_logits), _ = prediction_net(parent.hidden_state)

	# NODE EXPANSION STEP
	# Add nodes for each of the children of the expanded node 
	policy_distribution = softmax(policy_logits)

	for action_index, policy_prob in enumerate(policy_distribution):
		node.children[action_index].probability = Node(p)
	

```

In previous iterations of the AlphaGo family we would do all this with in an environment where we would always know exactly how the environment would change due to our actions. The central innovation of MuZero is to adapt this algorithm to environments, like atari, where this is not true.

## Learned Model

MuZero is unique because it is not just building a search tree to find optimal actions but also predicting the dynamics of it's environment. As such it has three learned networks with a shared base network. These three networks are called representation, dynamics, and prediction.

### Representation

Receives past observations and actions 1..t and returns the hidden state, an encoded representation of the game state.

### Dynamics

Receives the previous hidden state and the action we're are taking next and produces the immediate reward and a new hidden state.

### Prediction
Takes a hidden state and produces a policy and a value function.
