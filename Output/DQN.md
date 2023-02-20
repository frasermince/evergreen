---
category: "post"
stage: "budding"
date: 01/25/23
---
# DQN
One of the first algorithms you might encounter when learning about deep reinforcement learning is DQN. As one of the fundamental algorithms in RL, DQN (Deep Q-Network) has paved the way for many advancements in the field. 

In this blog post I will guide you through the mechanics of the DQN algorithm and explore some of its nuances. In this post I will assume a basic understanding of the standard RL problem and it's terminology, if you are not familiar with terms such as environments, policies, actions, etc recommend checking out the comprehensive [OpenAI Spinning up in RL Guide](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).

## What is DQN

DQN or Deep Q-Network is an off policy, model free algorithm that maximizes the Q value, or state action value, in order to improve in a given environment. The state action value is how valuable a state is given you take a specific action. This value will usually be represented as $Q(s,a)$.

Because it is off policy we can reuse data from a stored memory buffer despite it not being from the current action distribution. Model free means it works without needing to understand a model of how the environment changes when we take an action.

## Bellman Equation
Some of the most fundamental formulas in the field of RL are the Bellman Equations. These are recursive relationships that show how the value of the state you are in now relate to values at future states. For this given problem we care about the off policy state action value Bellman Equation. This is defined as follows:

$$
Q^*(s,a) = \mathop{\mathbb{E}}_{s' \sim \epsilon}[r(s,a) + \gamma  \max_{a'}Q^*(s',a')|s,a]
$$
Where $\epsilon$ refers to the environment, $Q$ refers to the state action value, $r$ refers to the reward at a given timestep, and $\gamma$ refers to the discount rate for future values. We discount future reward because reward now is more valuable than reward in the future. If we do not do this it is possible we learn to prefer reward infinitely far in the future.

So this equation shows that optimal Q value is equal to the reward now plus the discounted Q value of the optimal action of the next step. 

## Loss Function
Due to the recursiveness of the Bellman operator above we can minimize error in predicting Q values much like we would do in a supervised setting. 

$$
L_{i}(\theta) = \mathop{\mathbb{E}}_{s,a \sim p(\cdot)}[(y_{i} - Q(s,a))^2]
$$
Specifically what this means is that we sample the state and action used for our Q value from $p(\cdot)$ which is a behavior distribution of states and actions. In practice we will estimate this expectation over the distribution by sampling and using standard SGD descended optimizers found in deep learning frameworks. 

Instead of calculating a full expectation of the behavior we will sample from previous steps taken in the environment so our neural net can predict the Q value. This sampling of previous timesteps is possible because we are in an off policy setting where we do not care if we are using data from previous versions of a policy. The set of previous steps from the environment is called the Replay Memory.

$y_i$ in the above equation is:
$$
y_{i} = \mathop{\mathbb{E}}_{s' \sim \epsilon}[r + \gamma \max_{a'}Q(s',a';\theta_{i - 1}| s,a)]
$$
So we are taking the reward of the current state plus the Q value of the next state. In our error formula above we are then subtracting the current Q value. The difference between these two values should be portion of $Q(a,s)$ that accounts for the $r$ reward at the current state.

The biggest difference from supervised learning is the target itself also depends on network weights i.e. the very parameters we are changing are used in calculating the target. Due to the concerns about training stability that arise from this fact it is very common to use a periodically synced target network to derive the $y_{i}$ target values. So perhaps the parameters of this target network are synced every 1000 steps to have the same values as the main network.

Some simple pytorch pseudocode to demonstrate the main training loop looks like this:
```python
(first_observation, reward, actions, second_observation) = sample_from_replay_memory()
target_network_result = forward(second_observation, target_network)
target_value, action = torch.max(target_network_result, 1)

yj = reward + discount_factor * target_value
current_predicted_reward = forward(first_observation, network).choose(action)
loss = nn.MSELoss()(current_predicted_reward, yj)
```

There's a few additional things to be aware of. In particular you want to prevent calculating gradients for the network calls neccesary to compute the gradient. Adding this in would look something like this:

```python
(first_observation, reward, actions, second_observation) = sample_from_replay_memory()
with torch.no_grad():
	target_network_result = forward(second_observation, target_network)
	target_value, action = torch.max(target_network_result, 1)

yj = reward + discount_factor * target_value
current_predicted_reward = forward(first_observation, network).choose(action)
loss = nn.MSELoss()(current_predicted_reward, yj)
```

You will also need a interaction loop to collect the experiences from the environment. The simplest way of doing this will involve an exploration strategy called epsilon greedy. Where if you are under some random number you choose randomly and otherwise you choose the action that maximizes the q value from the main network.
```python
if random.uniform(0, 1) <= epsilon:
    action = self.env.action_space.sample()
else:
    values, action = torch.max(self(first_observation, self.network), 1)
    action = action.item()

second_observation, reward, is_done, info = env.step(action)
memory.append(state_tuple)
```

You would then store the timesteps in a format to access later.

```python
state_tuple = (self.observation, reward, action, is_done, second_observation)
```

You could then have a loop that does a step in the environment and trains every n steps

```python
i = 0
steps_to_train = 4
while True:
	play_step()
	if i % steps_to_train == 0:
		train_network() 
	i += 1
```

With this we have covered the biggest pieces of DQN. Feel free to checkout out [my implementation](https://github.com/frasermince/rl-papers/tree/master/dqn/dqn.py) and reach out if you have any thoughts or questions.

