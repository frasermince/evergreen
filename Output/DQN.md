---
category: "post"
---
*Introduction to DQN*
One of the first algorithms I implemented upon starting on my Reinforcement Learning journey is DQN. I want ot walk you through how it works and some of the nuances. This post is going to assume you know the very basics of RL but if not feel free to check out the [OpenAI Spinning up in RL Guide](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

*What is DQN*
DQN or Deep Q Networks is an [[Off Policy]], model free algorithm that predicts the Q value, or state action value, in order to improve in an environment. The state action value is how valuable a state is given that you take a specific action.

*Bellman Equation*
The action value Bellman equation is as follows
$$
Q^*(s,a) = \mathop{\mathbb{E}}_{s' \sim \epsilon}[r + \gamma  \max_{a'}Q^*(s',a')|s,a]
$$
Where $\epsilon$ refers to the environment, $Q$ refers to the state action value, $r$ refers to the reward at a given timestep, and $\gamma$ refers to the discount rate for future values.

The naive approach would be to iteratively estimate the Q value using the Bellman equation. This value iteration method does converge to the optimal solution but in practice is infeasible. This is because the Q value is estimated separately each time without potential for generalization. 

*Loss Function*
Due to the recursiveness of the Bellman operator above we can minimize error in predicting Q values much like we would do in a supervised setting. 


$$
L_{i}(\theta) = \mathop{\mathbb{E}}_{s,a \sim p(\cdot)}[(y_{i} - Q(s,a))^2]
$$
Specifically what this means is that we sample the state and action used for our Q value from $p(\cdot)$ which is a behavior distribution of states and actions. In practice we will replace this expectation over the distribution with the familiar sampling approaches that work well with standard neural nets. So instead of calculating a full expectation of the behavior we will sample from previous steps taken in the environment and use that to approximate the distribution. This sampling of previous timesteps is a crucial part of the algorithm and the set of previous steps from the environment is called the "Replay Memory".

$y_i$ in the above equation is:
$$
y_{i} = \mathop{\mathbb{E}}_{s' \sim \epsilon}[r + \gamma \max_{a'}Q(s',a';\theta_{i - 1}| s,a)]
$$
So we are taking the reward of the current state plus the Q value of the next state. In our error formula above we are then subtracting the current Q value. The difference between these two values should be portion of $Q(a,s)$ that accounts for the $r$ reward at the current state.

The biggest difference from supervised learning is the target itself also depends on network weights i.e. the very parameters we are changing are used in calculating the target. Due to the concerns about training stability that arise from this fact it is very common to use a periodically synced target network to derive the target values. So perhaps the parameters of this target network is synced every 1000 steps to have the same values as the main network.

Some simple pytorch code to show the main training loop looks like this:
```python
(first_observation, reward, actions, second_observation) = sample_from_replay_memory()
target_network_result = forward(second_observation, target_network)
target_value, action = torch.max(target_network_result, 1)

yj = reward + discount_factor * target_value
current_predicted_reward = forward(first_observation, network).choose(action)
loss = nn.MSELoss()(current_predicted_reward, yj)
```

There's a few additional caveats. For example you would want to prevent gradients for the portion of the loss calculating that computes the target. That would look something like this

```python
(first_observation, reward, actions, second_observation) = sample_from_replay_memory()
with torch.no_grad():
	target_network_result = forward(second_observation, target_network)
	target_value, action = torch.max(target_network_result, 1)

yj = reward + discount_factor * target_value
current_predicted_reward = forward(first_observation, network).choose(action)
loss = nn.MSELoss()(current_predicted_reward, yj)
```

You will also need separate code to collect the actual experiences from the environment. This will often be done using an epsilon greedy strategy. Basically if you are under some random number you choose randomly and otherwise you choose the action that maximizes the q value from the main network.
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


These present the biggest pieces involved in DQN. Feel free to checkout out [my code](https://github.com/frasermince/rl-papers/tree/master/dqn/dqn.py) and reach out if you have any thoughts or questions.

#budding 