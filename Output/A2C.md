---
category: "post"
---
Building from the foundation of Q learning we learned in the last article we now explore an on policy algorithm using policy gradients and actor critic methods. DQN relied upon a experience replay memory in order to get it's results. This makes it an Off Policy algorithm. 

An Off Policy algorithm is one that can rely upon previous memories to update the parameters. The benefit of this is the memory replay often used in these algorithms reduces the effects of the non-stationarity nature of RL i.e. the fact the target also relies upon the function approximator itself. It also will de-coorelates updates so that we are training on data that is randomly sampled instead of all adjacent. Off Policy methods also have the benefit of being far more sample efficient due to being able to reuse old data.

In contrast an On Policy algorithm only relies upon data from the most recent policy. The benefit of an On Policy method is that you are directly optimizing the thing you actually care about. A value function like in DQN can act indirectly to optimize towards a good policy but there are more failure modes and less stability due to this indirectness.

So it would be advantageous to directly optimize for the thing we care about, namely the policy. But how do we do this without losing stability?

## Basic Policy Gradient
We want to be able to optimize the reward over a trajectory $\tau$. A trajectory being a series of states, actions, and rewards.

So:
$$
R(\tau) = \sum_{t=0}^{H}R(s_{t}, u_{t})
$$

What we want to do is calculate the utility which is equal to the expectation of the reward over all trajectories sampled from a policy.

$$
U(\theta) = \mathop{\mathbb{E}}_{\pi_{\theta}}[R(\tau))]
$$
or another way of looking at it is we want the reward of a trajectory times the probability of it occurring.

$$
U(\theta) = \sum_{\tau}P(\tau;\theta)R(\tau))
$$
So we take the gradient w.r.t. $\theta$:
$$
\nabla_{\theta} U(\theta) = \nabla_{\theta} \sum_{\tau}P(\tau;\theta)R(\tau))
$$

However the you cannot sample from a gradient  so this is not exactly tractable. But using a simple algebraic trick we can turn this into something that we can sample from. What follows is called the REINFORCE trick, the log likelihood trick, or the score trick. Despite it's many names it is fairly straightforward.

First because the only thing that depends on $\theta$ is the probability of  being in a trajectory we can move the gradient $\nabla$ into the summations.
$$
\nabla_{\theta} U(\theta) = \sum_{\tau}\nabla_{\theta} P(\tau;\theta)R(\tau))
$$
It would be ideal if this was multiplied by a probability P that does not contain the nabla. Because then we could change this into an expectation that can be sample from.
To do this we will multiply everything by $\frac{P(\tau;\theta)}{P(\tau;\theta)}$.
$$
\nabla_{\theta} U(\theta) = \sum_{\tau}\frac{P(\tau;\theta)}{P(\tau;\theta)}\nabla_{\theta} P(\tau;\theta)R(\tau))
$$
We can do this of course because we just mutiplied the whole thing by one.

Due to a simple application of the chain rule we know that the above is the same as

$$
\nabla_{\theta} U(\theta) = \sum_{\tau}P(\tau;\theta)\nabla_{\theta} \log P(\tau;\theta)R(\tau))
$$

Now we have something that looks quite a lot like an expectation. Which we can now change to look like this:
$$
\nabla_{\theta} U(\theta) = \mathop{\mathbb{E}}_{\tau\sim \pi_{\theta}}[\nabla_{\theta} \log P(\tau;\theta)R(\tau))]
$$
Now we have an expectation that we can sample from. Now with one more simple step we can change back to an equation in the form of policies instead of who trajectories:
$$
\nabla_{\theta} U(\theta) = \mathop{\mathbb{E}}_{\tau\sim \pi_{\theta}}\left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t} | s_{t})R(\tau)) \right]
$$
And thus we have derived the formula for the simplest policy gradient.

## Reward To Go
To complicate matters a little we are currently using rewards across the whole trajectory. This obviously does not makes sense because our goal is to determine the consequences of our actions. Previous rewards before are current action are irrelevant to our decision making. We only care about rewards from here on out. We call this idea the "rewards to go". We can adjust our formula accordingly to take this into account.
$$
\nabla_{\theta} U(\theta) = \mathop{\mathbb{E}}_{\tau\sim \pi_{\theta}}\left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t} | s_{t})\sum^{T}_{t'=t}R(s_{t'},a_{t'},s_{t'+1}) \right]
$$

## Baselines
One way we can reduce the variance and keep stability is by subtracting a learned function called the [[baseline]] $b_{t}(s_{t})$ from our reward. So:
$$
 = \mathop{\mathbb{E}}_{\tau\sim \pi_{\theta}}\left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t} | s_{t})(\sum^{T}_{t'=t}R(s_{t'},a_{t'},s_{t'+1} )- b(s_{t})) \right]
$$

The reason we can do this is because the expectation of the gradient of the log probabilities is equal to zero. A great article explaining this is found [here](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma)

A baseline that works great for our purposes is the on policy state value function $V^{\pi}(s_{t})$. Or the average value if a policy gets to state $s_t$ and acts according to the policy in the future.

*A2C*
So given the chosen baseline the formula for the A2C loss is as follows:
$$
= \mathop{\mathbb{E}}_{d,\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(a|s)(R(s,a) - V^{\pi}(s_{t}))]
$$
