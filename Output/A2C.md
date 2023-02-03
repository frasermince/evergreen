Building from the foundation of Q learning we learned in the last article we now explore an on policy algorithm using policy gradients and actor critic methods. DQN relied upon a experience replay memory in order to get it's results. This makes it an [[Off Policy]] algorithm. 

An [[Off Policy]] algorithm is one that can rely upon previous memories to update the parameters. The benefit of this is the memory replay often used in these algorithms reduces the effects of the non-stationarity nature of RL i.e. the fact the target also relies upon the function approximator itself. It also will de-coorelates updates so that we are training on data that is randomly sampled instead of all adjacent. Off Policy methods also have the benefit of being far more sample efficient due to being able to reuse old data.

In contrast an [[On Policy]] algorithm only relies upon data from the most recent policy. The benefit of an On Policy method is that you are directly optimizing the thing you actually care about. A value function like in DQN can act indirectly to optimize towards a good policy but there are more failure modes and less stability due to this indirectness.

So it would be advantageous to directly optimize for the thing we care about, namely the policy. But how do we do this without losing stability?

*Basic Policy Gradient*
What we want to do is take the gradient of the reward so we can optimize for maximum reward:
$$
\nabla\mathop{\mathbb{E}}_{\pi_{\theta}}[R(S,A)]
$$

However the you cannot sample from a gradient of an expectation so this is not exactly tractable. But using a simple algebraic trick we can turn this into something that we can sample from. What follows is called the REINFORCE trick or the log likelihood trick:
$$
\nabla\mathop{\mathbb{E}}_{\pi_{\theta}}[R(S,A)] = \nabla_{\theta}\sum_{s}d(s)\sum_{a}\pi_{\theta}(a|s)r_{sa}
$$
Where $r_{sa}$ is the expected reward given you are in a state and take an action and $d(s)$ is the probability of being in that state. 

Because the only thing that depends on $\theta$ is the policy $\pi$ we can move the gradient $\nabla$ into the summations.
$$
= \sum_{s}d(s)\sum_{a}r_{sa}\nabla_{\theta}\pi_{\theta}(a|s)
$$

Next we will multiply everything by $\frac{\pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}$.
$$
= \sum_{s}d(s)\sum_{a}r_{sa}\pi_{\theta}(a|s)\frac{\nabla_{\theta}\pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}
$$
We can do this of course because we basically just mutiplied the whole thing by one.

Due to a simple application of the chain rule we know that the above is the same as
$$
= \sum_{s}d(s)\sum_{a}\pi_{\theta}(a|s)r_{sa}\nabla_{\theta}\log \pi_{\theta}(a|s)
$$

Now we have a value that looks very close to what we started with and this equation is equivalent to:
$$
= \mathop{\mathbb{E}}_{d,\pi_{\theta}}R(s,a)\nabla_{\theta}\log \pi_{\theta}(a|s)
$$
Now we have an expectation that we can sample.

One way we can reduce the variance and keep stability is by subtracting a baselines from our loss.

*A2C*
The formula for the A2C loss is as follows:
$$
$$