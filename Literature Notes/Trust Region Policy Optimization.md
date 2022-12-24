**URL:** https://spinningup.openai.com/en/latest/algorithms/trpo.html

TRPO uses KL Divergence to stop performance collapse from one step to the other. Within a [[Vanilla Policy Gradient]] even small differences in parameter space can lead to vastly different behavior. The theoretical TRPO update is defined as:

$$
\displaylines{

\theta_{k+1} = \arg \max_{\theta} L(\theta_{k}, \theta)
\newline s.t. \bar{D}_{KL} (\theta\|\theta_{k}) \leq \delta
}$$
Where $L(\theta_{k}, \theta)$ is surrogate advantage. Meaning how the new policy performs compared to the old. $s.t.$ refers to "such that" and $\bar{D}_{KL} (\theta\|\theta_{k})$ refers to the average [[KL Divergence]] between the new parameters and the old. 

L is defined as follows:
$$
L(\theta_{k}, \theta)= \mathop{\mathbb{E}}_{s,a \sim \pi_{\theta_{k}}}
\left[ \frac{\pi_{\theta} (a|s) }{\pi_{\theta_{k}} (a|s)} A^{\pi_{\theta_{k}}}(s,a)\right]
$$
Meaning we divide the new policy by the old. Here $\theta_{k}$ refers to the old policy. We multiply by the [[Advantage Function ]]and take the expectation. This is referred to as the objective.

$\bar{D}_{KL}$ is defined as follows
$$
\bar{D}_{KL} = \mathop{\mathbb{E}}_{s \sim \pi_{\theta_{k}}}[D_{KL}(\pi_{\theta}(\cdot | s) \| \pi _{\theta_{k}}(\cdot | s))] 
$$
So this is the [[Expectation]] of the [[KL Divergence]]. This is referred to as the constraint

This theoretical update is not the easiest to work with so we do a [[Taylor Expansion]] on the objective and constraint. 

#to-expand 
* Explain why the theoretical version isn't the easiest to work with
* Fill in taylor expanded version here
* Fill in analytical solution
* Understand [[Backtracking Line Search]]
* Understand [[Lagrangian Duality]]
* Review [[Taylor Expansion]]
* Understand the desired proofs in the "You Should Know" section
* Understand the [[Conjugate Gradient Algorithm]]

This can also be analytically solved by [[Lagrangian Duality]]

#reinforcement-learning
