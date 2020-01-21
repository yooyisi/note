
notes from Taiwan University ML course, see: https://www.youtube.com/watch?v=z95ZYgPgXOY

## Policy gradient

s: state, a: action

Trajectory: $\tau = {s_1, a_1, s_2, a_2 ... s_t, a_t}$

Probability of this trajectory: $P_{\theta}(\tau) = p(s_1)p(a_1|s_1)p(s_2|a_1,s_1)$ ...

In which: 

- $\theta$ is the parameters of our policy

- $p(a_i|s_i)$ is what we can control. We will adjust the policy to maxmize **Reward** 

Reward: $R(\tau) = \sum r_t$  is a random variable, only its expectation could be calculated
$$
E(R_\theta) = \sum_{\tau}R(\tau)P_{\theta}(\tau)
$$
$$
\Delta E(R_\theta) = \sum_{\tau}R(\tau)\Delta P_{\theta}(\tau)
$$
$$
= \sum_{\tau}R(\tau) P_{\theta}(\tau) \Delta log P_{\theta}(\tau)
$$
sampling
$$
\approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n) \Delta log P_{\theta}(\tau^n)
$$
because other parts are not derivable
$$
= \frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{tn} R(\tau^n) \Delta log P_{\theta}(a_t^n|s_t^n)
$$


### Tip 1, add a baseline
$E(R_\theta) \approx \frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{tn} (R(\tau^n)-b) \Delta log P_{\theta}(a_t^n|s_t^n)$

if all rewards are positive, without baseline, the probability of not sampled actions will drop

### Tip 2, assign suitable credits
replace $R(\tau^n)$ in $(R(\tau^n)-b)$ by a time-relavant reward, time discount

## PPO - Proximal Policy Optimization
### On vs Off Policy
- on policy: learning and apply are the same (learn cheese by playing)
- off policy: learning and apply are different (learn cheese by watching)

**Problem of on-policy:**

Use policy \pi_\theta to collect data, update parameters. After updating, the distribution p_{\theta}(\tau) changed, you have to do sampling again
- time consuming

Off-policy, use another \pi_\theta' to interact with env, sampling a lot of data, \theta use those data to learn and update, until converge or bottle-neck, then generate new data with \theta' again.

$$
E_{(s_t,a_t)\sim \pi_\theta}[A^{\theta}(s_t,a_t)\Delta logP_{\theta}(a_t^n|s_t^n))]
$$
$$
= E_{(s_t,a_t)\sim \pi_\theta}[\frac{P_{\theta}(a_t^n,s_t^n)}{P_{\theta'}(a_t^n,s_t^n)} A^{\theta'}(s_t,a_t)\Delta logP_{\theta}(a_t^n|s_t^n))]
$$
$$
J^{\theta'}(\theta) = E_{(s_t,a_t)\sim \pi_\theta}[\frac{P_{\theta}(a_t^n|s_t^n)}{P_{\theta'}(a_t^n|s_t^n)}A^{\theta'}(s_t,a_t)]
$$

### importance sampling
> why off-policy is feasible

We need sampling from distribution p to calculate its Expectation. It can be archived via sampling from another distribution q, as long as p and q are 0 at the same time.

But var is not the same, to calculate the expectation, due to the potentially huge variance difference, a big amount of sampling data is needed.


### PPO/TRPO
using $\theta'$ to update $\theta$, and $P_{\theta'}$ and $P_{\theta}$ should not be too different

$$
J_{PPO}^{\theta'} = J_{PPO}^{\theta'} - \beta KL(\theta,\theta')
$$

TRPO use constraint, hard to apply, performance similiar
