
notes from Taiwan University ML course, see: https://www.youtube.com/watch?v=z95ZYgPgXOY

### Policy gradient

s: state, a: action

Trajectory: $\tau = {s_1, a_1, s_2, a_2 ... s_t, a_t}$

Probability of this trajectory: $P_{\theta}(\tau) = p(s_1)p(a_1|s_1)p(s_2|a_1,s_1)$ ...

In which: 

- $\theta$ is the parameters of our policy

- $p(a_i|s_i)$ is what we can control. We will adjust the policy to maxmize **Reward** 

Reward: $R(\tau) = \sum r_t$  is a random variable, only its expectation could be calculated
$$
E(R_\theta) = \sum_{\tau}R(\tau)P_{\theta}(\tau)

\Delta E(R_\theta) = \sum_{\tau}R(\tau)\Delta P_{\theta}(\tau)

= \sum_{\tau}R(\tau) P_{\theta}(\tau) \Delta log P_{\theta}(\tau)

sampling $\approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n) \Delta log P_{\theta}(\tau^n)$

= $\frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{tn} R(\tau^n) \Delta log P_{\theta}(a_t^n|s_t^n)$, because other parts are not derivable
$$

#### Tip 1, add a baseline
$E(R_\theta) \approx \frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{tn} (R(\tau^n)-b) \Delta log P_{\theta}(a_t^n|s_t^n)$

if all rewards are positive, without baseline, the probability of not sampled actions will drop

#### Tip 2, assign suitable credits
replace $R(\tau^n)$ in $(R(\tau^n)-b)$ by a time-relavant reward, time discount
