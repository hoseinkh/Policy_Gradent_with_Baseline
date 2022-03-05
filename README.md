# Policy_Gradent_for_CartPole

 We use tabular state-action value functions (Q-values) to find the optimal policy for controlling a Cart Pole.

<br />

## Task:

The goal is to directly model the policy function and try to optimize it. This approach is called **policy gradient**.



## Solution:

Here we model the policy function using a neural network, and try to optimize it. The policy gradient approach typically has hagh variance, and hence we use a technique called **baseline** to decrease the variance of the model. The baseline requires the calculating of the state value functions V(s). We use another neural network to calculate the value functions. The challenge of this project is to calculate the gradient when trying to optimize the policy function. Here we show a trick to do that.

---

**Polciy gradient** approaches aim at directly modeling the policy and trying to optimize it. There are few challenges for this approach:

1. What would be the objective function to train the policy? The answer is shown in the following figure:

   <p float="left">
     <img src="/figs/policy_gradient_objective_func.png" width="450" />
   </p>

   There are more details on how we arrive at this objective function for training the policy function approximator. I recommend reading the [1], [2] and [3] for more information on this.

2. How to manually calculate the gradient? This is an implementation challenge, rather than a theoretical one. If, for instance, we use neural networks as the function approximator, then calculating the gradient of the objective funciton w.r.t. the weights could be troublesome. There is an interesting trick that we can use to do this. We can use the ***automatic differentiation*** provided in Tensorflow. The trick is to use only operations that keras "knows about" aka that exist as operations in TensorFlow, and the TensorFlow will automatically a graph of operations to backpropagate against. Now, in order to do this, we manually implement a hidden layer in the code, and the fully connected neural network that we are using for the policy gradient is created by adding one layer after another. This allows TensorFlow to do automatic differentiation.

3. This approach results in high variance. There is a technique called "**baseline**" for which we use the state value functions **V(s)** to decrease the variance of the model. This requires a value function approximator (here we use another neural network) to estimate the state-value functions.



### Results

We apply this method to the Cart Pole environment. The results is shown in the following figure:

<p float="left">
  <img src="/figs/CartPole_policy_avg_reward.png" width="450" />
</p>


We also record (a video for) the performance of the algorithm for the optimal policy.



------

### References

1. [Algorithms for Reinforcement Learning; Csaba Szepesvári](https://sites.ualberta.ca/~szepesva/rlbook.html).
2. [Reinforcement Learning: An Introduction; Richard S. Sutton,  Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
3. [Reinforcement Learning: State-of-the-Art; Marco Wiering, Martijn van Otterlo (Eds.)](https://link.springer.com/book/10.1007/978-3-642-27645-3)

