###############################################################################
# For more info, see https://hoseinkh.github.io/
###############################################################################
import gym
import os
import sys
import numpy as np
"""
# if using tensorflow v1:
import tensorflow as tf
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
###############################################################################
# It is better to define everything directly. This allows tensorflow to ...
# ... automatically calculate the cost functions, and hence we get rid of ...
# ... the issue of manually feeding it to the tensorflow.
# To do this TensorFlow needs to remember what operations happen in what ...
# ... order during the forward pass. Then, during the backward pass, ...
# ... TensorFlow traverses this list of operations in reverse order to ...
# ... compute gradients.
class HiddenLayer:
  def __init__(self, inp_size_of_hidden_layer, out_size_of_hidden_layer, f=tf.nn.tanh, use_bias=True):
    self.W = tf.Variable(tf.random.normal(shape=(inp_size_of_hidden_layer, out_size_of_hidden_layer)))
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(out_size_of_hidden_layer).astype(np.float32))
    self.f = f
  ######################################
  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)
###############################################################################
# approximates pi(a | s)
class PolicyModel:
  def __init__(self, data_input_size, num_of_actions, hidden_layer_sizes):
    # create the graph
    # K = number of actions
    self.layers = []
    NN_input_size = data_input_size
    for NN_output_size in hidden_layer_sizes:
      layer = HiddenLayer(NN_input_size, NN_output_size)
      self.layers.append(layer)
      NN_input_size = NN_output_size
    #
    # final layer: we want to predict the probability for each action.
    # ... hence we use the softmax activitation function.
    layer = HiddenLayer(NN_input_size, num_of_actions, tf.nn.softmax, use_bias=False)
    self.layers.append(layer)
    #
    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, data_input_size), name='X')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
    #
    # calculate output and cost
    out_of_curr_layer = self.X
    for layer in self.layers:
      out_of_curr_layer = layer.forward(out_of_curr_layer)
    p_a_given_s = out_of_curr_layer
    #
    self.predict_op = p_a_given_s
    #
    # self.one_hot_actions = tf.one_hot(self.actions, K)
    #
    selected_probs = tf.log(
      tf.reduce_sum(
        p_a_given_s * tf.one_hot(self.actions, num_of_actions),
        reduction_indices=[1]
      )
    )
    #
    #
    cost = -tf.reduce_sum(self.advantages * selected_probs)
    #
    # self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
    self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
    # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
  ######################################
  def set_session(self, session):
    self.session = session
  ######################################
  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )
  ######################################
  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})
  ######################################
  def sample_action(self, X):
    p = self.predict(X)[0]
    return np.random.choice(len(p), p=p)
###############################################################################
# approximates V(s)
class ValueModel:
  def __init__(self, data_input_size, hidden_layer_sizes):
    # create the graph
    self.layers = []
    NN_input_size = data_input_size
    for NN_output_size in hidden_layer_sizes:
      layer = HiddenLayer(NN_input_size, NN_output_size)
      self.layers.append(layer)
      NN_input_size = NN_output_size
    #
    # final layer: note that we are predicting the value function for a state,
    # ... hence the output layer has the size of 1
    layer = HiddenLayer(NN_input_size, 1, lambda x: x)
    self.layers.append(layer)
    #
    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, data_input_size), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
    #
    # calculate output and cost
    out_of_curr_layer = self.X
    for layer in self.layers:
      out_of_curr_layer = layer.forward(out_of_curr_layer)
    Y_hat = tf.reshape(out_of_curr_layer, [-1]) # the output
    self.predict_op = Y_hat
    #
    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)
    self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
  ######################################
  def set_session(self, session):
    self.session = session
  ######################################
  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
  ######################################
  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})
###############################################################################
def play_one_td(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  #
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    #
    # if done:
    #   reward = -200
    #
    # update the models
    V_next = vmodel.predict(observation)[0]
    G = reward + gamma*V_next
    advantage = G - vmodel.predict(prev_observation)
    pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)
    #
    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1
  #
  return totalreward
###############################################################################
def play_one_mc(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  #
  states = []
  actions = []
  rewards = []
  #
  reward = 0
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    #
    states.append(observation)
    actions.append(action)
    rewards.append(reward)
    #
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    #
    if done:
      reward = -200
    #
    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1
  #
  # save the final (s,a,r) tuple
  action = pmodel.sample_action(observation)
  states.append(observation)
  actions.append(action)
  rewards.append(reward)
  #
  returns = []
  advantages = []
  G = 0
  for s, r in zip(reversed(states), reversed(rewards)):
    returns.append(G)
    advantages.append(G - vmodel.predict(s)[0])
    G = r + gamma*G
  returns.reverse()
  advantages.reverse()
  #
  # update the models
  pmodel.partial_fit(states, actions, advantages)
  vmodel.partial_fit(states, returns)
  #
  return totalreward
###############################################################################
# we are evaluating the performance of the model at each time t by ...
# ... taking the running average of the adjacent 100 iterations to that time t.
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.xlabel("Iterations")
  plt.ylabel("Average Time")
  # plt.show()
  curr_path = os.path.abspath(os.getcwd())
  plt.savefig(curr_path + '/figs/reward_running_avg_CartPole_tabular_state_action_values.png')
  plt.close()
###############################################################################
if __name__ == "__main__":
  curr_path = os.path.abspath(os.getcwd())
  env = gym.make('CartPole-v1')
  num_actions = env.observation_space.shape[0]
  K = env.action_space.n
  hidden_layer_sizes_for_PolicyModel = []
  hidden_layer_sizes_for_ValueModel = [10]
  pmodel = PolicyModel(num_actions, K, hidden_layer_sizes_for_PolicyModel)
  vmodel = ValueModel(num_actions, hidden_layer_sizes_for_ValueModel)
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  pmodel.set_session(session)
  vmodel.set_session(session)
  gamma = 0.99
  #
  if True:
    monitor_dir = os.getcwd() + "/videos/" + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
  #
  N = 2000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    totalreward = play_one_mc(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
  #
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())
  #
  plt.plot(totalrewards)
  plt.xlabel("Iterations")
  plt.ylabel("Running Average Time")
  plt.savefig(curr_path + '/figs/reward_avg_CartPole_tabular_state_action_values.png')
  #
  plot_running_avg(totalrewards)


