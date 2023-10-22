import numpy as np
import tensorflow as tf
from keras import layers
import gym
import random

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.001
LR_CRITIC = 0.002

# Actor Model
def create_actor_model(state_dim, action_dim, action_bound):
    model = tf.keras.Sequential([
        layers.Input(state_dim),
        layers.Dense(400, activation='relu'),
        layers.Dense(300, activation='relu'),
        layers.Dense(action_dim, activation='tanh')
    ])
    model.layers[-1].set_weights([0.1 * np.random.randn(*w.shape) for w in model.layers[-1].get_weights()])
    return model

# Critic Model
def create_critic_model(state_dim, action_dim):
    state_input = layers.Input(state_dim)
    action_input = layers.Input(action_dim)
    concat = layers.Concatenate()([state_input, action_input])
    hidden1 = layers.Dense(400, activation='relu')(concat)
    hidden2 = layers.Dense(300, activation='relu')(hidden1)
    output = layers.Dense(1)(hidden2)
    return tf.keras.Model([state_input, action_input], output)

class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.actor = create_actor_model(state_dim, action_dim, action_bound)
        self.target_actor = create_actor_model(state_dim, action_dim, action_bound)
        self.critic = create_critic_model(state_dim, action_dim)
        self.target_critic = create_critic_model(state_dim, action_dim)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)
        
        self.buffer = []

    def policy(self, state):
        sampled_action = tf.squeeze(self.actor(state))
        return sampled_action.numpy()

    def add_to_buffer(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)

    def buffer_save_to_file(self):
        # save as csv
        buffer = self.convert(self.buffer)
        np.savetxt("buffer.csv", buffer, delimiter=",")

    def convert(self, buffer):
        buffer_np = []
        buffer_row = []
        for i in range(len(buffer)):
            buffer_row = buffer_row + buffer[i][0][0].tolist()
            buffer_row = buffer_row + buffer[i][1].tolist()
            buffer_row = buffer_row + [buffer[i][2]]
            buffer_np.append(buffer_row)
            buffer_row = []
        return np.array(buffer_np)

    def train(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        batch = np.array(batch, dtype=object)
        states, actions, rewards, next_states, dones = np.split(batch, 5, axis=1)

        states = np.array(states.tolist()).squeeze()
        actions = np.array(actions.tolist()).squeeze()
        rewards = np.array(rewards.tolist()).squeeze()
        next_states = np.array(next_states.tolist()).squeeze()
        dones = np.array(dones.tolist()).squeeze()

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(next_states)
            target_critic_value = self.target_critic([next_states, target_actions])
            critic_value = self.critic([states, actions])

            target = rewards + GAMMA * target_critic_value * (1. - dones)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

            new_policy_actions = self.actor(states)
            actor_loss = -self.critic([states, new_policy_actions])
            actor_loss = tf.reduce_mean(actor_loss)

        critic_grad = tape1.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        actor_grad = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor.variables, TAU)
        self.update_target(self.target_critic.variables, self.critic.variables, TAU)

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

env = gym.make("Pendulum-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = DDPG(state_dim, action_dim, action_bound)
episodes = 200

for ep in range(episodes):
    state = env.reset()
    ep_reward = 0

    while True:
        state = np.array([state[0]])
        action = agent.policy(state.reshape(1, -1))
        action = action * action_bound
        action = np.array([action])
        next_state, reward, done, _, _ = env.step(action)
        agent.add_to_buffer((state, action, reward, next_state, done))
        if len(agent.buffer) > BATCH_SIZE:
            agent.buffer_save_to_file()
            agent.train()
        ep_reward += reward
        state = np.array([next_state])
        env.render()

        if done:
            break

    print(f"Episode: {ep + 1}, Reward: {ep_reward}")

env.close()

# Testing the agent
episodes_test = 10
for ep in range(episodes_test):
    state = env.reset()
    ep_reward = 0

    while True:
        env.render()  # This will visualize the environment
        action = agent.policy(state.reshape(1, -1))
        action = action * action_bound
        next_state, reward, done, _ = env.step(action)

        ep_reward += reward
        state = next_state

        if done:
            break

    print(f"Test Episode: {ep + 1}, Reward: {ep_reward}")

env.close()
