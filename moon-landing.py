import tensorflow as tf
from keras import layers, models
import numpy as np
import gym

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class StateProximator:
    def __init__(self, size):
        self.size = size
        self.state_buffer = np.zeros((size, 8))

    def add(self, observation):
        self.state_buffer[:-1] = self.state_buffer[1:]
        self.state_buffer[-1] = observation

    def get(self):
        return np.array([self.state_buffer])
    
    def reset(self):
        self.state_buffer = np.zeros((self.size, 8))

class ColumnConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ColumnConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # input_shape: (batch_size, rows, columns)
        _, columns, _ = input_shape
        self.conv_layers = [layers.Conv1D(self.filters, self.kernel_size) for _ in range(columns)]
        self.concat_layer = layers.Concatenate(axis=-1)

    def call(self, inputs):
        transposed_inputs = inputs
        conv_results = []
        for conv_layer, column in zip(self.conv_layers, tf.unstack(transposed_inputs, axis=1)):
            conv_results.append(conv_layer(tf.expand_dims(column, -1)))
        output = self.concat_layer(conv_results) 
        return output

class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv1 = ColumnConv1D(filters, kernel_size)
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        out = self.relu(x)
        out = tf.transpose(out, perm=[0, 2, 1])
        return out

class Actor(tf.keras.Model):
    def __init__(self, action_dim, filters, kernel_size, num_blocks):
        super(Actor, self).__init__()
        self.blocks = [ConvBlock(filters, kernel_size) for _ in range(num_blocks)]
        self.fc = layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        batch_size = tf.shape(state)[0]
        state = tf.transpose(state, perm=[0, 2, 1])  # Transpose to (batch_size, columns, rows)
        x = state
        for block in self.blocks:
            x = block(x)
        x = tf.transpose(x, perm=[0, 2, 1])  # Transpose back to (batch_size, rows, action_dim)
        x = tf.reshape(x, (batch_size, -1))  # Flatten
        action = self.fc(x)
        return action

class Critic(tf.keras.Model):
    def __init__(self, filters, kernel_size, num_blocks):
        super(Critic, self).__init__()
        self.blocks = [ConvBlock(filters, kernel_size) for _ in range(num_blocks)]
        self.fc1 = layers.Dense(2, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, state, action):
        batch_size = tf.shape(state)[0]
        state = tf.transpose(state, perm=[0, 2, 1])
        x = state
        for block in self.blocks:
            x = block(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, (batch_size, -1))
        x = self.fc1(x)
        x = tf.concat([x, action], axis=1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_high, action_low):
        self.actor = Actor(action_dim, filters=4, kernel_size=10, num_blocks=3)
        self.critic = Critic(filters=4, kernel_size=10, num_blocks=3)
        self.target_actor = Actor(action_dim, filters=4, kernel_size=10, num_blocks=3)
        self.target_critic = Critic(filters=4, kernel_size=10, num_blocks=3)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.action_high = action_high
        self.action_low = action_low

    def get_action(self, state, exploration_noise):
        action = self.actor(state)
        action += exploration_noise * np.random.normal(size=self.action_high.shape)
        action = np.clip(action, self.action_low, self.action_high)
        return action[0]

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99, tau=0.001):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Actor Loss
            target_actions = self.target_actor(next_states)
            critic_value = self.critic(next_states, target_actions)
            target_q_values = rewards + gamma * critic_value * (1 - dones)
            predicted_actions = self.actor(states)
            actor_loss = -tf.math.reduce_mean(self.critic(states, predicted_actions) * actions)

            # Critic Loss
            critic_loss = tf.losses.mean_squared_error(target_q_values, self.critic(states, actions))

        actor_gradients = tape1.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape2.gradient(critic_loss, self.critic.trainable_variables)

        # Manually apply gradients
        for grad, var in zip(actor_gradients, self.actor.trainable_variables):
            var.assign_sub(self.actor_optimizer.learning_rate * grad)

        for grad, var in zip(critic_gradients, self.critic.trainable_variables):
            var.assign_sub(self.critic_optimizer.learning_rate * grad)

        # Update target networks
        self.update_target_networks(tau)

    def update_target_networks(self, tau):
        for (source, target) in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target.assign(tau * source + (1 - tau) * target)

        for (source, target) in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target.assign(tau * source + (1 - tau) * target)

def env_reset(env):
    obs = env.reset()[0]
    obs = np.array([obs])
    return obs

def env_step(env, action):
    next_obs, reward, done, _, _ = env.step(action)
    next_obs = np.array([next_obs])
    reward = np.array([[reward]])
    done = np.array([[done]])
    return next_obs, reward, done

# Hyperparameters
env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

print(f"State Dimension: {state_dim}")
print(f"Action Dimension: {action_dim}")
print(f"Action High: {action_high}")
print(f"Action Low: {action_low}")

agent = DDPGAgent(state_dim, action_dim, action_high, action_low)
state_proximator = StateProximator(30)

max_episodes = 500
max_steps = 1000

# print out cuda devices
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

for episode in range(max_episodes):
    obs = env_reset(env)
    state_proximator.add(obs)
    state = state_proximator.get()
    
    episode_reward = 0

    for step in range(max_steps):
        print(f"Step: {step + 1}", end="\r")
        action = agent.get_action(state, exploration_noise=0.1)
        next_obs, reward, done = env_step(env, action)
        action = np.array([action])
        state_proximator.add(next_obs)
        next_state = state_proximator.get()
        agent.train(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            state_proximator.reset()
            break

    print(f"Episode: {episode + 1}, Reward: {episode_reward}")

    # Save the model every 5 episodes
    if (episode + 1) % 5 == 0:
        agent.actor.save_weights(f"actor_weights_{episode + 1}.h5")
        agent.critic.save_weights(f"critic_weights_{episode + 1}.h5")

env.close()
