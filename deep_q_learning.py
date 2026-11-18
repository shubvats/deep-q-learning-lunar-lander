
import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE


# ==========================================
# Exercise 1: Create Q-Network and Target Q-Network
# ==========================================

def create_q_network(state_size, num_actions):
    """
    Creates a Q-Network for Deep Q-Learning.

    Args:
        state_size: Shape of the environment state.
        num_actions: Number of possible actions.

    Returns:
        q_network: Neural network estimating Q-values.
    """
    q_network = Sequential([
        Input(shape=state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    return q_network


def create_target_q_network(state_size, num_actions):
    """
    Creates the target Q-Network.
    Uses the same architecture as the Q-Network.
    """
    target_q_network = Sequential([
        Input(shape=state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    return target_q_network


# Adam optimizer
def create_optimizer(ALPHA):
    return Adam(learning_rate=ALPHA)


# ==========================================
# Exercise 2: Compute Loss
# ==========================================

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the Mean Squared Error between
    target Q-values and predicted Q-values.

    Args:
        experiences: Tuple containing states, actions,
                     rewards, next_states, and done_vals.
        gamma: Discount factor.
        q_network: Main Q-Network.
        target_q_network: Target Q-Network.

    Returns:
        loss: Mean Squared Error.
    """

    # Unpack experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute maximum future Q-value
    max_qsa = tf.reduce_max(
        target_q_network(next_states),
        axis=-1
    )

    # Calculate target values
    y_targets = rewards + (
        gamma * max_qsa * (1 - done_vals)
    )

    # Get predicted Q-values
    q_values = q_network(states)

    # Select Q-values corresponding to chosen actions
    q_values = tf.gather_nd(
        q_values,
        tf.stack([
            tf.range(q_values.shape[0]),
            tf.cast(actions, tf.int32)
        ], axis=1)
    )

    # Calculate Mean Squared Error
    loss = MSE(y_targets, q_values)

    return loss
