import tensorflow as tf
import numpy as np
from skimage import color, transform

# TODO Build the policy gradient neural network
class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size[0], state_size[1], 1])
        # self.input_layer = tf.cast(self.input_layer, dtype=tf.float32)
        # self.input_layer = tf.reshape(self.input_layer, [None, 16, 8])
        '''self.input_layer = np.array(self.input_layer)
        print((self.input_layer))
        self.input_layer = self.input_layer.reshape(-1, 16, 8)'''
        # Neural net starts here
        '''conv_1 = tf.layers.conv2d(self.input_layer,
                          filters=8,
                          kernel_size=[4, 4],
                          padding="same",
                          activation=tf.nn.relu, name="Conv1")

        pool_1 = tf.layers.max_pooling2d(conv_1,
                                 pool_size=[2, 2],
                                 strides=2, name="Pool1")'''

        conv_2 = tf.layers.conv2d(self.input_layer,
                                  filters=2,
                                  kernel_size=[2, 2],
                                  padding="same",
                                  activation=tf.nn.relu, name="Conv2")

        pool_2 = tf.layers.max_pooling2d(conv_2,
                                         pool_size=[2, 2],
                                         strides=1, name="Pool2")
        dropout0 = tf.layers.dropout(pool_2, rate=.5, training=True, name="Dropout_0")
        flattenened_pooling = tf.layers.flatten(dropout0, name="Flatten")
        hidden_layer = tf.layers.dense(flattenened_pooling,
                                       4,
                                       activation=tf.nn.relu,
                                       kernel_initializer=initializer, name="A1")
        #dropout1 = tf.layers.dropout(hidden_layer, rate=.5, training=True, name="Dropout_1")
        hidden_layer_2 = tf.layers.dense(hidden_layer, 4, activation=tf.nn.relu, kernel_initializer=initializer, name="A2")
        ##dropout2 = tf.layers.dropout(hidden_layer_2, rate=.5, training=True, name="Dropout_2")
        # Output of neural net
        out = tf.layers.dense(hidden_layer_2, num_actions, kernel_initializer=initializer, activation=None, name="Output")

        self.outputs = tf.nn.softmax(out, name="Softmax_Output")
        self.choice = tf.argmax(self.outputs, axis=1, name="Choice")

        # Training Procedure
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name="Rewards")
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name="Actions")

        one_hot_actions = tf.one_hot(self.actions, num_actions, name="One_Hot")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions, name="SoftCross")

        self.loss = tf.reduce_mean(cross_entropy * self.rewards, name="Loss")

        self.gradients = tf.gradients(self.loss, tf.trainable_variables(), name="Gradients")

        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name="Optimizer")
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))


class EnvironmentControl:
    discount_rate = 0.95

    def __init__(self, state_size, max_steps_in_episode, num_of_episode, episode_batch_size):
        self.state_size = state_size
        self.max_steps_in_episode = max_steps_in_episode
        self.num_of_episodes = num_of_episode
        self.episode_batch_size = episode_batch_size

    def resize_state(self, state):
        if state.shape[-1] == 1:
            return state
        return transform.rescale(color.rgb2gray(state), 1/4).reshape(-1, self.state_size[0], self.state_size[1], 1)


    def discount_normalize_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        total_rewards = 0

        for i in reversed(range(len(rewards))):
            total_rewards = total_rewards * self.discount_rate + rewards[i]
            discounted_rewards[i] = total_rewards

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards
