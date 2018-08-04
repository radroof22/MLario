from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import os
import numpy as np
from model_mario_Lite import Agent, EnvironmentControl

env = gym_super_mario_bros.make("SuperMarioBros-v1")
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

#tf.reset_default_graph()

num_actions = 7
state_size = (60, 64)


max_steps_per_episode = 100000

path = os.getcwd() + "/mario-model/"


agent = Agent(num_actions, state_size)
EC = EnvironmentControl(state_size, None, None, None)

saver = tf.train.Saver()

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    for episode in range(5):

        state = env.reset()

        episode_history = []
        episode_rewards = 0

        for step in range(max_steps_per_episode):

            env.render()
            # Get weights for each action
            action_probabilities = sess.run(agent.outputs,
                                            feed_dict=
                                            {agent.input_layer: EC.resize_state(state)})
            # print(action_probabilities.shape)
            action_choice = np.random.choice(range(num_actions),
                                             p=action_probabilities[0])

            state, reward, done, _ = env.step(action_choice)
            state_next = EC.resize_state(state)


            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                print("Episode {}: {}".format(episode, episode_rewards))
                break



