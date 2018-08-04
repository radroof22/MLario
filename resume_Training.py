#!pip install gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from model_mario import EnvironmentControl, Agent
import tensorflow as tf
import os
import numpy as np
from model_mario_Lite import Agent, EnvironmentControl
# from google.colab import files

env = gym_super_mario_bros.make("SuperMarioBros-v1")
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
# TODO Create the training loop
tf.reset_default_graph()

# Modify these to match shape of actions and states in your environment
## print(env.action_space)
## print(env.observation_space)
num_actions = 7
state_size = (60, 64)


training_episodes = 1500
max_steps_per_episode = 10000
episode_batch_size = 7

path = os.getcwd() + "/mario-model/reserve"

agent = Agent(num_actions, state_size)
EC = EnvironmentControl(state_size, max_steps_per_episode, training_episodes, episode_batch_size)

saver = tf.train.Saver()

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    total_episode_rewards = []

    # Create a buffer of 0'd gradients
    gradient_buffer = sess.run(tf.trainable_variables())
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for episode in range(training_episodes):

        state = env.reset()

        episode_history = []
        episode_rewards = 0

        for step in range(max_steps_per_episode):

            ##if episode % 100 == 0:
            ##   env.render()

            # Get weights for each action
            action_probabilities = sess.run(agent.outputs,
                                            feed_dict=
                                            {agent.input_layer: EC.resize_state(state)})
            # print(action_probabilities.shape)
            action_choice = np.random.choice(range(num_actions),
                                             p=action_probabilities[0])

            state_next, reward, done, _ = env.step(action_choice)
            state_next = EC.resize_state(state_next)
            # print(state.shape)
            episode_history.append([EC.resize_state(state), action_choice, reward, state_next])
            state = state_next

            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                # print(episode_history.shape)
                episode_history = np.array(episode_history)
                episode_history[:, 2] = EC.discount_normalize_rewards(episode_history[:, 2])

                ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                    agent.actions: episode_history[:, 1],
                                                                    agent.rewards: episode_history[:, 2]})
                # add the gradients to the grad buffer:
                for index, gradient in enumerate(ep_gradients):
                    gradient_buffer[index] += gradient

                break

        if episode % episode_batch_size == 0:

            feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))

            sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)

            for index, gradient in enumerate(gradient_buffer):
                gradient_buffer[index] = gradient * 0

        if episode % 100 == 0:
            # saver.save(sess, path + "Mario-checkpoint", episode)
            save_path = saver.save(sess, path, global_step=episode)
            #print(os.path.exists("mario.ckpt"))
            #files.download(path + "mario.ckpt"+".index")
#             onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#             print(onlyfiles)
#             for file in onlyfiles:
#               files.download(str(path) + str(file))
            print("Episode {}:\t {} ".format(episode, str(np.mean(total_episode_rewards[-100:]))))