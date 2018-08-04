from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from model_mario import EnvironmentControl, Agent
import tensorflow as tf
import os
import numpy as np

state_size = (40, 64)
num_actions = 7
EC = EnvironmentControl(state_size, max_steps_in_episode=5000, num_of_episode=1000, episode_batch_size=5)
Mario = Agent(num_actions, state_size)


# Environment Setup
env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# Tensorflow
tf.reset_default_graph()
init = tf.global_variables_initializer()
#'''
saver = tf.train.Saver(max_to_keep=2)
path = "/model/"
if not os.path.exists(path):
    os.makedirs(path)
#'''


# Env Variables
total_reward = 0
done = True

## Training!!
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    total_episode_rewards = []
    print("*"*25)
    from pprint import pprint
    pprint(sess.graph)
    print("*" * 25)
    # Create a buffer of 0'd gradients
    gradient_buffer = sess.run(tf.trainable_variables()) \
                      + sess.run(tf.trainable_variables([Mario.conv_1,
                                                         Mario.conv_2,
                                                         Mario.hidden_layer_2]))
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for episode in range(EC.num_of_episodes):
        episode_rewards = 0

        for step in range(EC.max_steps_per_episode):

            if episode % 100 == 0:
                env.render()

            # Get weights for each action
            action_probabilities = sess.run(Mario.outputs,
                                            feed_dict=
                                            {Mario.input_layer: EC.resize_state(state)})
            # print(action_probabilities.shape)
            action_choice = np.random.choice(range(num_actions),
                                             p=action_probabilities[0])

            state_next, reward, done, _ = env.step(action_choice)
            state_next = EC.resize_state(state_next)
            # print(state.shape)
            episode_history.append([EC.resize_state(state), action_choice, reward, state_next])
            state = state_next

            episode_rewards += reward

            if done or step + 1 == EC.max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                # print(episode_history.shape)
                episode_history = np.array(episode_history)
                episode_history[:, 2] = EC.discount_normalize_rewards(episode_history[:, 2])

                ep_gradients = sess.run(Mario.gradients,
                                        feed_dict={Mario.input_layer: np.vstack(episode_history[:, 0]),
                                                   Mario.actions: episode_history[:, 1],
                                                   Mario.rewards: episode_history[:, 2]})
                # add the gradients to the grad buffer:
                for index, gradient in enumerate(ep_gradients):
                    gradient_buffer[index] += gradient

                break

        if episode % EC.episode_batch_size == 0:

            feed_dict_gradients = dict(zip(Mario.gradients_to_apply, gradient_buffer))

            sess.run(Mario.update_gradients, feed_dict=feed_dict_gradients)

            for index, gradient in enumerate(gradient_buffer):
                gradient_buffer[index] = gradient * 0

        if episode % 100 == 0:
            #saver.save(sess, path + "pg-checkpoint", episode)
            print("Episode {} Reward: {} ".format(episode, str(np.mean(total_episode_rewards[-100:]))))

env.close()