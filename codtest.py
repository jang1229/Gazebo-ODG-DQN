
#!/usr/bin/env python

#import gym_gazebo

import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
import math
import pylab

from distutils.dir_util import copy_tree
import os
import json
import rospy
import time
import numpy as np

from gym import utils, spaces

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry

from gym.utils import seeding
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty

import matplotlib.pyplot as plt
import matplotlib.pyplot as pp

import matplotlib.image as img
from PIL import Image
from math import cos, sin, radians, pi



class env():

    def __init__(self):
        # Launch the simulation with the given launchfile name
        self.vel_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.count = 0
        self.count2 = 0
        self.action_pre = 0
        self.when = 0
        self.ang_vel_pre=0
        self.raaa =[0,0,0,0]
        self.goal_reward = 0


    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        self.goal_reward =0

        #print point[0] #x
        #print point[1] #y
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
                self.goal_reward =0


        return data.ranges,done,self.goal_reward

    def step(self, action):

        point = [0] * 2
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        max_ang_speed = 0.7
        ang_vel = (action-60)*max_ang_speed*0.01 #from (-0.33 to + 0.33)
        vel_cmd = AckermannDriveStamped()
        vel_cmd.drive.speed= 0.5
        vel_cmd.drive.steering_angle = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:

                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

                datas = rospy.wait_for_message('/S_ODG_DATA', LaserScan, timeout=5)
              #  psdas = rospy.wait_for_message('/pf/pose/odom', Odometry, timeout=5)

            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")



        angmix= data.ranges.index(max(data.ranges))
        data.ranges = np.asarray(data.ranges)
        data.ranges[np.isinf(data.ranges)] = 12




      #  xlist[episode_step] = psdas.pose.pose.position.x
       # ylist[episode_step] = psdas.pose.pose.position.y
       # point[0]=psdas.pose.pose.position.x
       # point[1]=psdas.pose.pose.position.y


        state, done, self.goal_reward = self.calculate_observation(data)
        self.when = 0



        #print xlist
        if done ==True:
           self.count=0
           self.count2 =0
           self.when=0

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15 * (max_ang_speed - abs(ang_vel) + 0.0335), 2) + self.goal_reward

            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))

            laser_len = len(datas.ranges)  # 360
            # print ("Action : "+str(action)+" datas.angle_min:"+str (datas.angle_min))
            #  print action
            self.raaa[self.when + 1] = action
            self.action_prection = self.raaa[self.when]

            self.when = self.when + 1

            '''
            if action ==datas.angle_min:
               # print "***"
               #print ("Action : " + str(action) + " ang_vel : " + str(ang_vel) + " last100ScoresIndex=" + str(angmix))
           #     ang_vel
                ## -r + le
                reward = 20


            if  action == datas.angle_min:

                self.count2 = self.count2+ 1
                #print self.count
                reward =self.count2*5

                if abs(self.action_prection -action) <15:
                    self.count = self.count + 1
                   # print abs(self.action_prection -action)
                    #print ("Action : " + str(action) + " action_pre : " + str(self.action_prection ) )
                    reward=abs(self.action_prection - action)*self.count *100


                #print self.count
        '''

            if action == angmix:

              #  print ("Action : " + str(action) + " angmix=" + str(angmix))
           #     ang_vel
                ## -r + le
                reward = 20

        else:
            reward = -300
        # print state.type

        self.raaa[0] = action

        # print reward
        return np.asarray(state), reward, done, {}




    def reset(self):
        point = [0] * 2


        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        data.ranges = np.asarray(data.ranges)
        data.ranges[np.isinf(data.ranges)] = 12
        point[0] = -5.38660481757
        point[1] = -4.09487880244
        point=np.asarray(point)
        state,done ,self.goal_reward= self.calculate_observation(data)

        return np.asarray(state)









class DDQNAgent:


    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = (84, 84, 4)
        self.action_size = outputs

        # these is hyper parameters for the Double DQN
        self.discount_factor = discountFactor
        self.learning_rate = learningRate
        self.update_target_rate = 10000
        self.no_op_steps = 30
        self.epsilon = 1.0
        self.exploration_steps = 1000000.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps



        self.epsilon_decay = 0.999
        self.epsilon_min = 0.02
        self.batch_size = 32
        self.train_start = learnStart
        # create replay memory using deque
        self.memory = deque(maxlen=memorySize)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()
        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/DDQN_Main', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())




        if self.load_model:
            self.model.load_weights("/home/sun/path/Network Weights/DDQN_Main.h5")


    def optimizer(self):
        a = K.placeholder(shape=(None, ), dtype='int32')
        y = K.placeholder(shape=(None, ), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()

        return model
    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size, ))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            done.append(mini_batch[i][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if done[i]:
                target[i] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average_Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
  #  pre=np.zeros((24, 24))
  #  xdd = [0] * 120
  #  ydd = [0] * 120
  #  for x in range (0,120):
  #      xdd[x]=np.cos((30+x)*np.pi / 180.)*observe[119-x]
  #      ydd[x]= np.sin((30+x)*np.pi / 180.) * observe[119-x]

    #print observe
    #print np.cos((30.)*np.pi / 180.)
    plt.plot (observe)
    plt.savefig('./sin.png')

    path = './sin.png'

    image_pil = Image.open(path)
    image = np.array(image_pil)

    #print image.shape
    #area=[50,50,300,600]
    #cropped_img=image.crop(area)
    #plt.imshow(cropped_img)
    #plt.show()



    processed_observe = np.uint8(
        resize(rgb2gray(image), (84, 84), mode='constant') * 255)
    return processed_observe


def file_read(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    measures =120
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step

    rospy.init_node('DeDQN_NEW', anonymous=True)
    EPISODES = 1500
    xlist = [-5.31687531203] * EPISODES
    ylist = [ -4.09487880244] * EPISODES
    x_data = [0] * EPISODES
    y_data= [0] * EPISODES
    env = env()
    # get size of state and action from environment
   # state_size = env.observation_space.shape[0]
   # action_size = env.action_space.n
    network_inputs = 3
    network_outputs = 120
    memorySize = 1000000
    discountFactor = 0.99
    learningRate = 0.00025
    learnStart = 64



    agent = DDQNAgent(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)

    scores, episodes, global_step = [], [], 0


    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()
        episode_step=0
        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)



        # At start of episode, there is no preceding frame.
        # So just copy initial states to make history
        state = pre_processing(observe)

        #state=file_read(observe)
        #print state.shape

        history = np.stack((state, state, state, state), axis=2)
        #print history.shape
        history = np.reshape([history], (1, 84, 84, 4))
        #print history.shape
        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)
            # pre-process the observation --> history
            next_state = pre_processing(observe)

            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            # if the agent missed ball, agent is dead --> episode is not over


            reward = np.clip(reward, -1., 1.)

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, done)
            # every some time interval, train model
            agent.train_replay()
            # update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            # if agent is dead, then reset the history
            if done:
                done = False
            else:
                history = next_history

                # if done, plot the score over episodes
                if done:
                    if global_step > agent.train_start:
                        stats = [score, agent.avg_q_max / float(step), step,
                                 agent.avg_loss / float(step)]
                        for i in range(len(stats)):
                            agent.sess.run(agent.update_ops[i], feed_dict={
                                agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, e + 1)

                    print("episode:", e, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon,
                          "  global_step:", global_step, "  average_q:",
                          agent.avg_q_max/float(step), "  average loss:",
                          agent.avg_loss/float(step))

                    agent.avg_q_max, agent.avg_loss = 0, 0


            # save the model
            if e % 100 == 0:
                agent.model.save_weights("/home/sun/path/Network Weights/DDQN_Main.h5")

