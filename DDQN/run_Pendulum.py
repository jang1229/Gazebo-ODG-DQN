"""
Double DQN & Natural DQN comparison,
The Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import time
import numpy as np
import roslaunch
from distutils.dir_util import copy_tree
import os
import json
import liveplot

import rospy
import math
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from keras.models import load_model
from keras.layers import Lambda
import keras.backend as K

from gym import wrappers

import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


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




    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):

            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def step(self, action, cumulated_reward):
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



        datas.ranges = np.asarray(datas.ranges)
        datas.ranges[np.isinf(datas.ranges)] = 12

        done = self.calculate_observation(data)
        # print state
        state = datas.ranges
        self.when = 0

        if done ==True:
           self.count=0
           self.count2 =0
           self.when=0

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
           # print reward
           # print ("**")
            laser_len = len(datas.ranges) # 360
           # print ("Action : "+str(action)+" datas.angle_min:"+str (datas.angle_min))

            self.raaa[self.when+1] = action
            self.action_prection = self.raaa[self.when]

            self.when = self.when + 1


            #if action ==datas.angle_min:
               # print "***"
               #print ("Action : " + str(action) + " ang_vel : " + str(ang_vel) + " angmix=" + str(angmix))
           #     ang_vel
                ## -r + le
               # reward = 5


            if  action == datas.angle_min:

                self.count2 = self.count2+1
                #print self.count
                reward =self.count2*5

                if abs(self.action_prection -action) <15:
                    self.count = self.count + 1
                   # print abs(self.action_prection -action)
                    #print ("Action : " + str(action) + " action_pre : " + str(self.action_prection ) )
                    reward=abs(self.action_prection - action)*self.count *10


                #print self.count



        else:
            reward = -200
        #print state.type

        self.raaa[0] = action


        #print reward
        return np.asarray(state), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
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
                datas = rospy.wait_for_message('/S_ODG_DATA', LaserScan, timeout=5)

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

        datas.ranges = np.asarray(datas.ranges)
        datas.ranges[np.isinf(datas.ranges)] = 12

        done = self.calculate_observation(data)
       # print state
        state=datas.ranges


        return np.asarray(state)





if __name__ == '__main__':
    rospy.init_node('DQN_ODG', anonymous=True)
    env = env()


    def train(RL):

        total_steps = 0
        observation = env.reset()
        while True:
            # if total_steps - MEMORY_SIZE > 8000: env.render()

            action = RL.choose_action(observation)

            f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
            observation_, reward, done, info = env.step(np.array([f_action]))

            reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
            # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
            # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:  # learning
                RL.learn()

            if total_steps - MEMORY_SIZE > 20000:  # stop game
                break

            observation = observation_
            total_steps += 1
        return RL.q


    q_natural = train(natural_DQN)
    q_double = train(double_DQN)

    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()