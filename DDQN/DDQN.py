#!/usr/bin/env python

import gym
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


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




class env():
    def __init__(self):
        # Launch the simulation with the given launchfile name
        self.vel_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.count = 0
        self.count2 = 0
        self.action_pre = 0
        self.when = 0
        self.ang_vel_pre=0
        self.raaa =[0,0,0,0]
        self._seed()
        self.goal_reward =0



    def calculate_observation(self,data,point):
        min_range = 0.2
        done = False
        self.goal_reward =0

        #print point[0] #x
        #print point[1] #y
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
                self.goal_reward =0

        if -8.2< point[0] <-7.1 and -1.5<point[1]<-1.4:  ## 1
           # print ("4point")
            self.goal_reward =100
            done = False

        elif -4.6 < point[0] < -4.5 and 0.6 < point[1] < 1.6:##2
            self.goal_reward = 100
            done = False
            #print ("3point")
        elif -2.7 < point[0] < -1.8 and -1.5 < point[1] < -1.4: ##3
            self.goal_reward = 100
            done = False
            #print ("2point")


        elif -5.8 < point[0] < -5.5 and -6 < point[1] < -2:  # 4
            self.goal_reward = 100
            #print ("goal")
            done = True


        else :
            self.goal_reward = 0

        return data.ranges,done,self.goal_reward

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action,episode_step):

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
       # print vel_cmd
        data = None
        while data is None:
            try:

                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

               # datas = rospy.wait_for_message('/S_ODG_DATA', LaserScan, timeout=5)
                pcck = rospy.wait_for_message('/pf/pose/odom', Odometry, timeout=5)

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




        xlist[episode_step] = pcck.pose.pose.position.x
        ylist[episode_step] = pcck.pose.pose.position.y



        point[0]=pcck.pose.pose.position.x
        point[1]=pcck.pose.pose.position.y
        if episode_step==400:
             with open("/home/sun/path/rf_path/" + str(e) + "DDQN_path.txt", "w") as file:
                 for i in range(episode_step):
                    file.write(str(xlist[i + 1]) + ',')
                    file.write(str(ylist[i + 1]) + ',')


        state, done, self.goal_reward = self.calculate_observation(data, point)
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

#            laser_len = len(datas.ranges)  # 360
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
        return np.asarray(state), reward, done, {}, xlist, ylist




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
        state,done ,self.goal_reward= self.calculate_observation(data, point)

        return np.asarray(state)






class DoubleDQNAgent:


    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = inputs
        self.action_size = outputs

        # these is hyper parameters for the Double DQN
        self.discount_factor = discountFactor
        self.learning_rate = learningRate
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = learnStart
        # create replay memory using deque
        self.memory = deque(maxlen=memorySize)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()


        if self.load_model:
            self.model.load_weights("/home/sun/path/Network Weights/DDQN_Main_Set.h5")



    def build_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(12, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1,len(state)))
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)




if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step

    rospy.init_node('DDQN_NEW', anonymous=True)
    EPISODES = 1500
    xlist = [-5.31687531203] * EPISODES
    ylist = [ -4.09487880244] * EPISODES
    x_data = [0] * EPISODES
    y_data= [0] * EPISODES
    env = env()
    # get size of state and action from environment
   # state_size = env.observation_space.shape[0]
   # action_size = env.action_space.n
    network_inputs = 120
    network_outputs = 120
    memorySize = 1000000
    discountFactor = 0.99
    learningRate = 0.00025
    learnStart = 64
    reward_data = [0] * EPISODES
    rf_path = [0] * EPISODES


    agent = DoubleDQNAgent(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)

    scores, episodes = [], []

    for e in range(EPISODES):
       # print e
        done = False
        score = 0
        episode_step = 0
        state = env.reset()
      #  state = np.reshape(state, [1, state_size])

        while not done:
            '''
            if agent.render:
                env.render()
            '''
            # get action for the current state and go one step in environment
            #print state.shape
            #print type(state.reshape(1,len(state)))
            #print type(state)
            #print state

            action = agent.get_action(state)

            next_state, reward, done, info,xlist, ylist= env.step(action,episode_step)

            #next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100


            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("/home/sun/path/Network Weights/DDQN_Main.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                reward_data[e - 1] = score
                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if e > 1500:
                    sys.exit()

        # save the model
        if e % 100 == 0:


            agent.model.save_weights("/home/sun/path/Network Weights/DDQN_Main_Set.h5")

        if e % 100 == 0:

            with open("/home/sun/path/reward_dqn/" + str(e) + "_rewrad_DDQN.txt", "w") as file:
                for i in range (e):
                    file.write(str(reward_data[i + 1]) + ',')


        if e %500==0:
            with open("/home/sun/path/rf_path/"+ str(e)+"DDQN_path.txt", "w") as file:
                for i in range(e):
                    file.write(str(rf_path[i + 1]) + ',')

