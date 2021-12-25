#!/usr/bin/env python

#import gym_gazebo

from distutils.dir_util import copy_tree
import os
import json
import liveplot
import deepq
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
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.count = 0
        self.count2 = 0
        self.action_pre = 0
        self.when = 0
        self.ang_vel_pre=0
        self.raaa =[0,0,0,0]
        self._seed()

    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        print 89798
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                print 89798
                pccc = rospy.wait_for_message('/pf/pose/odom', Odometry, timeout=5)
                print 89798

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


        xlist[episode_step] = pccc.pose.pose.position.x
        ylist[episode_step] = pccc.pose.pose.position.y
        point[0]=pccc.pose.pose.position.x
        point[1]=pccc.pose.pose.position.y



        state,done = self.calculate_observation(data)



        if done ==True:
           self.count=0

        if not done: # false
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
          #  m2, s2 = divmod(int(time.time() - start_time), 60)
           # h2, m2 = divmod(m2, 60)
           # print data.ranges.index(max(data.ranges))

            if action == angmix:

               #print ("Action : " + str(action) + " ang_vel : " + str(ang_vel) + " angmix=" + str(angmix))
           #     ang_vel
                ## -r + le
                reward = 20


            if  action == angmix:

                self.count = self.count + 1
                reward = self.count*50

                #print self.count

        else:
            reward = -200
        #print reward
        return np.asarray(state), reward, done, {},xlist,ylist

    def reset(self):
        point = [0] * 2
        # Resets the state of the environment and returns an initial observation.
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


        state,done = self.calculate_observation(data)

        return np.asarray(state)

if __name__ == '__main__':
    rospy.init_node('DQNTEST_ODG', anonymous=True)
    #REMEMBER!: turtlebot_nn_setup.bash must be executed.

    env = env()

    continue_execution = True


    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 1
        steps = 1000
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 120
        network_outputs = 120
        network_structure = [300,300]
        current_epoch = 0

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open("/need/here/lid_test2(ok)/3841.json") as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights("/need/here/lid_test2(ok)/3841.h5")



    env._max_episode_steps = steps # env returns done after _max_episode_steps
    #env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    xlist = [-5.31687531203] * epochs
    ylist = [ -4.09487880244] * epochs
    x_data = [0] * epochs
    y_data= [0] * epochs
    rf_path=[0]*epochs




    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()

        cumulated_reward = 0
        done = False
        episode_step = 0

        x_data[epoch] = epoch


        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info,xlist,ylist = env.step(action)

            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            print observation
            print type(observation)
            print observation.shape
            deepQ.addMemory(observation, action, reward, newObservation, done)


            observation = newObservation
            if done:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(
                    steps) + " - Cumulated R: " + str(
                    cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (
                       h, m, s))

                with open("/home/sun/path/rf_path/rf_path.txt", "w") as file:
                    for i in range(episode_step):
                        file.write(str(rf_path[i + 1]) + ',')

    env.close()