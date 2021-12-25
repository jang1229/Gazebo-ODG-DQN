#!/usr/bin/env python
import time
import numpy as np
import roslaunch
import cv2, cv_bridge
from distutils.dir_util import copy_tree
import os
import json
import liveplot
import deepload
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from _ServoCtrlMsg import ServoCtrlMsg # ctrl_pkg.msg

from keras.models import load_model

from keras.models import model_from_json


class env():

    def __init__(self):
        # Launch the simulation with the given launchfile name

        #self.vel_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=5)
        self.vel_pub = rospy.Publisher('/manual_drive', ServoCtrlMsg, queue_size=5)
        #self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        #self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        #self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.servoCtrlMsg = ServoCtrlMsg()

        self.reward_range = (-np.inf, np.inf)



    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):

            if (min_range > data.ranges[i] > 0):
                done = True

                vel_cmd = ServoCtrlMsg()
                vel_cmd.throttle = 0.0
                vel_cmd.angle = 0.0
                self.vel_pub.publish(vel_cmd)

        return data.ranges,done

    def step(self, action):
        #rospy.wait_for_service('/gazebo/unpause_physics')
        #try:
         #   self.unpause()
        #except (rospy.ServiceException) as e:
         #   print ("/gazebo/unpause_physics service call failed")

        #max_ang_speed = 0.5
        #max_ang_speed = 0.319
        max_ang_speed = 0.727

        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = ServoCtrlMsg()
        vel_cmd.throttle= 0.00
        vel_cmd.angle = ang_vel
        self.vel_pub.publish(vel_cmd)


        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

       # rospy.wait_for_service('/gazebo/pause_physics')
        #try:
            #resp_pause = pause.call()
         #   self.pause()
        #except (rospy.ServiceException) as e:
         #   print ("/gazebo/pause_physics service call failed")

        data.ranges = np.asarray(data.ranges)
        data.ranges[np.isinf(data.ranges)] = 50

       # print data.ranges

        state,done = self.calculate_observation(data)


        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))


        else:
            reward = -100

        return np.asarray(state), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service('/gazebo/reset_simulation')
        #try:
            #reset_proxy.call()
         #   self.reset_proxy()
        #except (rospy.ServiceException) as e:
         #   print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        #rospy.wait_for_service('/gazebo/unpause_physics')
        #try:
            #resp_pause = pause.call()
         #   self.unpause()
        #except (rospy.ServiceException) as e:
           # print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

       # rospy.wait_for_service('/gazebo/pause_physics')
        #try:
            #resp_pause = pause.call()
         #   self.pause()
        #except (rospy.ServiceException) as e:
         #   print ("/gazebo/pause_physics service call failed")

        data.ranges = np.asarray(data.ranges)
        data.ranges[np.isinf(data.ranges)] = 50


        state,done = self.calculate_observation(data)
       # print state
        return np.asarray(state)



if __name__ == '__main__':
    rospy.init_node('dqn_rw', anonymous=True)

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = env()



    continue_execution = True
    #fill this if continue_execution=True
    resume_epoch = '200' # change to epoch to continue from



    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST

        epochs = 10000
        steps = 1000
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 360
        network_outputs = 21
        network_structure = [300,300]
        current_epoch = 0

        deepQ = deepload.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open("/need/here/42.json") as outfile:
            print 12332
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

        deepQ = deepload.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights("/need/here/42.h5")




    env._max_episode_steps = steps # env returns done after _max_episode_steps

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        cumulated_reward = 0
        done = False
        episode_step = 0

        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)


            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)

                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)
                   # print 'here'

            observation = newObservation

            if done:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 50:# last100ScoresIndex >= 100
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%50==0:
                        #save model weights and monitoring data every 100 epochs.
                        #deepQ.saveModel(path+str(epoch)+'.h5')
                       # env._flush()
                        #copy_tree(outdir,path+str(epoch))
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(path+str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)
                    if cumulated_reward >= 220 :
                        deepQ.saveModel(path + str(epoch) + '.h5')
                        print ( " - Cumulated R: " + str(cumulated_reward))

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)



    env.close()
