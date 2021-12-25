#!/usr/bin/env python
import time
import numpy as np
import roslaunch
from distutils.dir_util import copy_tree
import os
import json
import liveplot
import DQN_ODG_Deepq
import rospy
import math
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from keras.models import load_model
from keras.layers import Lambda
import keras.backend as K
import gym
from gym import wrappers

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir)]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

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

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = env()
    outdir = '/tmp/dqn3/'
    path = '/tmp/dqn3/'
    path2 = '/tmp/need3/'
    plotter = liveplot.LivePlot(outdir)

    continue_execution = False
    #fill this if continue_execution=True
    resume_epoch = '200' # change to epoch to continue from
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json  = resume_path + '.json'

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
        network_inputs = 120
        network_outputs = 120
        network_structure = [300,300]
        current_epoch = 0

        deepQ = DQN_ODG_Deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open(params_json) as outfile:
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
            last100Scores = d.get('last100Scores')

        deepQ = DQN_ODG_Deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights(weights_path)

        #clear_monitor_files(outdir)
       # copy_tree(monitor_path,outdir)



    env._max_episode_steps = steps # env returns done after _max_episode_steps
  #  env = gym.wrappers.Monitor(env, outdir, force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 120
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        cumulated_reward = 0
        cumulated_reward2 =0
        done = False
        episode_step = 0

        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action,cumulated_reward2)


            cumulated_reward += reward

            cumulated_reward2 = cumulated_reward
            #print cumulated_reward
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
                if last100ScoresIndex >= 120:# last100ScoresIndex >= 100
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%200==0:
                        #save model weights and monitoring data every 100 epochs.
                        deepQ.saveModel(path + str(epoch) + '.h5')
                       # env._flush()
                        #copy_tree(outdir,path+str(epoch))
                        #save simulation parameters.
                        parameter_keys = ['epochs', 'steps', 'updateTargetNetwork', 'explorationRate', 'minibatch_size',
                                          'learnStart', 'learningRate', 'discountFactor', 'memorySize',
                                          'network_inputs', 'network_outputs', 'network_structure', 'current_epoch',
                                          'last100Scores']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch,last100Scores]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(path+str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)



                    if cumulated_reward >= 5000:
                        deepQ.saveModel(path2 + str(epoch) + '.h5')

                        parameter_keys = ['epochs', 'steps', 'updateTargetNetwork', 'explorationRate', 'minibatch_size',
                                          'learnStart', 'learningRate', 'discountFactor', 'memorySize',
                                          'network_inputs', 'network_outputs', 'network_structure', 'current_epoch']

                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size,
                                            learnStart, learningRate, discountFactor, memorySize, network_inputs,
                                            network_outputs, network_structure, epoch]

                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))

                        with open(path2 + str(epoch) + '.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)


            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)
     #   if epoch % 10 == 0:
      #      plotter.plot(env)


    env.close()



