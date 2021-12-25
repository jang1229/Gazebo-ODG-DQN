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
import math
import gym
from gym import utils, spaces
import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan


from nav_msgs.msg import Path, Odometry
from gym.utils import seeding
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
import tf2_ros
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Twist, Transform, TransformStamped
import matplotlib.pyplot as plt




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

    def error_average(self, x_point, y_point,episode_step):
        Origin_x = -5
        Origin_y = -1.5
        Radius = 2.5

        Distance = math.sqrt(pow((Origin_x - x_point[episode_step]), 2) + pow((Origin_y - y_point[episode_step]), 2))
        #print ("**")
        #print Distance
        #print ("**")
        Distance = abs(Radius - Distance)

        return Distance

    def calculate_observation(self,data,point):
        min_range = 0.2
        done = False
        goal_reward =0
        #print point[0] #x
        #print point[1] #y
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
                goal_reward =0
        if -8.2< point[0] <-7.1 and -1.5<point[1]<-1.4:  ## 1
            #print ("4point")
            goal_reward =100
            done = False

        if -4.6 < point[0] < -4.5 and 0.6 < point[1] < 1.6:##2
            goal_reward = 100
            done = False
           # print ("3point")
        if -2.7 < point[0] < -1.8 and -1.5 < point[1] < -1.4: ##3
            goal_reward = 100
            done = False
            #print ("2point")
        if -5.9 < point[0] < -5.8 and -4.7 < point[1] < -3.8:  # 4
            goal_reward = 100
            #print ("1point")
            done = False


        if -5.7< point[0] <-5.6 and -4.7<point[1]<-3.8:
            done = True
            goal_reward =0
        return data.ranges,done,goal_reward

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

        data = None
        while data is None:
            try:

                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                pc = rospy.wait_for_message('/pf/pose/odom', Odometry, timeout=5)
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



        self.when = 0


        xlist[episode_step] = pc.pose.pose.position.x
        ylist[episode_step] = pc.pose.pose.position.y
        point[0]=pc.pose.pose.position.x
        point[1]=pc.pose.pose.position.y
        episode_step_c=0



        state, done, goal_reward = self.calculate_observation(data, point)
        #print xlist
        if done ==True:
           self.count=0
           self.count2 =0
           self.when=0


        if not done: # false
            # Straight reward = 5, Max angle reward = 0.5
            #print goal_reward
            #episode_step_c+=episode_step

            #print episode_step_c

            #print episode_step
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2) +goal_reward

            #print reward
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
          #  m2, s2 = divmod(int(time.time() - start_time), 60)
           # h2, m2 = divmod(m2, 60)
           # print data.ranges.index(max(data.ranges))

            self.raaa[self.when+1] = action
            self.action_prection = self.raaa[self.when]

            self.when = self.when + 1

            #print ("Action : " + str(action) + " action_pre : " + str(self.action_prection ) )
            #
            '''
            if action == angmix:

              #  print ("Action : " + str(action) + " angmix=" + str(angmix))
           #     ang_vel
                ## -r + le
                reward = 20


            if  action == angmix:
                self.count2 = self.count2+ 1
                #print self.count
                reward =self.count2*5

                if abs(self.action_prection -action) <15:
                    self.count = self.count + 1
                   # print abs(self.action_prection -action)
                    #print ("Action : " + str(action) + " action_pre : " + str(self.action_prection ) )
                    reward=abs(self.action_prection - action)*self.count *100

   '''

            if action == angmix:

              #  print ("Action : " + str(action) + " angmix=" + str(angmix))
           #     ang_vel
                ## -r + le
                reward = 20


        else:
            reward = -300
        #print reward
        self.raaa[0] = action

        return np.asarray(state), reward, done, {},xlist,ylist

    def reset(self):
        point=[0] * 2

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
        state,done ,goal_reward= self.calculate_observation(data, point)

        return np.asarray(state)







if __name__ == '__main__':
    rospy.init_node('DDQN_ODG', anonymous=True)
    #REMEMBER!: turtlebot_nn_setup.bash must be executed.

    env = env()
    outdir = '/tmp/ddqn/'
    path = '/tmp/ddqn/'
    path2 = '/tmp/need/'

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

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        print 'when'
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

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights(weights_path)


    env._max_episode_steps = steps # env returns done after _max_episode_steps
    #env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)




    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0
    average_data=0
    start_time = time.time()
    xlist = [0] * epochs
    ylist = [0] * epochs
    x_data = [0] * epochs
    y_data= [0] * epochs
    averagelist=[0]*epochs
    reward_print=0
    reward_data=[0] *epochs

    episode_step_c = [0] * epochs
    episode_step_end=0
    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):

        observation = env.reset()
        #node = OdometryNode()
        cumulated_reward = 0
        cumulated_average=0
        done = False
        episode_step = 0
        action_per =0
        tempp=0
        cumulated_reward1=0
        x_data[epoch]=epoch
        y_data[epoch-2]=average_data

        #print epoch
        #print y_data
        #print average_data
      #  plt.subplot(121)
      #  plt.scatter(epoch, average_data)
      #  plt.pause(0.001)
      #  plt.title("Accuracy", fontsize=15)
      #  plt.xlabel("epoch", fontsize=13)
       # plt.ylabel("error_average", fontsize=13)



        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

             #reset
            # print stepCounter #noreset




            newObservation, reward, done, info,xlist,ylist = env.step(action,episode_step)



            cumulated_reward += reward

            #print cumulated_reward

            error_average = env.error_average(xlist,ylist,episode_step)
            cumulated_average +=error_average
            #print cumulated_average
            #print episode_step
            average_data= cumulated_average/(episode_step+1)



            #cumulated_reward += reward
            if done :
                episode_step_end =0
                reward_data[epoch]=cumulated_reward

                #episode_step_end=episode_step
                #cumulated_reward=cumulated_reward+episode_step_end

                #print episode_step_end
           # if done:
                #print cumulated_reward/(episode_step+1)
             #   plt.subplot(122)
                #plt.scatter(epoch, cumulated_reward/(episode_step+1))
             #   plt.scatter(epoch, cumulated_reward )
             #   plt.pause(0.001)
              #  plt.title("Reward", fontsize=15)
              #  plt.xlabel("epoch", fontsize=13)
               # plt.ylabel("Total_Reward", fontsize=13)


            #cumulated_reward += reward


           # print error_average

            if highest_reward < cumulated_reward:


                highest_reward = cumulated_reward

               # plt.scatter(xlist, ylist)
                #plt.pause(0.001)


            deepQ.addMemory(observation, action, reward, newObservation, done)
            #print highest_reward
            #print cumulated_reward
            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation

            if done:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:

                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))


                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))



                    if (epoch)%100==0:

                        with open("/home/sun/path/re_path/"+str(epoch)+"_Accuracy_DQN.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(y_data[i]) + ',')

                    if (epoch)%100==0:

                        with open("/home/sun/path/reward_dqn/"+str(epoch)+"_rewrad_DQN.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(reward_data[i+1]) + ',')





                    if cumulated_reward >= 3000:
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

                        epoch =10000

                    if cumulated_reward >= 3000:
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




    env.close()