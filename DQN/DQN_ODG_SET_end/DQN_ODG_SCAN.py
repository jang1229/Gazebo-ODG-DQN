#!/usr/bin/env python

#import gym_gazebo

from distutils.dir_util import copy_tree
import os
import json
import liveplot
import DQN_ODG_Deepq
import rospy
import time
import numpy as np
import math
import gym
from gym import utils, spaces

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
        self.goal_reward =0

    def error_average(self, x_point, y_point,episode_step):
        Origin_x = -5
        Origin_y = -1.5
        Radius = 2.6

        step_Distance=math.sqrt(pow((x_point[episode_step] - x_point[episode_step-1]), 2) + pow((y_point[episode_step] - y_point[episode_step-1]), 2))
        Distance = math.sqrt(pow((Origin_x - x_point[episode_step]), 2) + pow((Origin_y - y_point[episode_step]), 2))

    #    Distance = abs(Radius - Distance)
        Distance = abs(Radius - Distance)

        return Distance , step_Distance

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

        return done,self.goal_reward

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

                datas = rospy.wait_for_message('/S_ODG_DATA', LaserScan, timeout=5)
                pccc = rospy.wait_for_message('/pf/pose/odom', Odometry, timeout=5)

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




        xlist[episode_step] = pccc.pose.pose.position.x
        ylist[episode_step] = pccc.pose.pose.position.y
        point[0]=pccc.pose.pose.position.x
        point[1]=pccc.pose.pose.position.y
        episode_step_c=0


        done,self.goal_reward = self.calculate_observation(data, point)
        state = datas.ranges
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
        '''

            if  action == datas.angle_min:

                self.count2 = self.count2+ 1
                #print self.count
                reward =self.count2*5
                '''
                if abs(self.action_prection -action) <15:
                    self.count = self.count + 1
                   # print abs(self.action_prection -action)
                    #print ("Action : " + str(action) + " action_pre : " + str(self.action_prection ) )
                    reward=abs(self.action_prection - action)*self.count *100


                #print self.count

 '''

        else:
            reward = -300
        # print state.type

        self.raaa[0] = action

        # print reward
        return np.asarray(state), reward, done, {}, xlist, ylist

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

        point[0] = -5.38660481757
        point[1] = -4.09487880244
        point=np.asarray(point)

        done,self.goal_reward= self.calculate_observation(data, point)

        state = datas.ranges
        return np.asarray(state)







if __name__ == '__main__':
    rospy.init_node('DQN_ODG_MAX', anonymous=True)
    #REMEMBER!: turtlebot_nn_setup.bash must be executed.

    env = env()
    outdir = '/tmp/ddqn/'
    path = '/tmp/ddqn/'
    path2 = '/home/sun/path/Network Weights/'

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
        steps = 10000
        updateTargetNetwork = 100000
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

        deepQ = DQN_ODG_Deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights(weights_path)


    env._max_episode_steps = steps # env returns done after _max_episode_steps
    #env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)




    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    #highest_reward = 0
    average_data=0
    start_time = time.time()
    xlist = [-5.31687531203] * epochs
    ylist = [ -4.09487880244] * epochs
    x_data = [0] * epochs
    y_data= [0] * epochs
    re_dis=[0]*epochs
    averagelist=[0]*epochs
    reward_print=0
    reward_data=[0] *epochs
    episode_step_save=[0]*epochs
    speed_data=[0]*epochs
    time_data=[0]*epochs
    MinODG =0
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
        dis_data1=0
        reward=0
        one_start_time = time.time()
        s1=0
        x_data[epoch]=epoch
        y_data[epoch-2]=average_data
        #print y_data

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
           # print updateTargetNetwork
           # print "~~~~~"
           # print stepCounter
           # print "------"
            # env.render()
            qValues = deepQ.getQValues(observation)
            #print explorationRate

            action = deepQ.selectAction(qValues, explorationRate)
            #print action
             #reset
            # print stepCounter #noreset




            newObservation, reward, done, info,xlist,ylist = env.step(action,episode_step)



            cumulated_reward += reward

            #print cumulated_reward

            error_average,dis_data = env.error_average(xlist,ylist,episode_step)
            dis_data1+=dis_data
            re_dis[epoch - 2] = dis_data1
            episode_step_save[epoch-1]=episode_step
            #print episode_step_save

            #print re_dis


            cumulated_average +=error_average
            #print cumulated_average
            #print episode_step
            average_data= cumulated_average/(episode_step+1)

            reward_data[epoch-1] = cumulated_reward
            #print reward_data
            #cumulated_reward += reward
         #   if done :
                #episode_step_end =0
          #      reward_data[epoch]=cumulated_reward

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

            #if highest_reward < cumulated_reward:


             #   highest_reward = cumulated_reward

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

                m1, s1 = divmod(int(time.time() - one_start_time), 60)
                time_data[epoch - 1] = s1



                try:
                    v1 = dis_data1 / s1
                except ZeroDivisionError:
                    v1=0

                speed_data[epoch - 1] = v1


               # print speed_data
                #print cumulated_reward
               # print time_data



                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:

                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2))+" - Cumulated R: " + str(cumulated_reward))


                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))


                    if (epoch)%100==0:

                        with open("/home/sun/path/Distance/"+str(epoch)+"_Distance_DQN_R.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(re_dis[i]) + ',')

                    if (epoch)%100==0:

                        with open("/home/sun/path/Distance/step"+str(epoch)+"_step_DQN_R.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(episode_step_save[i]) + ',')

                    if (epoch)%100==0:

                        with open("/home/sun/path/re_path/"+str(epoch)+"_Accuracy_DQN_R.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(y_data[i]) + ',')

                    if (epoch)%100==0:

                        with open("/home/sun/path/reward_dqn/"+str(epoch)+"_rewrad_DQN_R.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(reward_data[i+1]) + ',')

                    if (epoch)%100==0:

                        with open("/home/sun/path/speed_time/speed/"+str(epoch)+"_speed_DQN_R.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(speed_data[i+1]) + ',')

                    if (epoch)%100==0:

                        with open("/home/sun/path/speed_time/time/"+str(epoch)+"_time_DQN_R.txt", "w") as file:
                            for i in range(epoch):
                                file.write(str(time_data[i+1]) + ',')



                    if (epoch)%100==0:
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