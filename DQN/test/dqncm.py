#!/usr/bin/env python
import time
import numpy as np
import cv2
import roslaunch
from distutils.dir_util import copy_tree
import os
import json
import liveplot
import deepqcm
import rospy
import memory


from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class env():

    def __init__(self):
        # Launch the simulation with the given launchfile name

        self.vel_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.reward_range = (-np.inf, np.inf)



        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1

    def calculate_observation(self,image_data):
        min_range = 220
        done = False
        #image_data= np.asarray(image_data)
        #cv2.imshow('s',image_data)
        #print image_data
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #print image_data
        image_data=image_data.reshape(-1,)
        for i, item in enumerate(image_data):

            if (min_range > image_data[i] > 200):
               # print done

                done = True

        return done

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        max_ang_speed = 0.3
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = AckermannDriveStamped()
        vel_cmd.drive.speed= 0.5
        vel_cmd.drive.steering_angle = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        image_data = None
        success = False
        cv_image = None

        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/camera/zed/rgb/image_rect_color', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                # temporal fix, check image is not corrupted
                if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    success = True
                else:
                    pass

            except:
                pass


        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")






        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        # cv_image = np.asarray(cv_image)

        # state = cv_image.reshape(cv_image.shape[0], cv_image.shape[1])
        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
       # print state
        done = self.calculate_observation(state)  ## here

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -200


       # print done

        return state, reward, done, {}

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

        image_data = None
        success = False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/camera/zed/rgb/image_rect_color', Image, timeout=5)

                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

                if not (cv_image[h // 2, w // 2, 0] == 178 and cv_image[h // 2, w // 2, 1] == 178 and cv_image[h // 2, w // 2, 2] == 178):
                    success = True
                else:
                    pass
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

      #  cv_image = np.asarray(cv_image)
      #  cv_image[np.isinf(cv_image)] = 150


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        # cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

       # state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
       # print cv_image.shape
        state = cv_image.reshape(1,1,cv_image.shape[0], cv_image.shape[1])

        #done= self.calculate_observation(state)
        #print state

        #print state.shape

        return state



if __name__ == '__main__':
    rospy.init_node('dqn', anonymous=True)

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = env()

    outdir = '/tmp/dqn2/'
    path = '/tmp/dqn2/'
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
        epochs = 1000
        steps = 1000
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size =64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 84
        network_outputs = 21
        network_structure = [300,300]
        current_epoch = 0


        deepQ = deepqcm.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)

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

        deepQ = deepqcm.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)

        deepQ.initNetworks(network_structure)

        deepQ.loadWeights(weights_path)




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
        #print observation
       # print 77777

        cumulated_reward = 0
        done = False
        episode_step = 0

        # run until env returns done
        while not done:

            # env.render()
            qValues = deepQ.getQValues(observation)
            #print qValues
            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)

            deepQ.addMemory(observation, action, reward, newObservation, done)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward



            if stepCounter >= learnStart:

                if stepCounter <= updateTargetNetwork:

                   ## print minibatch_size # 64

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
                    if (epoch)%1000==0:
                        #save model weights and monitoring data every 100 epochs.
                      #  deepQ.saveModel(path+str(epoch)+'.h5')
                       # env._flush()
                        copy_tree(outdir,path+str(epoch))
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(path+str(epoch)+'.json', 'w') as outfile:
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