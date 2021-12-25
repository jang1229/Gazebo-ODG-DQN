#!/usr/bin/env python
import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import osmath
import random

from gym import wrappers

import time
from distutils.dir_util import copy_tree
import os
import json
from keras.models import Sequential, load_model
from keras import Sequential, optimizers
from keras.models import load_model

from keras.initializers import normal
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import memory

from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from keras import backend as K

K.set_image_data_format('channels_first')


class env():

    def __init__(self):
        # Launch the simulation with the given launchfile name
        self.vel_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=5)

        self.unpause = rospy.ServiceProxy('/gdataazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.reward_range = (-np.inf, np.inf)

        self._seed()

        self.last50actions = [0] * 50

        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1


    def calculate_observation(self, image_data):
        min_range = 220
        done = False
        # image_data= np.asarray(image_data)
        #cv2.imshow('s',image_data)
        # print image_data
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #print image_data

        # image_data = image_data.reshape(-1, )
        #print image_data.shape  # 84,84,3

        i = 30
        #i = 80

        #for j in range(32, 52):
        for j in range(10, 22):

            if np.all(50<=image_data[i, j, 0] <= 55 and 25<=image_data[i, j, 1] <= 35 and 10<=image_data[i, j, 2] <= 14):
                #        print image_data

                done = True

        #        image_data = image_data.reshape(-1, )
        #        for i, item in enumerate(image_data):

        #            if (min_range > image_data[i] > 200):
        # print done

        #                done = True

        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # 21 actions
        max_ang_speed = 0.3
        ang_vel = (action - 10) * max_ang_speed * 0.1  # from (-0.33 to + 0.33)

        vel_cmd = AckermannDriveStamped()
        vel_cmd.drive.speed = 0.5
        vel_cmd.drive.steering_angle = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        image_data = None
        success = False
        cv_image = None
        while image_data is None:

            try:
                image_data = rospy.wait_for_message('/camera/zed/rgb/image_rect_color', Image, timeout=5)

                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                # temporal fix, check image is not corrupted

                # if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                #   success = True
                # print 'success'
                # else:
                # pass
                # print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.last50actions.pop(0)  # remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)

        cv_image2 = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        # cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        done = self.calculate_observation(cv_image2)
        # 21 actions

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15 * (max_ang_speed - abs(ang_vel) + 0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))


        else:
            reward = -200

        # state2= state.reshape(-1)
        # print state2.shape
        # for j in range(0, 7056):
        # print state2[j]

        return state, reward, done, {}

        # test STACK 4
        #cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        #self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        #return self.s_t, reward, done, {}  # observation, reward, done, info

    def reset(self):

        self.last50actions = [0] * 50  # used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        image_data = None
        success = False
        cv_image = None
        while image_data is None:  # or success is False:
            try:
                image_data = rospy.wait_for_message('/camera/zed/rgb/image_rect_color', Image, timeout=5)

                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                # temporal fix, check image is not corrupted
                # if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                #   success = True
                # else:
                #   pass
                # print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        # cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])

        return state

        # test STACK 4
        #self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        #self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])

        # print self.s_t.shape
        #return self.s_t


class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """

    def __init__(self, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self):
        model = self.createModel()
        self.model = model

    def createModel(self):
        # Network structure must be directly changed here.

        model = Sequential()
        model.add(Convolution2D(16, (3, 3), strides=(2, 2), input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(16, (3, 3), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(network_outputs))
        # adam = Adam(lr=self.learningRate)
        # model.compile(loss='mse',optimizer=adam)
        model.compile(RMSprop(lr=self.learningRate), 'MSE')
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ", i, ": ", weights
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        #    state = np.float32(state / 255.0)
        # if np.random.rand() <= EXPLORE:
        #
        #  print random.randrange(21)
        # return random.randrange(21)
        # else:
        # q_value = self.model.predict(state)
        # return np.argmax(q_value[0])

        predicted = self.model.predict(state)

        # return q_value[0]
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)

        return predicted[0]

    def getMaxQ(self, qValues):

        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:

            return reward

        else:

            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate:
            action = np.random.randint(0, self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)

            X_batch = np.empty((1, img_channels, img_rows, img_cols), dtype=np.float64)
            Y_batch = np.empty((1, self.output_size), dtype=np.float64)
            for sample in miniBatch:

                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)

                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)

                else:
                    qValuesNewState = self.getQValues(newState)

                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, state.copy(), axis=0)

                Y_sample = qValues.copy()

                Y_sample[action] = targetValue

                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)

                X_batch = np.log(X_batch)
                X_batch = np.asarray(X_batch)
                X_batch[np.isnan(X_batch)] = 0
                X_batch[np.isinf(X_batch)] = 50
                Y_batch = np.log(Y_batch)
                Y_batch = np.asarray(Y_batch)
                Y_batch[np.isnan(Y_batch)] = 0
                Y_batch[np.isinf(Y_batch)] = 50
                # print X_batch

                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward] * self.output_size]), axis=0)
            # print X_batch.shape
            # print Y_batch.shape

            self.model.fit(X_batch, Y_batch, batch_size=len(miniBatch), epochs=1, verbose=0)

        his=self.model.fit(X_batch, Y_batch, batch_size=len(miniBatch), epochs=1, verbose=0)
        print his.history

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())


if __name__ == '__main__':
    rospy.init_node('camera_dqn', anonymous=True)
    # REMEMBER!: turtlebot_cnn_setup.bash must be executed.
    env = env()

    outdir = '/tmp/dqn/'

    continue_execution = False
    # fill this if continue_execution=True
    weights_path = '/tmp/dqn200.h5'
    monitor_path = '/tmp/dqn200'
    params_json = '/tmp/dqn200.json'

    img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels
    epochs = 100000
    steps = 1000

    if not continue_execution:
        minibatch_size = 32
        learningRate = 0.00025  # 1e6#1e-3#
        discountFactor = 0.99
        network_outputs = 21
        memorySize = 100000
        learnStart = 100  # 10000 # timesteps to observe before training
        EXPLORE = memorySize  # frames over which to anneal epsilon
        INITIAL_EPSILON = 1  # starting value of epsilon
        FINAL_EPSILON = 0.01  # final value of epsilon
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0

        deepQ = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks()

    else:
        # Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_outputs = d.get('network_outputs')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            EXPLORE = d.get('EXPLORE')
            INITIAL_EPSILON = d.get('INITIAL_EPSILON')
            FINAL_EPSILON = d.get('FINAL_EPSILON')
            loadsim_seconds = d.get('loadsim_seconds')

        deepQ = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks()
        deepQ.loadWeights(weights_path)

    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False

    start_time = time.time()

    # start iterating from 'current epoch'.
    for epoch in xrange(current_epoch + 1, epochs + 1, 1):
        observation = env.reset()

        cumulated_reward = 0

        # number of timesteps
        for t in xrange(steps):
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)
            newObservation, reward, done, info = env.step(action)

            deepQ.addMemory(observation, action, reward, newObservation, done)
            observation = newObservation

            # We reduced the epsilon gradually
            if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
                explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            if stepCounter == learnStart:
                print("Starting learning")

            if stepCounter >= learnStart:
                deepQ.learnOnMiniBatch(minibatch_size, False)

            if (t == steps - 1):
                print ("reached the end")
                done = True

            done
            cumulated_reward += reward

            if done:

                last100Rewards[last100RewardsIndex] = cumulated_reward
                last100RewardsIndex += 1
                if last100RewardsIndex >= 100:
                    last100Filled = True
                    last100RewardsIndex = 0
                m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                h, m = divmod(m, 60)
                if not last100Filled:
                    print ("EP " + str(epoch) + " - {} steps".format(t + 1) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(
                        round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print ("EP " + str(epoch) + " - {} steps".format(t + 1) + " - last100 C_Rewards : " + str(
                        int((sum(last100Rewards) / len(last100Rewards)))) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(
                        round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s))
                    # SAVE SIMULATION DATA

                    if (epoch) % 100 == 0:
                        # save model weights and monitoring data every 100 epochs.
                        #                        deepQ.saveModel('/tmp/dqn'+str(epoch)+'.h5')

                        copy_tree(outdir, '/tmp/dqn' + str(epoch))

                        # save simulation parameters.
                        parameter_keys = ['explorationRate', 'minibatch_size', 'learnStart', 'learningRate',
                                          'discountFactor', 'memorySize', 'network_outputs', 'current_epoch',
                                          'stepCounter', 'EXPLORE', 'INITIAL_EPSILON', 'FINAL_EPSILON',
                                          'loadsim_seconds']
                        parameter_values = [explorationRate, minibatch_size, learnStart, learningRate, discountFactor,
                                            memorySize, network_outputs, epoch, stepCounter, EXPLORE, INITIAL_EPSILON,
                                            FINAL_EPSILON, s]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open('/tmp/turtle_c2c_dqn_ep' + str(epoch) + '.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)
                break

            stepCounter += 1
            if stepCounter % 2500 == 0:
                print("Frames = " + str(stepCounter))

    env.close()
