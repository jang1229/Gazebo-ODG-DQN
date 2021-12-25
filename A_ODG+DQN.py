#!/usr/bin/env python


import time
import numpy as np
import os
import json
from nav_msgs.msg import Path, Odometry
import matplotlib.pyplot as plt

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray
import math
import numpy as np
from numpy import linalg as LA
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import csv
import os

def path_call():


    epoch=200
    list_file = open("/home/sun/path/re_path/"+str(epoch)+"_Accuracy_DQN.txt", 'r').read().split(',')
#    list_file = open("/home/sun/path/re_path/"test_1_epoch_6400_Accuracy_DQN", 'r').read().split(',')

    step = epoch
  #  print list_file  #dqn ok Time: 6:48:53
    listt = [0] * step
    for i in range(step):
        list_file[i] =float(list_file[i])
        listt[i] = i
    x= np.asarray(list_file[0:step-1])
    y= np.asarray(listt[0:step-1])



    list_file2 = open("/home/sun/path/reward_dqn/"+str(epoch)+"_rewrad_DQN.txt", 'r').read().split(',')

    for i in range(step):
        list_file2[i] = float(list_file2[i])
    x2 = np.asarray(list_file2[0:step-1])
    y2= np.asarray(listt[0:step-1])




    list_file3 = open("/home/sun/path/Distance/" + str(epoch) + "_Distance_DQN.txt", 'r').read().split(',')
    #    list_file = open("/home/sun/path/re_path/"test_1_epoch_6400_Accuracy_DQN", 'r').read().split(',')

    for i in range(step):
        list_file3[i] = float(list_file3[i])
        listt[i] = i
    x3 = np.asarray(list_file3[0:step - 1])
    y3 = np.asarray(listt[0:step - 1])

    list_file4 = open("/home/sun/path/Distance/step" + str(epoch) + "_step_DQN.txt", 'r').read().split(',')
    for i in range(step):
        list_file4[i] = float(list_file4[i])
    x4 = np.asarray(list_file4[0:step - 1])
    y4 = np.asarray(listt[0:step - 1])


    list_file5 = open("/home/sun/path/speed_time/speed/" + str(epoch) + "_speed_DQN.txt", 'r').read().split(',')
    #    list_file = open("/home/sun/path/re_path/"test_1_epoch_6400_Accuracy_DQN", 'r').read().split(',')

    for i in range(step):
        list_file5[i] = float(list_file5[i])
        listt[i] = i
    x5 = np.asarray(list_file5[0:step - 1])
    y5 = np.asarray(listt[0:step - 1])

    list_file6 = open("/home/sun/path/speed_time/time/" + str(epoch) + "_time_DQN.txt", 'r').read().split(',')
    for i in range(step):
        list_file6[i] = float(list_file6[i])
    x6 = np.asarray(list_file6[0:step - 1])
    y6 = np.asarray(listt[0:step - 1])


    plt.subplot(321)
    plt.plot(y,x)
    plt.title("Accuracy", fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel("error average", fontsize=13)


    plt.subplot(322)
    plt.plot(y2,x2)
    plt.title("Reward", fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel("Total Reward", fontsize=13)

    plt.subplot(323)
    plt.plot(y3,x3)
    plt.title("step", fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel("step", fontsize=13)


    plt.subplot(324)
    plt.plot(y4,x4)
    plt.title("Distance ", fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel("Total Distance", fontsize=13)



    plt.subplot(325)
    plt.plot(y5,x5)
    plt.title("Speed Scalar", fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel("Speed", fontsize=13)

    plt.subplot(326)
    plt.plot(y6,x6)
    plt.title("time ", fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel(" Finished time", fontsize=13)




    plt.show()




if __name__ == '__main__':
  path_call()

