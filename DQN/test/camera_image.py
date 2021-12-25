#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from nav_msg.msg import Odometry

rospy.init_node('sphero')

pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
rate =rospy.Rate(0.5)

def odom(msg):
    go = Odometry()

    print "pose x = " + str(go.pose.pose.position.x)
    print "pose y = " + str(go.pose.pose.position.y)

    print "or x" +str(go.pose.pose.oroentation.x)
    print "or y" +str(go.pose.pose.oroentation.y)

    sud = rospy.Subscriber('/vesc/odom',Odometry,odom)

def imu(msg):
    allez =imu()

    print "angular z = " + str(allez.angular_velocity.z)
    print "angular y = " + str(allez.angular_velocity.y)

    print "linear x" + str(allez.linear_acceleration.x)
    print "linear y" + str(allez.linear_acceleration.y)
    rate.sleep()
    sub =rospy.Subscriber('')


