#!/usr/bin/env python
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_gym_ros.msg import RaceInfo

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_gym_ros.msg import RaceInfo

from tf2_ros import transform_broadcaster
from tf.transformations import quaternion_from_euler

import numpy as np

import gym


class Agent(object):
    def __init__(self):
        self.ego_scan_topic = rospy.get_param('ego_scan_topic')
        self.ego_odom_topic = rospy.get_param('ego_odom_topic')
        self.opp_odom_topic = rospy.get_param('opp_odom_topic')
        self.ego_drive_topic = rospy.get_param('ego_drive_topic')
        self.race_info_topic = rospy.get_param('race_info_topic')

        self.scan_distance_to_base_link = rospy.get_param('scan_distance_to_base_link')

        self.map_path = rospy.get_param('map_path')
        self.map_img_ext = rospy.get_param('map_img_ext')
        print(self.map_path, self.map_img_ext)
        exec_dir = rospy.get_param('executable_dir')

        scan_fov = rospy.get_param('scan_fov')
        scan_beams = rospy.get_param('scan_beams')
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams

        csv_path = rospy.get_param('waypoints_path')

        wheelbase = 0.3302
        mass = 3.47
        l_r = 0.17145
        I_z = 0.04712
        mu = 0.523
        h_cg = 0.074
        cs_f = 4.718
        cs_r = 5.4562

        self.racecar_env = gym.make('f110_gym:F110-v0')
        self.racecar_env.init_map(self.map_path, self.map_img_ext, False, False)
        self.racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)
        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

    def scan_callback(self, scan_msg):
        initial_state = {'x': [0.0, 2.0], 'y': [0.0, 2.0], 'theta': [0.0, 0.0]}
        self.racecar_env.reset(initial_state)
        print initial_state
        # print('got scan, now plan')
        drive = AckermannDriveStamped()
        drive.drive.speed = 0.3
        drive.drive.steering_angle = 0.2
        self.drive_pub.publish(drive)


if __name__ == '__main__':
    rospy.init_node('dummy_agent')
    dummy_agent = Agent()
    rospy.spin()


