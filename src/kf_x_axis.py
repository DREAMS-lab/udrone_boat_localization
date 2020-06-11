#!/usr/bin/env python3
"""
Set up ROS on python3 to get this code working
"""
import rospy
import tf
from filterpy.kalman import KalmanFilter
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped


class Localization:
    """ localization module for Boat-Udrone system """
    rospy.init_node('LocalizationUdrone', anonymous=True)
    last_t = rospy.Time.now()

    def __init__(self):
        rospy.Subscriber('/aruco_single/pose', PoseStamped, callback=self.aruco_pose_subscriber)
        self.observation = None
        self.listener = tf.TransformListener()

        # Kalman filter definition
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.dt = 0.01
        self.kf.x = np.array([[0.5], [0.5]])
        self.kf.P = np.array([[2, 0], [0, 1]])
        self.kf.H = np.array([[1., 0]])
        self.kf.R = np.eye(1) * 1.5
        self.kf.Q = np.eye(2) * 0.002
        self.kf.F = np.array([[1., self.dt],
                              [0., 1]])

        self.plot = True
        self.stop_count = 10000 # , means 50 sec if dt is 0.01
        self.count = 0
        self.aruco_previous_timestamp = rospy.Time.now()
        self.aruco_pose_timestamp = rospy.Time.now()
        self.is_observation = False
        if self.plot:
            self.save_time = []
            self.save_observation = []
            self.save_ground_truth_pos = []
            self.save_pos_estimate = []
            self.save_vel_estimate = []
            self.save_observation_time = []

        self.observations()
        self.plot_fusion()

    def aruco_pose_subscriber(self, msg):
        """ This subscriber publishes the timestamp of the aruco detection.
            This will make sure the rate of Kalman update is synchronized to image detections
        """
        self.aruco_pose_timestamp = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs)

    def observations(self):
        rate = rospy.Rate(100.0)
        # print(kf.x)

        while not rospy.is_shutdown():
            try:
                # (trans, rot) = listener.lookupTransform('to', 'from', rospy.Time(0))
                # timestamp is used to make sure update happens only after an observation
                if (self.aruco_pose_timestamp - self.aruco_previous_timestamp) > rospy.Duration(0):
                    self.is_observation = True
                    (trans, rot) = self.listener.lookupTransform('/udrone', '/world', self.aruco_pose_timestamp)
                    self.aruco_previous_timestamp = self.aruco_pose_timestamp
                    print("yes")
                    print("Aruco Observation: ", trans)

                if self.plot:
                    (gtrans, grot) = self.listener.lookupTransform('/ground_truth_udrone', '/ground_truth_heron', rospy.Time(0))
                    # print("Ground Truth: ", gtrans)

                # Need code for skip when aruco transform is old
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as Ex:
                print(Ex)
                rate.sleep()
                continue

            self.kf.predict()
            print("Predicted pos x: ", self.kf.x[0][0])

            # Do update call only when the camera detects a aruco marker
            if self.is_observation:
                self.kf.update(np.array([trans[0]]))
                self.is_observation = False
                if self.plot:
                    self.save_observation.append(trans[0])
                    self.save_observation_time.append(self.count)

            print("Updated pos x  : ", self.kf.x[0][0])

            if self.plot:
                self.count += 1
                self.save_time.append(self.count)
                self.save_pos_estimate.append(self.kf.x[0][0])
                self.save_vel_estimate.append(self.kf.x[1][0])
                self.save_ground_truth_pos.append(gtrans[0])
                print(self.count)
                if self.count > self.stop_count:
                    break

            rate.sleep()

    def plot_fusion(self):
        # xs = np.array(xs)
        plt.scatter(self.save_observation_time, self.save_observation, marker='o', c="green", label='Position Sensor (x-axis)')
        plt.plot(self.save_time, self.save_pos_estimate, ls='-', label='State Estimate (x-axis)')
        plt.plot(self.save_time, self.save_ground_truth_pos, ls='-.', label='Ground Truth Estimate')
        plt.title("Position sensor (x-axis) v/s State estimate v/s ground truth")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.show()


if __name__ == "__main__":
    Localization()

