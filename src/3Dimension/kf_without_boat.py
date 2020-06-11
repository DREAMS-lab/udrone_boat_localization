#!/usr/bin/env python3
"""
State space equations over x-axis without boat speed taken into consideration

"""
import rospy
import tf
from filterpy.kalman import KalmanFilter
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from mpl_toolkits import mplot3d
# create logs
import os

if not os.path.exists('logs'):
    os.makedirs('logs')


class Localization:
    """ localization module for Boat-Udrone system """

    rospy.init_node('LocalizationUdrone', anonymous=True)
    last_t = rospy.Time.now()

    def __init__(self):
        rospy.Subscriber('/aruco_single/pose', PoseStamped, callback=self.aruco_pose_subscriber)
        self.observation = None
        self.listener = tf.TransformListener()

        # Kalman filter definition
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.dt = 0.01

        self.kf.x = np.array([[0.5],
                              [0.5],
                              [0.5],
                              [0.5],
                              [5],
                              [0.5]])

        self.kf.P = np.array([[2, 0, 0, 0, 0, 0],
                              [0, 2, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 2, 0, 0],
                              [0, 0, 0, 0, 2, 0],
                              [0, 0, 0, 0, 0, 2]])

        self.kf.H = np.array([[1., 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]])

        self.kf.R = np.array([[2, 0, 0],
                              [0, 2, 0],
                              [0, 0, 2]])

        self.kf.Q = np.array([[0.002, 0, 0, 0, 0, 0],
                              [0, 0.02, 0, 0, 0, 0],
                              [0, 0, 0.002, 0, 0, 0],
                              [0, 0, 0, 0.02, 0, 0],
                              [0, 0, 0, 0, 0.002, 0],
                              [0, 0, 0, 0, 0, 0.02]])

        self.kf.F = np.array([[1., self.dt, 0, 0, 0, 0],
                              [0,  1., 0, 0, 0, 0],
                              [0, 0, 1., self.dt, 0, 0],
                              [0, 0, 0, 1., 0, 0],
                              [0, 0, 0, 0, 1., self.dt],
                              [0, 0, 0, 0, 0, 1.]])

        # Variables to maintain kalman update only on new aruco observation
        self.aruco_previous_timestamp = rospy.Time.now()
        self.aruco_pose_timestamp = rospy.Time.now()
        self.is_observation = False

        # Plots to stop_count iterations
        self.plot = True
        self.stop_count = 6000  # 6000, means 60 sec if dt is 0.01
        self.count = 0
        if self.plot:
            self.save_time = np.empty((0, 1))  # n x timestamps
            self.save_observation = np.empty((0, 7))  # n x 7, where 7 values are x, y, z and quaternion
            self.save_ground_truth = np.empty((0, 7))  # n x 7, where 7 values are x, y, z and quaternion
            self.save_estimate = np.empty((0, 6))  # n x 2, where 2 values are the state estimates x and xdot
            self.save_observation_time = np.empty((0, 1))  # n x timestamps

        self.run_filter()
        # self.save_data_to_file()
        self.plot_fusion(dim=3, axis=0)

    def save_data_to_file(self):
        np.save('logs/kf_with_boat_ground_truth.npy', self.save_ground_truth)
        np.save('logs/kf_with_boat_observations.npy', self.save_observation)
        np.save('logs/kf_with_boat_observation_time.npy', self.save_observation_time)
        np.save('logs/kf_with_boat_estimates.npy', self.save_estimate)

    def aruco_pose_subscriber(self, msg):
        """ This subscriber publishes the timestamp of the aruco detection.
            This will make sure the rate of Kalman update is synchronized to image detections
        """
        self.aruco_pose_timestamp = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs)

    def run_filter(self):
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
                    print("Aruco Observation: ", trans)

                if self.plot:
                    (gtrans, grot) = self.listener.lookupTransform('/ground_truth_udrone',
                                                                   '/ground_truth_heron',
                                                                   rospy.Time(0))
                    print("Ground Truth: ", gtrans)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as Ex:
                print(Ex)
                rate.sleep()
                continue

            self.kf.predict()
            print("Predict     :", self.kf.x.T.flatten()[0], self.kf.x.T.flatten()[2], self.kf.x.T.flatten()[4])

            # Do kalman update call only when the camera detects an aruco marker
            if self.is_observation:
                self.kf.update(np.array([trans]))
                self.is_observation = False
                if self.plot:
                    self.save_observation_time = np.append(self.save_observation_time,
                                                           np.array([[self.count]]),
                                                           axis=0)
                    self.save_observation = np.append(self.save_observation,
                                                      np.array([trans + rot]),
                                                      axis=0)

                print("Update      :", self.kf.x.T.flatten()[0], self.kf.x.T.flatten()[2], self.kf.x.T.flatten()[4])

            if self.plot:
                self.count += 1
                self.save_time = np.append(self.save_time, np.array([[self.count]]), axis=0)
                self.save_estimate = np.append(self.save_estimate, np.array([self.kf.x.T.flatten()]), axis=0)
                self.save_ground_truth = np.append(self.save_ground_truth,
                                                   np.array([gtrans + grot]),
                                                   axis=0)
                print(self.count)
                if self.count > self.stop_count:
                    break

            rate.sleep()

    def plot_fusion(self, dim=2, axis=0):
        if dim == 2:
            # 2D plots
            ax = plt.subplot(111)
            if self.save_observation_time.shape[0] > 0:
                ax.scatter(self.save_observation_time, self.save_observation[:, axis], marker='o', c="green",
                           label='Sensor Observation')

            # 0 is x-axis, 1 is x-axis vel
            # 2 is y-axis, 3 is y-axis vel
            # 4 is z-axis, 5 is z-axis vel
            ax.plot(self.save_time, self.save_estimate[:, axis*2], ls='-', label='State Estimate')
            ax.plot(self.save_time, self.save_ground_truth[:, axis], ls='-.', label='Ground Truth')
            plt.title("Sensor observation v/s State estimate v/s Ground truth")
            ax.set_xlabel("Time")
            ax.set_ylabel("Position")
            ax.legend()
            plt.legend()
            plt.show()

        elif dim == 3:
            # 3D plots
            # x-axis is the time
            # y-axis and z-axis will show positions

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            # Plot estimates
            x = self.save_time
            y = self.save_estimate[:, axis]
            z = self.save_estimate[:, axis+2]  # x, x_dot, y, y_dot, z, z_dot
            ax.plot3D(x, y, z, 'blue')

            # Plot ground truth
            x = self.save_time
            y = self.save_ground_truth[:, axis]
            z = self.save_ground_truth[:, axis+1]  # x, y, z
            ax.plot3D(x, y, z, 'orange')

            # Plot observations
            x = self.save_observation_time
            y = self.save_observation[:, axis]
            z = self.save_observation[:, axis+1]  # x, y, z
            ax.scatter(x, y, z, color='r')

            ax.set_xlabel('Time')
            ax.set_ylabel('X-axis position')
            ax.set_zlabel('Y-axis position')
            ax.set_title('Sensor observation v/s State estimate v/s Ground truth')
            plt.show()


if __name__ == "__main__":
    Localization()
