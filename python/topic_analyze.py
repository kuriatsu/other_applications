#! /usr/bin/python
# -*- coding : utf-8 -*-

import rospy
import csv

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from autoware_msgs.msg import Lane
from autoware_msgs.msg import DetectedObjectArray
from std_msgs.msg import String

from ras_carla.msg import RasObjectArray
from visualization_msgs.msg import InteractiveMarkerUpdate
from visualization_msgs.msg import InteractiveMarkerFeedback
from derived_object_msgs.msg import ObjectArray

class TopicAnalyze(object):
    def __init__(self):
        rospy.init_node('topic_analyze_node', anonymous=True)

        # # sensor
        # self.sub_points_raw = rospy.Subscriber('/points_raw', PointCloud2, self.pointsRawCallback)
        # self.points_raw_time = rospy.get_time()
        # self.points_raw_rate = ['points_raw']
        #
        self.sub_image_raw = rospy.Subscriber('/carla/ego_vehicle/camera/rgb/ros_camera/image_color', Image, self.imageRawCallback)
        self.image_raw_time = rospy.get_time()
        self.image_raw_rate = ['image_raw']
        #
        # # pre processing
        # self.sub_points_filtered = rospy.Subscriber('/filtered_points', PointCloud2, self.filteredPointsCallback)
        # self.filtered_points_time = rospy.get_time()
        # self.filtered_points_delay = ['filtered_points']
        #
        # self.sub_points_no_ground = rospy.Subscriber('/points_no_ground', PointCloud2, self.pointsNoGroundCallback)
        # self.points_no_ground_time = rospy.get_time()
        # self.points_no_ground_delay = ['points_no_ground']
        #
        # # recognition localization
        # self.sub_ndt = rospy.Subscriber('/ndt_pose', PoseStamped, self.ndtCallback)
        # self.ndt_pose_time = rospy.get_time()
        # self.ndt_pose_delay = ['ndt_pose']
        #
        # # recognition detection lidar
        # self.sub_lidar_objects = rospy.Subscriber('/detection/lidar_objects', DetectedObjectArray, self.lidarObjectsCallback)
        # self.lidar_objects_time = rospy.get_time()
        # self.lidar_objects_delay = ['lidar_objects']
        #
        # self.sub_lidar_tracked_objects = rospy.Subscriber('/detection/lidar_tracker/objects', DetectedObjectArray, self.lidarTrakcedObjectsCallback)
        # self.lidar_tracked_objects_time = rospy.get_time()
        # self.lidar_tracked_objects_delay = ['lidar_tracked_objects']
        #
        # # recognition detection camera
        # self.sub_vision_objects = rospy.Subscriber('/detection/vision_objects', DetectedObjectArray, self.visionObjectCallback)
        # self.vision_objects_time = rospy.get_time()
        # self.vision_objects_delay = ['vision_objects']
        #
        # self.sub_vision_tracked_objects = rospy.Subscriber('/detection/tracked_objects', DetectedObjectArray, self.visionTrackedObjectsCallback)
        # self.vision_tracked_objects_time = rospy.get_time()
        # self.vision_tracked_objects_delay = ['vision_tracked_objects']
        #
        # # recognition detection combine
        # self.sub_combined_objects = rospy.Subscriber('/detection/combined_objects', DetectedObjectArray, self.combinedObjectsCallback)
        # self.combined_objects_time = rospy.get_time()
        # self.combined_objects_delay = ['combined_objects']
        #
        # # decision
        # self.sub_final_waypoints = rospy.Subscriber('/final_waypoints', Lane, self.finalWaypointsCallback)
        # self.final_waypoints_time = rospy.get_time()
        # self.final_waypoints_delay = ['final_waypoints']
        #
        # # control
        # self.sub_twist_cmd = rospy.Subscriber('/twist_cmd', TwistStamped, self.twistCmdCallback)
        # self.twist_cmd_time = rospy.get_time()
        # self.twist_cmd_delay = ['twist_cmd']

        # carla HMI check
        self.sub_carla_objects = rospy.Subscriber('/carla/ego_vehicle/objects', ObjectArray, self.carlaObjectsCallback)
        self.carla_objects_time = rospy.get_time()
        self.carla_objects_rate = ['carla_objects']

        self.sub_visualized_objects = rospy.Subscriber('/ras_visualizer_node/update', InteractiveMarkerUpdate, self.visualizedObjectsCallback)
        self.visualized_objects_time = rospy.get_time()
        self.visualized_objects_delay = ['visualized_objects']

        self.sub_operation = rospy.Subscriber('/ras_visualizer_node/feedback', InteractiveMarkerFeedback, self.operationCallback)
        self.operation_time = rospy.get_time()

        self.sub_ras_objects = rospy.Subscriber('/managed_objects', RasObjectArray, self.rasObjectsCallback)
        self.ras_objects_time = rospy.get_time()
        self.ras_objects_delay = ['ras_objects']

    # def pointsRawCallback(self, in_msg):
    #     self.points_raw_rate.append(self.points_raw_time - rospy.get_time())
    #     self.points_raw_time = rospy.get_time()
    #     print('points_raw')
    #
    def imageRawCallback(self, in_msg):
        self.image_raw_rate.append(self.image_raw_time - rospy.get_time())
        self.image_raw_time = rospy.get_time()
        print('image_raw')
    #
    # def filteredPointsCallback(self, in_msg):
    #     self.filtered_points_time = rospy.get_time()
    #     self.filtered_points_delay.append(self.filtered_points_time - self.points_raw_time)
    #     # print('filtered_points : ', self.filtered_points_time - self.points_raw_time)
    #
    # def pointsNoGroundCallback(self, in_msg):
    #     self.points_no_ground_time = rospy.get_time()
    #     self.points_no_ground_delay.append(self.points_no_ground_time - self.points_raw_time)
    #     # print('points_no_ground : ', self.points_no_ground_time - self.points_raw_time)
    #
    # def ndtCallback(self, in_msg):
    #     self.ndt_pose_time = rospy.get_time()
    #     self.ndt_pose_delay.append(self.ndt_pose_time - self.filtered_points_time)
    #
    # def lidarObjectsCallback(self, in_msg):
    #     self.lidar_objects_time = rospy.get_time()
    #     self.lidar_objects_delay.append(self.lidar_objects_time - self.points_no_ground_time)
    #
    # def lidarTrakcedObjectsCallback(self, in_msg):
    #     self.lidar_tracked_objects_time = rospy.get_time()
    #     self.lidar_tracked_objects_delay.append(self.lidar_tracked_objects_time - self.lidar_objects_time)
    #
    # def visionObjectCallback(self, in_msg):
    #     self.vision_objects_time = rospy.get_time()
    #     self.vision_objects_delay.append(self.vision_objects_time - self.image_raw_time)
    #
    # def visionTrackedObjectsCallback(self, in_msg):
    #     self.vision_tracked_objects_time = rospy.get_time()
    #     self.vision_tracked_objects_delay.append(self.vision_tracked_objects_time - self.vision_objects_time)
    #
    # def combinedObjectsCallback(self, in_msg):
    #     self.combined_objects_time = rospy.get_time()
    #     self.combined_objects_delay.append(self.combined_objects_time - max(self.vision_tracked_objects_time, self.lidar_tracked_objects_time))
    #
    # def finalWaypointsCallback(self, in_msg):
    #     self.final_waypoints_time = rospy.get_time()
    #     self.final_waypoints_delay.append(self.final_waypoints_time - self.combined_objects_time)
    #
    # def twistCmdCallback(self, in_msg):
    #     self.twist_cmd_time = rospy.get_time()
    #     self.twist_cmd_delay.append(self.twist_cmd_time - self.final_waypoints_time)

    def carlaObjectsCallback(self, in_msg):
        self.carla_objects_rate.append(rospy.get_time() - self.carla_objects_time)
        self.carla_objects_time = rospy.get_time()

    def visualizedObjectsCallback(self, in_msg):
        self.visualized_objects_time = rospy.get_time()
        self.visualized_objects_delay.append(self.visualized_objects_time - self.carla_objects_time)

    def operationCallback(self, in_msg):
        self.operation_time = rospy.get_time()

    def rasObjectsCallback(self, in_msg):
        if self.operation_time is not None:
            self.ras_objects_time = rospy.get_time()
            self.ras_objects_delay.append(self.ras_objects_time - self.operation_time)
            self.operation_time = None

    def save(self):
        buf = []
        # buf.append(self.points_raw_rate)
        buf.append(self.image_raw_rate)
        # buf.append(self.points_no_ground_delay)
        # buf.append(self.filtered_points_delay)
        # buf.append(self.ndt_pose_delay)
        # buf.append(self.lidar_objects_delay)
        # buf.append(self.lidar_tracked_objects_delay)
        # buf.append(self.vision_objects_delay)
        # buf.append(self.vision_tracked_objects_delay)
        # buf.append(self.combined_objects_delay)
        # buf.append(self.final_waypoints_delay)
        # buf.append(self.twist_cmd_delay)

        buf.append(self.carla_objects_rate)
        buf.append(self.visualized_objects_delay)
        buf.append(self.ras_objects_delay)

        print(buf)
        with open('topic_analyze_result.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(buf)

        print('results saved')

def main():
    try:
        topic_analyze = TopicAnalyze()
        print('started')
        rospy.spin()
    finally:
        topic_analyze.save()
        print('finished')


if __name__ == '__main__':
    main()
