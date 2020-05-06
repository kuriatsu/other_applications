#! /usr/bin/python
# -*- coding : utf-8 -*-

import rospy


from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

class TopicAnalyze(object):
    def __init__(self):
        rospy.init_node('topic_analyze_node', anonymous=True)

        self.sub_points_raw = rospy.Subscriber('/points_raw', PointCloud2, self.pointsRawCallback)
        self.sub_points_filtered = rospy.Subscriber('/filtered_points', PointCloud2, self.filteredPointsCallback)
        self.sub_points_no_ground = rospy.Subscriber('/points_no_ground', PointCloud2, self.pointsNoGroundCallback)

        self.points_raw_time = None
        self.filtered_points_delay = None
        self.points_no_ground_time = None

    def pointsRawCallback(self, points):
        self.points_raw_time = rospy.Time(0)
        print('points_raw : ', self.points_raw_time)

    def filteredPointsCallback(self, points):
        # self.filtered_points_time =
        print('filtered_points : ', points.header.stamp)

    def pointsNoGroundCallback(self, points):
        # self.filtered_points_time =
        print('points_no_ground : ', points.header.stamp)


def main():
    topic_analyze = TopicAnalyze()
    rospy.spin()

if __name__ == '__main__':
    main()
