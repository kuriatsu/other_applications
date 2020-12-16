#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rosba
import numpy as np


def readCsv(filename):
    with open(filename, 'r') as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        return [row for row in reader]


def readRosbag(filename, waypoints, waypoint_interval, goal_confirm_waypoint):

    bag = rosbag.Bag(filename)

    aligned_data = {}
    profile_data = []


    target_object = None
    wall_waypoint_index = None
    wall_waypoint = None
    intervene = None
    is_correcting_data = False


    for topic, msg, time in bag.read_messages():

        if topic is '/managed_objects' and not is_correcting_data:
            for object in msg.objects:
                if object.is_important:
                    target_object = object
                    break

        if topic is '/wall_object' and not is_correcting_data:
            wall_waypoint_index, wall_waypoint = findClosestWaypoint(waypoints, msg.object.pose.position)
            is_correcting_data = True

        if topic is '/feedback_object':
            intervene = 'touch'

        if topic is '/joy':
            if msg.axes[] != 0.0:
                intervene = 'accel'
            if msg.axex[] != 0.0:
                intervene = 'brake'
            if msg.button[]:
                intervene = 'button'


        if topic is '/carla/ego_vehicle/odometry' and is_correcting_data:
            # ros_time = time
            step_data = {}

            ego_waypoint_index, ego_waypoint = findClosestWaypoint(waypoints, msg.pose.position)
            ego_to_wall = (wall_waypoint_index - ego_waypoint_index) * waypoint_interval
            if ego_waypoint > goal_confirm_waypoint:
                ego_to_wall = ((msg.pose.position.x - wall_waypoint[0]) ** 2 + (msg.pose.position.y - wall_waypoint[1]) ** 2) ** 0.5

            if ego_waypoint < -20:
                is_correcting_data = False
                aligned_data[target_object.id] = profile_data
                del profile_data[:]

            else:
                step_data['time'] = msg.haader.stamp
                step_data['vel'] = msg.twist.linear.x
                step_data['dist'] = ego_to_wall
                step_data['intervene'] = intervene

                profile_data.append(step_data)
                step_data.clear()
                intervene = None




def findClosestWaypoint(waypoints, position):
    min_dist = 100000
    closest_waypoint_index = None
    closest_waypoint = None

    for index, waypoint in enumerate(waypoints):
        dist = (position.x - waypoint[0]) ** 2 + (position.y - waypoint[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_waypoint_index = index
            closest_waypoint = waypoint

    return closest_waypoint_index, closest_waypoint


def main():
    waypoints = readCsv("")
    data = readRosbag("", waypoints, waypoint_interval=0.5, goal_confirm_waypoint=5000)

if __name__ == "__main__":
    main()
