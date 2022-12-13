#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rosbag
import numpy as np
import csv
import pickle
import sys

def readCsv(filename):
    with open(filename, 'r') as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        return [row for row in reader]


def readRosbag(filename, waypoints, waypoint_interval, goal_confirm_waypoint, scenario_info):

    bag = rosbag.Bag(filename)
    # scenario_info = np.array(scenario_info)

    extracted_data = {} # final data, all interventions....
    profile_data = [] # data of all time step while correcting data

    # face direction params
    face_angle_thres = -0.8
    face_position_x_min = 80
    face_position_x_max = 400
    face_position_y_min = 200
    face_position_y_max = 450
    current_face_direction = 'front'

    target_object = None
    target_object_waypoint_index = None
    target_object_waypoint = None
    target_object_position = None
    intervene = None
    is_correcting_data = False

    start_time = None
    ego_mileage = None
    last_ego_position = None

    actor_action = None
    experiment_type = None


    for topic, msg, time in bag.read_messages():

        # if topic == '/managed_objects':
        #     print(is_correcting_data)
        #     for object in msg.objects:
        #         print(object.object.id)
        #         if object.object.id == 1000:
        #             print(object)

        if topic == '/managed_objects' and not is_correcting_data:
            for object in msg.objects:
                if object.is_important:
                    print(object.object.id)
                    # get target object info
                    target_object = object
                    target_object_waypoint_index, target_object_waypoint = findClosestWaypoint(waypoints, object.object.pose.position)
                    target_object_position = object.object.pose.position

                    # initialize
                    start_time = time.to_sec()
                    ego_mileage = 0.0
                    last_ego_position = None
                    is_correcting_data = True
                    experiment_type = None
                    actor_action = None

                    # get action type and experiment type frm scenario_info
                    if str(object.object.id) in [info[1] for info in scenario_info]:
                        experiment_type = scenario_info[ [ info[1] for info in scenario_info ].index( str(object.object.id) ) ][2]
                        scenario_id = scenario_info[ [ info[1] for info in scenario_info ].index( str(object.object.id) ) ][0]
                        if 'call' in scenario_id or 'phone' in scenario_id or 'pose' in scenario_id:
                            actor_action = 'pose'
                        if 'cross' in scenario_id:
                            actor_action = 'cross'
                        if 'static' in scenario_id:
                            actor_action = 'static'

                    print('find target : ' + str(object.object.id))

                    break

        # if topic is '/wall_object' and not is_correcting_data:
        #     target_object_waypoint_index, target_object_waypoint = findClosestWaypoint(waypoints, msg.object.pose.position)

        if topic == '/touch_event':
            intervene = 'touch'

        if topic == '/joy':
            if (-1.0 < msg.axes[2] < 0.0) or (0.0 < msg.axes[2] < 1.0) and (-1.0 < msg.axes[3] < 0.0) or (0.0 < msg.axes[3] < 1.0):
                intervene = 'throttle'
            elif (-1.0 < msg.axes[3] < 0.0) or (0.0 < msg.axes[3] < 1.0):
                intervene = 'throttle'
            elif (-1.0 < msg.axes[2] < 0.0) or (0.0 < msg.axes[2] < 1.0):
                intervene = 'brake'
            elif msg.buttons[23]:
                intervene = 'button'

        if topic == '/face_position':
            ## for posenet detection
            # if face_position_x_min < msg.pose.position.x < face_position_x_max and face_position_y_min < msg.pose.position.y < face_position_y_max:
            #     if msg.pose.orientation.z == 0.0:
            #         current_face_direction = current_face_direction
            #     elif msg.pose.orientation.z > face_angle_thres:
            #         current_face_direction = 'ui'
            #     else:
            #         current_face_direction = 'front'

            if msg.angular.z == 1.0:
                current_face_direction = 'ui'

            elif msg.angular.z == -1.0:
                current_face_direction = 'front'


        if topic == '/carla/ego_vehicle/odometry' and is_correcting_data:
            # ros_time = tim
            # data at each time step
            step_data = []

            # skip if not collection range
            if not is_correcting_data:
                continue

            # to calc mileage
            if last_ego_position is not None:
                ego_mileage += ((msg.pose.pose.position.x - last_ego_position.x) ** 2 + (msg.pose.pose.position.y - last_ego_position.y) ** 2) ** 0.5

            last_ego_position = msg.pose.pose.position

            # get waypoint and distance to stop wall position
            ego_waypoint_index, ego_waypoint = findClosestWaypoint(waypoints, msg.pose.pose.position)
            ego_to_wall = (target_object_waypoint_index - ego_waypoint_index) * waypoint_interval

            # deal with the boundary(goal-start) of waypoint
            # if obstacle is in front of boundary and ego_vehicle is behind the boundary
            if ego_to_wall > goal_confirm_waypoint * waypoint_interval:
                ego_to_wall = -(len(waypoints) - target_object_waypoint_index + ego_waypoint_index) * waypoint_interval
                print(len(waypoints), target_object_waypoint_index, ego_waypoint_index)
            # if obstacle is behind the boundary and ego_vehicle is in front of the boundary
            elif ego_to_wall < -goal_confirm_waypoint * waypoint_interval:
                ego_to_wall = (len(waypoints) - ego_waypoint_index + target_object_waypoint_index) * waypoint_interval
                print(len(waypoints), target_object_waypoint_index, ego_waypoint_index)
            # stop collecting data and save it (end)
            print(ego_to_wall)

            if ego_to_wall < -20.0:
                is_correcting_data = False

                # if the objct behind 20m appears again (control intervention remains the object important), cause error
                if not profile_data:
                    continue
                # save data for an actor
                actor_data = {
                    'world_id': target_object.object.id,
                    'data': offsetMileage(profile_data),
                    'experiment_type': experiment_type,
                    'actor_action' : actor_action
                    }
                # save to output dataframe
                extracted_data[target_object.object.id] = actor_data
                profile_data = []

                print('saved for ' + str(target_object.object.id))

            else:
                # store data [time, ego_vel, ego_mileage, target_ego_dist, wp_dist, intervene_type]
                step_data = [
                    time.to_sec() - start_time,
                    msg.twist.twist.linear.x,
                    ego_mileage,
                    ((target_object_position.x - msg.pose.pose.position.x) ** 2 + (target_object_position.y - msg.pose.pose.position.y) ** 2) ** 0.5,
                    ego_to_wall, # 障害物からどれだけ離れているか (offsetMileageで再計算)
                    intervene,
                    current_face_direction
                    ]
                profile_data.append(step_data)
                intervene = None


    return extracted_data



def findClosestWaypoint(waypoints, position):
    min_dist = 100000
    closest_waypoint_index = None
    closest_waypoint = None

    for index, waypoint in enumerate(waypoints):
        dist = (position.x - float(waypoint[0])) ** 2 + (position.y - float(waypoint[1])) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_waypoint_index = index
            closest_waypoint = waypoint

    return closest_waypoint_index, closest_waypoint


def offsetMileage(profile_data):

    mileage_list = [data[3] for data in profile_data]
    offset_mileage = profile_data[mileage_list.index(min(mileage_list))][2]
    for data in profile_data:
        # offset mileage based the crossing point with vehicle and object
        data[2] -= offset_mileage
        if ((data[4] > 0.0 and data[2] < 0.0)) or ((data[4] < 0.0 and data[2] > 0.0)):
            data[2] *= -1

    return profile_data

def savePickle(data):
    with open("/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/" + sys.argv[1] + "/Town01.pickle", 'wb') as f:
        pickle.dump(data, f)


def main():
    waypoints = readCsv("/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/town1.csv")
    scenario_info = readCsv("/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/" + sys.argv[1] + "/actor_id_Town01.csv")
    extracted_data = readRosbag("/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/"  + sys.argv[1] + "/"  + sys.argv[1] + "_Town01.bag", waypoints, 1.0, 100, scenario_info)
    # print(np.array(extracted_data.get(981).get('data')))
    savePickle(extracted_data)


if __name__ == "__main__":
    main()
