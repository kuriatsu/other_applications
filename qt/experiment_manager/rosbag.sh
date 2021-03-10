#! /bin/bash
source /opt/ros/melodic/setup.bash

today=`date +'%Y_%m_%d'`
time=`date +'%H_%M_%S'`
dir=/media/kuriatsu/SamsungKURI/master_study_bag/$today

if [ -e $dir ]; then
	echo "file found"
else
	mkdir -p $dir
fi

rosbag record /managed_objects /wall_object /feedback_object /ff_target /joy /fake_control_cmd /carla/ego_vehicle/odometry /fake_control_cmd /carla/ego_vehicle/vehicle_control_cmd -O "$dir/$time"
