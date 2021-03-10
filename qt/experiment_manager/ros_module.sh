#! /bin/bash
source ~/Source/catkin_ws/devel/setup.bash

roslaunch enjoy_carla ros_module.launch town:=$1
