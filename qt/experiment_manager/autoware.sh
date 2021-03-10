#! /bin/bash
source /home/kuriatsu/Source/carla-autoware/catkin_ws/devel/setup.bash
source /home/kuriatsu/Source/autoware-1.13/install/setup.bash

roslaunch /home/kuriatsu/Source/carla-autoware/autoware_launch/autoware.launch town:=$1
