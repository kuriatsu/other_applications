#! /bin/bash
source /home/kuriatsu/Source/catkin_ws/devel/setup.bash

roslaunch enjoy_carla camera_hud.launch town:=$1
