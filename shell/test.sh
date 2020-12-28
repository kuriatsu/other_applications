#! /bin/bash
source devel/setup.bash
roslaunch carla_helper carla_helper.launch &
roslaunch teleop_carla teleop_carla.launch &
roslaunch ras_carla ras_carla.launch &
python src/carla_helper/script/camera.py --rolename ros_camera_back --res 800x600
