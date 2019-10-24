#!/usr/bin/env bash

xhost local:ros

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /home/kuriatsu/Downloads/Installer/:/home/Downloads/ \
-v /run/udev/:/run/udev/ \
--hostname="ros" \
mad/ros-kinetic:base
