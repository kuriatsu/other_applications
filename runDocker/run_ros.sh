#!/usr/bin/env bash

xhost local:ros

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
--hostname="ros" \
mad/ros-kinetic:base
