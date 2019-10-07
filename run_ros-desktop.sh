#!/usr/bin/env bash

xhost local:ros

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/IntAutowareFiles:/home/AutowareFiles/ \
-v /media/kuriatsu/SamsungKURI/:/media/ssd/ \
--hostname="ros" \
mad/ros-kinetic:desktop
