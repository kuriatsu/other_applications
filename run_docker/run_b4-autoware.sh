#!/usr/bin/env bash

xhost local:b4_autoware

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/B4AutowareFiles:/root/AutowareFiles/ \
-v /media/kuriatsu/SamsungKURI/:/media/ssd/ \
--hostname="b4_autoware" \
mad/ros-kinetic:B4autoware
