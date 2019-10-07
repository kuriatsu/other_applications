#!/usr/bin/env bash

xhost local:mad_carla

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-p 2000-2002:2000-2002 \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/CarlaFiles:/home/carla/CarlaFiles/ \
-v /media/kuriatsu/SamsungKURI/:/media/ssd/ \
--hostname="mad_carla" \
mad/ros-kinetic:carla
