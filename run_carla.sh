#!/usr/bin/env bash

xhost local:mad-carla

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-p 2000-2002:2000-2002 \
-u mad-carla \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/carla/:/home/mad-carla/share/ \
--network=carla-network \
--hostname="mad-carla" \
mad/carla:16.04-0.9.6 \
bash
