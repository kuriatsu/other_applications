#!/usr/bin/env bash

xhost local:carla

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
--device /dev/snd \
-p 2000-2002:2000-2002 \
-u mad-carla \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/carla/:/home/mad-carla/share/ \
-v /media/kuriatsu/SamsungKURI1/:/media/ssd/ \
--network=carla-network \
--hostname="carla" \
mad/carla:16.04-0.9.6-1.11.0 \
bash
