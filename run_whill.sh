#!/usr/bin/env bash

xhost local:int_autoware

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/IntAutowareFiles:/home/int_autoware/AutowareFiles/ \
-v /media/kuriatsu/SamsungKURI/:/media/ssd/ \
--hostname="int_autoware" \
mad/whill:16.04-9.0.0
