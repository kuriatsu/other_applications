#!/usr/bin/env bash

xhost local:mad-whill

docker run -it --runtime=nvidia \
-e DISPLAY=$DISPLAY \
-u mad-whill \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Program/Docker/whill:/home/mad-whill/sharedFiles/ \
-v /media/kuriatsu/SamsungKURI/:/media/ssd/ \
-v /dev/serial/by-id/:/dev/serial/by-id/ \
-v /dev/input/js0/:/dev/input/js0/ \
--hostname="mad-whill" \
mad/whill:16.04-9.0.0 \
bash
