#!/usr/bin/env bash

xhost local:pie

docker run -it -u kuriatsu --gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v /run/udev/:/run/udev/ \
-v /home/kuriatsu/Docker/shareFiles/pie:/home/kuriatsu/share \
-v /media/kuriatsu/SamsungKURI:/media/ssd \
--hostname="pie" \
kuriatsu/pie \
bash
