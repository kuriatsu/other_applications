#!/usr/bin/env bash

docker run -it -u kuriatsu --name dropbox_tkd \
-v /home/kuriatsu/DropboxTkd:/home/kuriatsu/Dropbox \
--hostname="dropbox" \
kuriatsu/dropbox:1.0
