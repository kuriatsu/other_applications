#!/usr/bin/env bash

docker run -it -u kuriatsu --name dropbox_kuri \
-v /home/kuriatsu/DropboxKuri:/home/kuriatsu/Dropbox \
--hostname="dropbox" \
kuriatsu/dropbox:1.0
