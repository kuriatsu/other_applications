#! /bin/bash

docker start dropbox_tkd dropbox_kuri
docker exec -t dropbox_tkd /usr/bin/dropbox start
docker exec -t dropbox_kuri /usr/bin/dropbox start
