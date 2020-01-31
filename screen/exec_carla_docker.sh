#!/bin/bash
names=$(docker ps -f status=running --format "{{.Names}}" | sed -n 1P)
docker exec -it -u mad-carla "$names" bash
