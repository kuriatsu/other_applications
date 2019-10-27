#!/bin/bash

mount_dir="$1"
google-drive-ocamlfuse "$mount_dir"

cp -r /home/kuriatsu/Documents/presentation/other "$mount_dir"/study_docs
cp -r /home/kuriatsu/Documents/presentation/kabuki "$mount_dir"/study_docs
cp -r /home/kuriatsu/Documents/presentation/report "$mount_dir"/study_docs
cp -r /home/kuriatsu/Documents/presentation/seminar "$mount_dir"/study_docs
cp -r /home/kuriatsu/Documents/presentation/tex "$mount_dir"/study_docs

cp -r /home/kuriatsu/Pictures/xcf "$mount_dir"/study_picture/M1
cp -r /home/kuriatsu/Videos/kdenlive "$mount_dir"/study_video

fusermount -u "$mount_dir"
