#!/bin/bash

mount_dir="$1"

backup_list=(
"/home/kuriatsu/Documents/brainIV $mount_dir/documents"
"/home/kuriatsu/Documents/lecture $mount_dir/documents"
"/home/kuriatsu/Documents/presentation/interaction $mount_dir/documents"
"/home/kuriatsu/Documents/presentation/kabuki $mount_dir/documents"
"/home/kuriatsu/Documents/presentation/other $mount_dir/documents"
"/home/kuriatsu/Documents/presentation/seminar $mount_dir/documents"
"/home/kuriatsu/Documents/report $mount_dir/documents"
"/home/kuriatsu/Documents/tex $mount_dir/documents"
"/home/kuriatsu/Pictures/material $mount_dir/picture/M1"
"/home/kuriatsu/Pictures/edited $mount_dir/picture/M1"
"/home/kuriatsu/Pictures/svg $mount_dir/picture/M1"
"/home/kuriatsu/Pictures/xcf $mount_dir/picture/M1"
"/home/kuriatsu/Pictures/graph $mount_dir/picture/M1"
"/home/kuriatsu/Videos/edited $mount_dir/video"
"/home/kuriatsu/Videos/kdenlive $mount_dir/video"
"/home/kuriatsu/Videos/material $mount_dir/video"
)

# len=${#backup_list[@]}
# i=0
# barmax=30
for item in "${backup_list[@]}" ; do
    # bar=""
    # # rsync -ru "$item"
    # for j in $(seq 1 $barmax) ; do
    #     if [ $j -eq $i ] ; then
    #         bar="$bar>"
    #     elif [ $j -le $i ] ; then
    #         bar="$bar-"
    #     else
    #         bar="$bar "
    #     fi
    # done
    # persentage=$(( $i * 30 / $len))
    # echo -en "local [$bar] $persentage% cloud\r"
    # echo $item
    # i=$(( $i + 1))
    rsync -ruv --exclude '.*' $item
done
fusermount -u "$mount_dir"

shutdown now
