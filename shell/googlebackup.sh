#!/bin/bash

mount_dir="$1"

backup_dir_list=(
"/home/kuriatsu/Documents/presentation/other"
"/home/kuriatsu/Documents/presentation/kabuki"
"/home/kuriatsu/Documents/presentation/report"
"/home/kuriatsu/Documents/presentation/seminar"
"/home/kuriatsu/Documents/tex"
"/home/kuriatsu/Pictures/xcf"
"/home/kuriatsu/Videos/kdenlive"
"/home/kuriatsu/Videos/raw"
)

google_dir_list=(
"$mount_dir/study_docs/other"
"$mount_dir/study_docs/kabuki"
"$mount_dir/study_docs/report"
"$mount_dir/study_docs/seminar"
"$mount_dir/study_docs/tex"
"$mount_dir/study_picture/M1/xcf"
"$mount_dir/study_video/kdenlive"
"$mount_dir/study_video/raw"
)


# google-drive-ocamlfuse "$mount_dir"
for i in `seq 0 ${#backup_dir_list[@]}`; do

    if [[ i -eq ${#backup_dir_list[@]} ]]; then
        break
    fi

    for backup_file in `find "${backup_dir_list[i]}" -type f`; do

        backup_file_name="${backup_file##*/}"
        target_end_dir="${backup_dir_list[i]##*/}"
        target_file="${google_dir_list[i]}/${backup_file#*"$target_end_dir"/}"
        # target_file="$target_dir"
        # echo $backup_file
        # echo $target_file
        # echo "--------------------------"

        if [ -e $target_file ]; then
            if [ `date +%Y%m%d%H%M -r $backup_file` -eq `date +%Y%m%d%H%M -r $target_file` ]; then
                echo "$target_file stay"
            # else
            #     echo "$target_file update"
            fi
        # else
        #      echo "$target_file upload"
        fi

    done
done
# cp -vr /home/kuriatsu/Documents/presentation/other "$mount_dir"/study_docs
# cp -vr /home/kuriatsu/Documents/presentation/kabuki "$mount_dir"/study_docs
# cp -vr /home/kuriatsu/Documents/presentation/report "$mount_dir"/study_docs
# cp -vr /home/kuriatsu/Documents/presentation/seminar "$mount_dir"/study_docs
# cp -vr /home/kuriatsu/Documents/tex "$mount_dir"/study_docs
#
# cp -vr /home/kuriatsu/Pictures/xcf "$mount_dir"/study_picture/M1
# cp -vr /home/kuriatsu/Videos/kdenlive "$mount_dir"/study_video
#
# fusermount -u "$mount_dir"
