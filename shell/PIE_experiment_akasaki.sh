#! /usr/bin/bash
export QT_X11_NO_MITSHM=1
xinput map-to-output 9 DP-5

subject_name="kuriatsu"
output_file="/home/kuriatsu/share/PIE_result/$(date "+%y%m%d%H%M%S")/"
mkdir -p ${output_file}

clip_list=(
"/media/ssd/PIE_data/PIE_clips/set02/video_0001.mp4"
"/media/ssd/PIE_data/PIE_clips/set02/video_0002.mp4"
)

anno_list=(
"/media/ssd/PIE_data/annotations/set02/video_0001_annt.xml"
"/media/ssd/PIE_data/annotations/set02/video_0002_annt.xml"
)

attrib_list=(
"/media/ssd/PIE_data/annotations_attributes/set02/video_0001_attributes.xml"
"/media/ssd/PIE_data/annotations_attributes/set02/video_0002_attributes.xml"
)

for i in {0..1}; do
    log_file="${output_file}/${subject_name}_$(( $i+1 )).csv"
    python3 PIE_experiment.py --video ${clip_list[$i]} --anno ${anno_list[$i]} --attrib ${attrib_list[$i]} --log $log_file --window_position "3840x0"
done
