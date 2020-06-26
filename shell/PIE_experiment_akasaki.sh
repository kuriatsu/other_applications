#! /usr/bin/bash
export QT_X11_NO_MITSHM=1

output_file="/home/kuriatsu/share/PIE_result/$(date "+%y%m%d%H%M%S")/"
mkdir -p ${output_file}

clip_list=(
"/media/ssd/PIE_data/PIE_clips/set04/video_0001.mp4"
"/media/ssd/PIE_data/PIE_clips/set04/video_0002.mp4"
"/media/ssd/PIE_data/PIE_clips/set04/video_0003.mp4"
"/media/ssd/PIE_data/PIE_clips/set04/video_0004.mp4"
"/media/ssd/PIE_data/PIE_clips/set04/video_0005.mp4"
"/media/ssd/PIE_data/PIE_clips/set04/video_0006.mp4"
)

anno_list=(
"/media/ssd/PIE_data/annotations/set04/video_0001_annt.xml"
"/media/ssd/PIE_data/annotations/set04/video_0002_annt.xml"
"/media/ssd/PIE_data/annotations/set04/video_0003_annt.xml"
"/media/ssd/PIE_data/annotations/set04/video_0004_annt.xml"
"/media/ssd/PIE_data/annotations/set04/video_0005_annt.xml"
"/media/ssd/PIE_data/annotations/set04/video_0006_annt.xml"
)

attrib_list=(
"/media/ssd/PIE_data/annotations_attributes/set04/video_0001_attributes.xml"
"/media/ssd/PIE_data/annotations_attributes/set04/video_0002_attributes.xml"
"/media/ssd/PIE_data/annotations_attributes/set04/video_0003_attributes.xml"
"/media/ssd/PIE_data/annotations_attributes/set04/video_0004_attributes.xml"
"/media/ssd/PIE_data/annotations_attributes/set04/video_0005_attributes.xml"
"/media/ssd/PIE_data/annotations_attributes/set04/video_0006_attributes.xml"
)

cowsay -f kitty 'Prease input your name'
read subject_name
cowsay 'Prease input operation type'
read operation_type

cowsay -f meow 'Are you ready? [Enter]'
read

for i in {0..2}; do
    log_file="${output_file}/${subject_name}_$(( $i+1 ))_${operation_type}.csv"
    python3 /home/kuriatsu/share/git_other_app/python/PIE_experiment.py --video ${clip_list[$i]} --anno ${anno_list[$i]} --attrib ${attrib_list[$i]} --log $log_file --window_position "3840x0"
done

cowsay -f hiyoko 'Did yow answer the questions? [Enter]'
read
cowsay -f goat 'Prease input operation type'
read operation_type
cowsay 'Are you ready? [Enter]'
read

for i in {0..2}; do
    log_file="${output_file}/${subject_name}_$(( $i+1 ))_${operation_type}.csv"
    python3 /home/kuriatsu/share/git_other_app/python/PIE_experiment.py --video ${clip_list[$i]} --anno ${anno_list[$i]} --attrib ${attrib_list[$i]} --log $log_file --window_position "3840x0"
done

cowsay -f dragon 'Thanks a lot !!!!'
