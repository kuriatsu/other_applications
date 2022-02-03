#!/bin/bash
subjects=(ando aso hikosaka ichiki ienaga ikai isobe ito kato1 kato2 matsubara nakakuki nakatani negi okawa otake1 otake2 sumiya taga yamamoto yasuhara)

out_dir="/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment"
if [ -e ${out_dir}/result/summary_rm_wrong.csv ]; then
    rm ${out_dir}/result/summary_rm_wrong.csv
fi
touch ${out_dir}/result/summary_rm_wrong.csv

for subject in ${subjects[@]}; do
    echo $subject
    python3 carla_summarize.py $subject
    tail -n +2 ${out_dir}/${subject}/summary_rm_wrong.csv >> ${out_dir}/result/summary_rm_wrong.csv
done
