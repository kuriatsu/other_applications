#!/bin/bash
# exception handring if no arguments are given
if [ $# -lt 1 ]; then
    echo "Enter file path at least 1" 1>&2
    exit 1
fi

# create dirs for unzip (buffar_dir) and output (output_dir)
input_dir="$1"
output_dir="$input_dir/out"; mkdir -p $output_dir
buffar_dir="$input_dir/buf_files"; mkdir -p $buffar_dir

# do while reading student directories
while read student_dir; do
    # extract student number
    student_num=${student_dir%%(*}
    student_num=${student_num##*/}

    # create dirs for unzip (buffar_dir_student) and output (output_dir_student)
    output_dir_student="$output_dir/$student_num"; mkdir -p $output_dir_student
    output_dir_student_picture="$output_dir/$student_num/picture"; mkdir -p $output_dir_student_picture
    buffar_dir_student="$buffar_dir/$student_num"; mkdir -p $buffar_dir_student

    # start
    echo "extracting_student_number="$student_num

    # unzip and extract target files
    find "$student_dir" -mindepth 1 -name "*.zip" -print0 | xargs -0 -n 1 -I {} unzip -q {} -d "$buffar_dir_student/"
    find "$buffar_dir_student" -mindepth 1 -name "*.c" -print0 | xargs -0 -n 1 -I {} mv -f {} "$output_dir_student/"
    find "$buffar_dir_student" -mindepth 1 -name "*.h" -print0 | xargs -0 -n 1 -I {} mv -f {} "$output_dir_student/"
    find "$buffar_dir_student" -mindepth 1 -name "*.png" -print0 | xargs -0 -n 1 -I {} mv -f {} "$output_dir_student/"

    # extract target files for fuckin' stupid student who did not compress files
    find "$student_dir" -mindepth 1 -name "*.c" -print0 | xargs -0 -n 1 -I {} cp -f {} "$output_dir_student/"
    find "$student_dir" -mindepth 1 -name "*.h" -print0 | xargs -0 -n 1 -I {} cp -f {} "$output_dir_student/"
    find "$student_dir" -mindepth 1 -name "*.png" -print0 | xargs -0 -n 1 -I {} cp -f {} "$output_dir_student_picture/"

    # end
    echo "completed"

done < <(find "$input_dir" -maxdepth 1 -mindepth 1 -name "0*")

# cleaning
rm -r "$buffar_dir"
echo "Finished. Output directory is "$output_dir
