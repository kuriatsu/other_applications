#!/bin/bash
# exception handring if no arguments are given
if [ $# -lt 1 ]; then
    echo "Enter file path at least 1" 1>&2
    exit 1
fi

# create dirs for unzip (buffar_dir) and output (output_dir)
input_dir="$1"
output_dir="$input_dir/out"; mkdir -p $output_dir

# do while reading student directories
while read student_file; do
    # extract student number
    # ${変数名#パターン}  # 前方からの最短マッチを削除
    # ${変数名##パターン} # 前方からの最長マッチを削除
    # ${変数名%パターン}  # 後方からの最短マッチを削除
    # ${変数名%%パターン} # 後方からの最長マッチを削除
    student_num=${student_file##*081}
    student_num=081${student_num%.*}
	extension=${student_file##*.}

   	echo "student_num="$student_num

	if [[ $student_num =~ [0-9]{9} ]]; then
		if [ ${#extension} -gt 3 ]; then
	    	echo "no extension="$student_num
            mv "$student_file" "$output_dir/$student_num"
	    else
	    	echo "change_file="$student_num.$extension
            mv "$student_file" "$output_dir/$student_num.$extension"
	    fi
	else
		echo "no_student_num="$student_file
        mv "$student_file" "$output_dir/"

        # student_id=${student_file##__*}
        # student_id=${student_id%.*}
        #
        # if [ ${#extension} -gt 3 ]; then
	    # 	echo "no extension="$student_id
        #     # mv "$student_file" "$output_dir/$student_id"
	    # else
	    # 	echo "change_file="$student_id.$extension
        #     # mv "$student_file" "$output_dir/$student_id.$extension"
	    # fi
	fi

    # start
    # end
    #echo "completed"

done < <(find "$input_dir" -maxdepth 1 -mindepth 1)

# cleaning
echo "Finished. Output directory is "$output_dir
