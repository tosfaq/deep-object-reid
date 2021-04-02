#!/bin/bash

work_dir=$(realpath "$(dirname $0)")
exp_folder=${work_dir}/$1
test_file_path=${work_dir}/$1/combine_all.txt
for dir in ${exp_folder}/*     # list directories in the form "/tmp/dirname/"
do
    shopt -s extglob           # enable +(...) glob syntax
    result=${dir%%+(/)}    # trim however many trailing slashes exist
    result=${result##*/}       # remove everything before the last / that still remains
    printf '%s\n' "$result" >> $test_file_path
    cat ${dir}/train.log* | grep 'mAP:' >> $test_file_path
    cat ${dir}/train.log* | grep 'Rank-1  :' >> $test_file_path
    cat ${dir}/train.log* | grep 'Rank-5  :' >> $test_file_path
done
