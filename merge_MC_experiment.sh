#!/bin/bash

# Prepare merging folder
mkdir dataset
cd dataset
mkdir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
cd ..

# go to dataset
cd output_100_1000_15h

for i in `seq 1 8`;
do
    echo $i
    cd output$i/dataset/
    
    for j in `seq 0 17`;
    do
        cd $j
        for k in `ls`;
        do
            n=$k
            n=_$n
            n=$i$n
            echo cp $k ../../../../dataset/$j/$n
            cp $k ../../../../dataset/$j/$n
        done
        cd ..
    done
    cd ../..
done
