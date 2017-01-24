#!/bin/bash

# Prepare merging folder
mkdir dataset
cd dataset
mkdir 0 3 4
cd ..

# go to dataset
cd outputs_12h_1000_100

for i in `seq 1 8`;
do
    echo $i
    cd output$i/dataset/
    
    for j in 0 3 4;
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
