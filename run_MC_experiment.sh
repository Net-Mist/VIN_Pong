#!/bin/bash

cp atari_uct/build/atari_uct /tmp

for i in `seq 1 8`;
do
    echo $i
    
    cd /tmp
    mkdir output$i
    cd output$i
    mkdir frames dataset
    cd dataset
    mkdir 0 3 4
    cd ../..
    ./atari_uct -rom_path=/home/mist/.local/lib/python3.6/site-packages/atari_py/atari_roms/pong.bin -save_data=true -depth=100 -num_traj=1000 -save_path=output$i&   
    sleep 10
done


