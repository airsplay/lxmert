#!/usr/bin/env bash
DATAROOT="/netscratch/aabhat"
DATADIR="$DATAROOT/sample/results"
ROOTDIR="`dirname \"$0\"`"
ROOTDIR="`readlink -f ${ROOTDIR}`"
sudo userdocker run --rm -it -w $ROOTDIR -v /netscratch:/netscratch dlcc/pytorch:20.05 \
     bash run/get_activations.bash\