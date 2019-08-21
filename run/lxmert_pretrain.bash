# The name of experiment
name=lxmert

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain_new.py \
    --taskMaskLM --taskObjPredict --taskMatched --taskQA \
    --visualLosses obj,attr,feat \
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --fromScratch \
    --batchSize 256 --optim bert --lr 1e-4 --epochs 12 \
    --tqdm --output $output ${@:2}

