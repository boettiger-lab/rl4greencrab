#!/bin/bash

echo "start training count-time"
for seed in $(seq 6 10); do
    CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count-time/ppo.yaml --id $seed &
    CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count-time/td3.yaml --id $seed &
    CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count-time/tqc.yaml --id $seed &
done

#echo "start training count"
#for seed in $(seq 6 10); do
 #   CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count/ppo.yaml --id $seed &
  #  CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count/td3.yaml --id $seed &
   # CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count/tqc.yaml --id $seed &
#done

#echo "start training size-time"
#for seed in $(seq 6 10); do
   # CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/size-time/ppo.yaml --id $seed &
   # CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/size-time/td3.yaml --id $seed &
 #   CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/size-time/tqc.yaml --id $seed &
#done

#echo "start training count-biomass-time"
#for seed in $(seq 6 10); do
 #   CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count-biomass-time/ppo.yaml --id $seed &
  #  CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count-biomass-time/td3.yaml --id $seed &
   # CUDA_VISIBLE_DEVICES="" nohup python train.py -f ../hyperpars/count-biomass-time/tqc.yaml --id $seed &
#done