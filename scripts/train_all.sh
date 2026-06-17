#!/bin/bash

#echo "start training count-time"
#for seed in $(seq 1 5); do
 #   CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/count-time/ppo.yaml --id $seed &
    #CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/count-time/rppo.yaml --id $seed &
    #CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/count-time/td3.yaml --id $seed &
    #CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/count-time/tqc.yaml --id $seed &
#done

#echo "start training count"
#for seed in $(seq 1 5); do
 #   CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/count/ppo.yaml --id $seed &
  #  nohup python train.py -f ../hyperpars/count/rppo.yaml --id $seed &
  #  CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/count/td3.yaml --id $seed &
   # CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/count/tqc.yaml --id $seed &
#done

#echo "start training size-time"
#for seed in $(seq 1 5); do
 #   CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/size-time/ppo.yaml --id $seed &
  #  nohup python train.py -f ../hyperpars/count-biomass-time/rppo.yaml --id $seed &
  #  CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/size-time/td3.yaml --id $seed &
   # CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/size-time/tqc.yaml --id $seed &
#done

echo "start training rppo"
for seed in $(seq 1 5); do
    CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/count/rppo.yaml --id $seed &
    CUDA_VISIBLE_DEVICES=0 nohup python train.py -f ../hyperpars/count-biomass-time/rppo.yaml --id $seed &
    CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/count-time/rppo.yaml --id $seed &
    CUDA_VISIBLE_DEVICES=1 nohup python train.py -f ../hyperpars/size-time/rppo.yaml --id $seed &
done