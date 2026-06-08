#!/bin/bash

echo "start training count-time"
nohup python train.py -f ../hyperpars/count-time/ppo.yaml &
nohup python train.py -f ../hyperpars/count-time/rppo.yaml &
nohup python train.py -f ../hyperpars/count-time/td3.yaml &
nohup python train.py -f ../hyperpars/count-time/tqc.yaml &

echo "start training count"
nohup python train.py -f ../hyperpars/count/ppo.yaml &
nohup python train.py -f ../hyperpars/count/rppo.yaml &
nohup python train.py -f ../hyperpars/count/td3.yaml &
nohup python train.py -f ../hyperpars/count/tqc.yaml &