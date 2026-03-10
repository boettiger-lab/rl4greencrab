#!/bin/bash

echo "start training count-time"
nohup python train.py -f ../hyperpars/twoact/count-time/ppo.yaml &
nohup python train.py -f ../hyperpars/twoact/count-time/rppo.yaml &
nohup python train.py -f ../hyperpars/twoact/count-time/td3.yaml &
nohup python train.py -f ../hyperpars/twoact/count-time/tqc.yaml &

echo "start training count"
nohup python train.py -f ../hyperpars/twoact/count/ppo.yaml &
nohup python train.py -f ../hyperpars/twoact/count/rppo.yaml &
nohup python train.py -f ../hyperpars/twoact/count/td3.yaml &
nohup python train.py -f ../hyperpars/twoact/count/tqc.yaml &