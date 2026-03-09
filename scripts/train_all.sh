#!/bin/bash

echo "start training count-time"
nohup python train.py -f ../hyperpars/twoact/count-time/ppo.yml &
nohup python train.py -f ../hyperpars/twoact/count-time/rppo.yml &
nohup python train.py -f ../hyperpars/twoact/count-time/td3.yml &
nohup python train.py -f ../hyperpars/twoact/count-time/tqc.yml &

echo "start training count"
nohup python train.py -f ../hyperpars/twoact/count/ppo.yml &
nohup python train.py -f ../hyperpars/twoact/count/rppo.yml &
nohup python train.py -f ../hyperpars/twoact/count/td3.yml &
nohup python train.py -f ../hyperpars/twoact/count/tqc.yml &
