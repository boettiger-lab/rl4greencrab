#!/bin/bash

nohup python train.py -f ../hyperpars/twoact/ppo-twoactnorm.yml &
nohup python train.py -f ../hyperpars/twoact/rppo-twoactnorm.yml &
nohup python train.py -f ../hyperpars/twoact/td3-twoactnorm.yml &
nohup python train.py -f ../hyperpars/twoact/tqc-twoactnorm.yml &
nohup python train.py -f ../hyperpars/twoact/td3-twoactnorm.yml &

nohup python train.py -f ../hyperpars/twoact/ppo-twoactnorm-def.yml &
nohup python train.py -f ../hyperpars/twoact/rppo-twoactnorm-def.yml &
nohup python train.py -f ../hyperpars/twoact/td3-twoactnorm-def.yml &
nohup python train.py -f ../hyperpars/twoact/tqc-twoactnorm-def.yml &
nohup python train.py -f ../hyperpars/twoact/td3-twoactnorm-def.yml &
