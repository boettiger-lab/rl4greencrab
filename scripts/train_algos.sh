#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

python train_gcse_manual.py -t 1000000 -a ppo -ne 10 &
python train_gcse_manual.py -t 1000000 -a rppo -ne 10 &
python train_gcse_manual.py -t 1000000 -a her -ne 10 &
python train_gcse_manual.py -t 1000000 -a tqc -ne 10 &
