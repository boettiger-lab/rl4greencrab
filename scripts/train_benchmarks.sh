#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/6-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/7-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/8-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/9-tqc_nmem-1_bmk.yml &
