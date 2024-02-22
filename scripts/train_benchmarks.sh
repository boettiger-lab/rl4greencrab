#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/10-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/11-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/12-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/13-tqc_nmem-1_bmk.yml &
python train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/14-tqc_nmem-1_bmk.yml &