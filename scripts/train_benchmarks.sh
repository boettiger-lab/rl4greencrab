#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

python scripts/train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/1-tqc_nmem-1_bmk.yml &
python scripts/train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/2-tqc_nmem-1_bmk.yml &
python scripts/train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/3-tqc_nmem-1_bmk.yml &
python scripts/train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/4-tqc_nmem-1_bmk.yml &
python scripts/train.py --file ../hyperpars/systematic-benchmarks/TQC-Nmem_1/5-tqc_nmem-1_bmk.yml &