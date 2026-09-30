#!/bin/bash
# usage: ./run_jobs.sh jobs.txt [parallel] [extra args...]
f=$1; P=${2:-6}; shift 2
extra="$*"
cd "$(dirname "$0")"
grep -v '^\s*$' "$f" | xargs -P "$P" -I{} sh -c 'tag=$(echo "{}" | sed -E "s/.*--tag ([^ ]+).*/\1/"); python train_ppo.py {} '"$extra"' > results/logs/$tag.log 2>&1'
