#!/opt/venv/bin/python
# Train PPO fully on the GPU:  python scripts/train_gpu.py -f hyperpars/count-time/ppo.yaml
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("--id", help="Override id in config file", type=str)
parser.add_argument("--n-envs", help="Number of parallel GPU envs", type=int)
args = parser.parse_args()

from rl4greencrab.utils.gpu import gpu_train

kwargs = {}
if args.id:
    kwargs["id"] = args.id
if args.n_envs:
    kwargs["gpu_n_envs"] = args.n_envs

gpu_train(args.file, **kwargs)
