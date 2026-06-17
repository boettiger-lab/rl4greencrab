#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("--id", help="Override id in config file", type=str)
args = parser.parse_args()

import rl4greencrab
from rl4greencrab.utils.sb3 import sb3_train

kwargs = {}
if args.id:
    kwargs["id"] = args.id

sb3_train(args.file, **kwargs)
