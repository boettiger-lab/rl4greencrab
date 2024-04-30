#!/opt/venv/bin/python

# script args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("-pb", "--progress_bar", help="Use  progress bar for training", type=bool)
args = parser.parse_args()

# imports
import rl4greencrab
from rl4greencrab.utils import sb3_train

import os

# transform to absolute file path
abs_filepath = os.path.abspath(args.file)

# change directory to script's directory (since io uses relative paths)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# train
sb3_train(abs_filepath, progress_bar = args.progress_bar)