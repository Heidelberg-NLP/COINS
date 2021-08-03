import sys
import os
import argparse

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", type=str, default='gen',
                    choices=["gen"])
parser.add_argument("--experiment_num", type=str, default="0")

args = parser.parse_args()

if args.experiment_type == "gen":
    from main_conceptnet import main
    main(args.experiment_num)
