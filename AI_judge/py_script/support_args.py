import argparse
import os

def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--da_num', type=int, default=2)
    parser.add_argument('--att_num', type=int, default=0)
    parser.add_argument('--model_num', type=int, default=2)

    parser.add_argument('--bat_num', type=int, default=25)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--tr_it', type=int, default=20)
    parser.add_argument('--te_it', type=int, default=20)
    parser.add_argument('--pr_it', type=int, default=10)

    return parser.parse_args(args)


