import argparse
import os

def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--da_num', type=int, default=2)
    parser.add_argument('--model_num', type=int, default=4)
    parser.add_argument('--att_str', type=str, default='gender')

    if os.path.isdir('/share/home/hulianting/Project/Project12_unfairness/'):
        parser.add_argument('--bat_num', type=int, default=120)
        parser.add_argument('--epoch', type=int, default=150)
        parser.add_argument('--tr_it', type=int, default=2000)
        parser.add_argument('--te_it', type=int, default=1000000)
        parser.add_argument('--pr_it', type=int, default=100)
    elif os.path.isdir('/home/hulianting/Projects/Project12_unfairness/'):
        parser.add_argument('--bat_num', type=int, default=10)
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--tr_it', type=int, default=1000)
        parser.add_argument('--te_it', type=int, default=1000)
        parser.add_argument('--pr_it', type=int, default=10)
    else:
        parser.add_argument('--bat_num', type=int, default=5)
        parser.add_argument('--epoch', type=int, default=20)
        parser.add_argument('--tr_it', type=int, default=200)
        parser.add_argument('--te_it', type=int, default=20)
        parser.add_argument('--pr_it', type=int, default=10)

    return parser.parse_args(args)