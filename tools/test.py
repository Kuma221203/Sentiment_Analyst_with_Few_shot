import sys

sys.path.append(".")

import argparse
import yaml
import torch

from utils.getter import get_model
import time

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_way", 
        default=3, 
        help="number class", 
        type=int
    )
    parser.add_argument(
        "--k_shot", 
        default=5, 
        help="number support sample per class", 
        type=int
    )
    parser.add_argument(
        "--k_query", 
        default=20, 
        help="number query sample per class", 
        type=int
    )
    parser.add_argument(
        "--path_test", 
        default='data/test/', 
        help="path folder of test set", 
        type = str
    )
    parser.add_argument(
        "--config_path", 
        default="configs/protonet.yml", 
        help="config file path",
        type = str
    )
    parser.add_argument(
        "--weights_path",
        default="weights/protonet.pt",
        help="weights path to store model",
        type = str
    )
    return parser.parse_args()

def test(args):
    #load model
    model_configs = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    model = get_model(model_configs)
    model.load_state_dict(torch.load(args.weights_path))

    #training
    start = time.time()
    y_true, y_predict = model.test(args.n_way, args.k_shot, args.k_query, args.path_test)
    end = time.time()
    print('Time test', end-start)

if __name__ == "__main__":
    args = get_parser()
    test(args)