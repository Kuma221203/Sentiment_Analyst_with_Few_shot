import sys

sys.path.append(".")

import argparse
import yaml
import torch

from utils.dataloader import get_dataloader
from utils.getter import get_model
import time
import random

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_way", 
        default=2, 
        help="number class", 
        type=int
    )
    parser.add_argument(
        "--k_shot", 
        default=10, 
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
        "--epochs", 
        default=20, 
        help="number train epochs", 
        type=int
    )
    parser.add_argument(
        "--path_train", 
        default='data/train/', 
        help="path folder of train set", 
        type=str
    )
    parser.add_argument(
        "--config_path", 
        default="configs/protonet.yml", 
        help="config file path", 
        type=str
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="weights path to store model",
        type=str
    )
    parser.add_argument(
        "--using_pretrain",
        default= False,
        help="using pretrain",
        type=bool
    )
    parser.add_argument(
        "--save_weights",
        default= False,
        help="save weights model",
        type=bool
    )
    return parser.parse_args()

def train(args):
    # load model
    model_configs = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    model = get_model(model_configs)

    if(args.using_pretrain and args.weights_path):
        model.load_state_dict(torch.load(args.weights_path))

    #training
    start = time.time()
    model.train(args.epochs, args.n_way, args.k_shot, args.k_query, args.path_train)
    end = time.time()
    print('Time train', end-start)

    # model saving
    if(args.save_weights and args.weights_path):
        torch.save(model.state_dict(), args.weights_path)
        print(f"Model was saved in {args.weights_path}")

if __name__ == "__main__":
    args = get_parser()
    train(args)