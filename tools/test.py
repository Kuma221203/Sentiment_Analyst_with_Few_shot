import sys

sys.path.append(".")

import argparse
import yaml
import torch

from utils.dataloader import get_dataloader
from utils.getter import get_model
import time

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_way", default=2, help="number class", type=int)
    parser.add_argument("--n_shot", default=10, help="number support sample per class", type=int)
    parser.add_argument("--n_query", default=20, help="number query sample per class", type=int)
    parser.add_argument("--tasks", default=1000, help="number test task", type=int)
    parser.add_argument("--batch_size", default=16, help="batch size (if using maml or protoMAML)", type=int)
    parser.add_argument("--path_test", default='data/test/', help="path folder of test set", type = str)
    parser.add_argument(
        "--config_path", 
        default="configs/protonet.yml", 
        help="config file path",
        type = str
    )
    parser.add_argument(
        "--weights_path",
        default="weights/protoNet.pt",
        help="weights path to store model",
        type = str
    )
    return parser.parse_args()

def test(args):
    #load model
    model_configs = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    model = get_model(model_configs)
    model.load_state_dict(torch.load(args.weights_path))

    #dataloader
    if(model_configs['name'] == 'ProtoNet'):
        args.batch_size = args.tasks
    test_loader = get_dataloader(args.n_way, args.n_shot, args.n_query, args.tasks, args.path_test)

    start = time.time()
    model.test(test_loader)
    end = time.time()

    print('Time test', end-start)

if __name__ == "__main__":
    args = get_parser()
    test(args)