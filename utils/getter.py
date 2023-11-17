import sys
sys.path.append('.')
from models.protonet import *
from models.maml import *
from models.protomaml import *

def get_model(config, **kwargs):
    assert "name" in config
    config.setdefault("args", {})
    if config["args"] is None:
        config["args"] = {}
    return globals()[config["name"]](config["args"], **kwargs)