# neura_nova/api.py

import os
from .config import load_config, update_config_results_ff, update_config_results_cnn

from .ff.setup import build_and_train_ff_model_with_config
from .cnn.setup import build_and_train_cnn_model_with_config

# from neura_nova.ffhpscript import create_ff_config_file
# from neura_nova.cnnhpscript import create_cnn_config_file

ff_file_config  = os.path.join(os.path.dirname(__file__), "config", "ffconfigurations.json")
cnn_file_config = os.path.join(os.path.dirname(__file__), "config", "cnnconfigurations.json")

cnn_file_config2 = os.path.join(os.path.dirname(__file__), "config", "cnnconfigurations2.json")


def run_ff_model():
    # create_ff_config_file(ff_file_config)

    configs = load_config(ff_file_config)
    results = []
    index   = 1
    for config in configs:
        print(f"\n[FEED-FORWARD] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_ff_model_with_config(config, str(index))
        results.append(result)
        index += 1
    update_config_results_ff(results, os.path.join(os.path.dirname(__file__), 'results', 'ff', 'results.json'))

def run_cnn_model():
    # create_cnn_config_file(cnn_file_config)

    configs = load_config(cnn_file_config2)
    results = []
    index   = 11
    for config in configs:
        print(f"\n[CONVOLUTIONAL] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_cnn_model_with_config(config, str(index))
        results.append(result)
        index += 1
    update_config_results_cnn(results, os.path.join(os.path.dirname(__file__), 'results', 'cnn', 'results2.json'))