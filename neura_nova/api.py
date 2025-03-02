# neura_nova/api.py

import os
from .config import load_config, update_config_results

from neura_nova.ffhpscript import create_ff_config_file
from .ff.setup import build_and_train_ff_model_with_config

from neura_nova.cnnhpscript import create_cnn_config_file
from .cnn.setup import build_and_train_cnn_model_with_config

# TODO: __REGOLA DI DECISIONE__

# TODO [PROF]: ABBIAMO PENSATO AD UN DROP-OUT PER AUMENTARE LA GENERALIZZAZIONE
# TODO [PROF]: ABBIAMO PENSATO AD UN WEIGHT-DECAY (TERMINE DI PENALITA' NELLA LOSS) NELL'AGGIORNAMENTO DEI PESI

ff_file_config  = os.path.join(os.path.dirname(__file__), "config", "ffconfigurations.json")
cnn_file_config = os.path.join(os.path.dirname(__file__), "config", "cnnconfigurations.json")

def run_ff_model():
    create_ff_config_file(ff_file_config)
    configs = load_config(ff_file_config)
    results = []

    index = 1
    for config in configs:
        print(f"\n[FEED-FORWARD] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_ff_model_with_config(config)
        results.append(result)
        index += 1
    update_config_results(results, os.path.join(os.path.dirname(__file__), 'results', 'ff', 'results.json'))

def run_cnn_model():
    create_cnn_config_file(cnn_file_config)
    configs = load_config(cnn_file_config)
    results = []

    index = 1
    for config in configs:
        print(f"\n[CONVOLUTIONAL] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_cnn_model_with_config(config)
        results.append(result)
        index += 1
    update_config_results(results, os.path.join(os.path.dirname(__file__), 'results', 'cnn', 'results.json'))