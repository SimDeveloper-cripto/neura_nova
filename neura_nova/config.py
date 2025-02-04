# neura_nova/config.py

import os
import json

def load_ff_config():
    config_file = os.path.join(os.path.dirname(__file__), 'ffconfig.json')
    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs
