# neura_nova/config.py

import os
import json

def load_config(file_path):
    config_file = os.path.join(os.path.dirname(__file__), file_path)
    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs
