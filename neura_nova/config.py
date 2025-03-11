# neura_nova/config.py

import os
import json

def load_config(file_path):
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, file_path)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs

# ff
def update_config_results_ff(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump([results], file, indent=4)

    with open(filename, 'r') as f:
        data = json.load(f)

    best_model_index = max(
        enumerate(data),
        key=lambda x: float(x[1].get("avg_test_accuracy", "0"))
    )[0]

    data.append({"best_model": best_model_index + 1})
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# cnn
def update_config_results_cnn(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump([results], file, indent=4)

    with open(filename, 'r') as f:
        data = json.load(f)

    models_data = data[0]

    best_model_index = max(
        enumerate(models_data),
        key=lambda x: float(x[1].get("avg_test_accuracy", "0"))
    )[0]

    data.append({"best_model": best_model_index + 1})
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump(data, file, indent=4)