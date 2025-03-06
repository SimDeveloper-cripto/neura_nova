# neura_nova/config.py

import os
import math
import json

def load_config(file_path):
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, file_path)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs

def update_config_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump([results], file, indent=4)

    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    experiment_results = data[0]
    accuracies = []

    for exp in experiment_results:
        try:
            acc = float(exp.get("test_accuracy", "0"))
            accuracies.append(acc)
        except ValueError:
            continue

    if accuracies:
        mean_accuracy = sum(accuracies) / len(accuracies)
        mean_accuracy = round(mean_accuracy, 2)

        std_accuracy = math.sqrt(sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies))
        std_accuracy = round(std_accuracy, 2)
    else:
        mean_accuracy = 0.0
        std_accuracy  = 0.0

    experiment_results.append({
        "ar_mean": f"{mean_accuracy:.2f}",
        "std_deviation": f"{std_accuracy:.2f}"
    })

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)