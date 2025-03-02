import os
import json
import random
from itertools import product

def create_ff_config_file(filename):
    # Iperparametri variabili: hidden layer, batch size, epochs, validation_set, functions
    hidden_layer   = [1, 2, 3]
    batch_sizes    = [64, 128]
    epoch          = [15, 20]
    validation_set = [20000, 30000]
    functions      = ['sigmoid', 'relu']

    one_layer_neurons = [256, 128]
    two_layer_neurons = [[512, 256], [256, 64]]

    # Iperparametri fissi
    train_dimension = 60000
    test_dimension  = 10000
    learning_rate   = 0.001
    beta1           = 0.9
    beta2           = 0.999
    epsilon         = 1e-8
    output_layer    = 10

    configurations = []
    for hl, bs, fn, ep, vs in product(hidden_layer, batch_sizes, functions, epoch, validation_set):
        config = {
            "batch_size": bs,
            "epochs": ep,
            "validation_dimension": vs,
            "train_dimension": train_dimension,
            "test_dimension": test_dimension,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "layers": []
        }

        if hl == 1:
            for neurons in one_layer_neurons:
                new_config = config.copy()
                new_config["layers"] = [
                    {"neurons": neurons, "activation": fn},
                    {"neurons": output_layer, "activation": "identity"}
                ]
                configurations.append(new_config)
        else:
            for neurons in two_layer_neurons:
                new_config = config.copy()
                new_config["layers"] = [
                    {"neurons": neurons[0], "activation": fn},
                    {"neurons": neurons[1], "activation": fn},
                    {"neurons": output_layer, "activation": "identity"}
                ]
                configurations.append(new_config)

    # Selezioniamo 10 configurazioni casuali
    selected_configurations = random.sample(configurations, min(10, len(configurations)))

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(selected_configurations, file, indent=4)