import os
import json
import random
from itertools import product

def create_cnn_config_file(filename):
    # Iperparametri variabili: hidden layer, batch size, epochs, validation_set, functions
    hidden_layer        = [1, 2]
    batch_sizes         = [64, 128]
    epoch               = [15, 20]
    validation_set      = [20000, 30000]
    functions           = ['sigmoid', 'relu']
    filters             = [8, 12, 16]
    double_filters      = [[8, 12], [12, 16], [8, 16]]
    hidden_layer_neuron = 64

    # Iperparametri fissi
    train_dimension = 60000
    test_dimension  = 10000
    learning_rate   = 0.001
    beta1           = 0.9
    beta2           = 0.999
    epsilon         = 1e-8
    output_layer    = 10
    kernel_size     = 3
    stride          = 1

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
            "conv_layers": [],
            "max_pool_layers": [],
            "fc_layers": []
        }

        if hl == 1:
            for filter in filters:
                new_config = config.copy()
                new_config["conv_layers"] = [
                    {
                         "filters": filter,
                         "kernel_size": kernel_size,
                         "stride": stride,
                         "activation": fn
                    }
                ]
                new_config["max_pool_layers"] = [
                    {
                        "kernel_size": 2,
                        "stride": 2
                    }
                ]
                new_config["fc_layers"] = [
                    {
                        "neurons": output_layer,
                        "activation": "identity"
                    }
                ]
                configurations.append(new_config)
        else:
            for filter in double_filters:
                new_config = config.copy()
                new_config["conv_layers"] = [
                    {
                        "filters": filter[0],
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "activation": fn
                    },
                    {
                        "filters": filter[1],
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "activation": fn
                    }
                ]
                new_config["max_pool_layers"] = [
                    {
                        "kernel_size": 2,
                        "stride": 2
                    },
                    {
                        "kernel_size": 2,
                        "stride": 2
                    }
                ]
                new_config["fc_layers"] = [
                    {
                        "neurons": hidden_layer_neuron,
                        "activation": fn
                    },
                    {
                        "neurons": output_layer,
                        "activation": "identity"
                    }
                ]
                configurations.append(new_config)

    # Selezioniamo 10 configurazioni casuali
    selected_configurations = random.sample(configurations, min(10, len(configurations)))

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(selected_configurations, file, indent=4)