import json
import random
from itertools import product

if __name__ == "__main__":
    # iperparametri variabili: hidden layer, batch size, epoch, validation set e functions
    hidden_layer = [1, 2]
    batch_sizes = [64, 128]
    epoch = [15, 20]
    validation_set = [20000, 30000]
    functions = ['sigmoid', 'relu']

    one_layer_neurons = [128, 64]
    two_layer_neurons = [[512, 256], [256, 64]]

    # iperparametri fissi
    train_dimension = 60000
    test_dimension = 10000
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    output_layer = 10

    # Creiamo tutte le combinazioni possibili per gli iperparametri principali
    configurations = []

    for hl, bs, fn, ep, vs in product(hidden_layer, batch_sizes, functions, epoch, validation_set):
        # Creiamo la configurazione base con i parametri comuni
        config = {
            "batch_size": bs,
            "epoch": ep,
            "validation_set": vs,
            "train_dimension": train_dimension,
            "test_dimension": test_dimension,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "layers": []
        }

        # Aggiungiamo i layer in base al numero di hidden layers
        if hl == 1:
            for neurons in one_layer_neurons:
                new_config = config.copy()  # Creiamo una copia per ogni variazione
                new_config["layers"] = [
                    {"neurons": neurons, "activation": fn},
                    {"neurons": output_layer, "activation": "identity"}
                ]
                configurations.append(new_config)

        else:  # Se hidden_layer è 2, usiamo le coppie di two_layer_neurons
            for neurons in two_layer_neurons:
                new_config = config.copy()  # Creiamo una copia per ogni variazione
                new_config["layers"] = [
                    {"neurons": neurons[0], "activation": fn},
                    {"neurons": neurons[1], "activation": fn},
                    {"neurons": output_layer, "activation": "identity"}
                ]
                configurations.append(new_config)

    # Selezioniamo 10 configurazioni casuali (se ce ne sono almeno 10)
    selected_configurations = random.sample(configurations, min(10, len(configurations)))

    # Stampa le configurazioni selezionate
    for config in selected_configurations:
        print(json.dumps(config, indent=4))

    # Scriviamo il file JSON solo con le 10 configurazioni casuali
    with open("ffconfigurations.json", "w") as file:
        json.dump(selected_configurations, file, indent=4)

    print("10 configurazioni casuali salvate in ffconfigurations.json")
