# neura_nova/api.py

# TODO: USE GPU TO COMPUTE

from .config import load_config
from .graphic_utils import visualize_predictions, plot_metrics
from .ff.setup import build_and_train_ff_model_with_config, save_to_json
from .cnn.setup import load_and_preprocess_data_for_cnn, build_cnn_model

# TODO 1 [PROF]: MEDIA ARITMETICA OPPURE PRECISION PER ACCURACY
# TODO 2 [PROF]: BISOGNA CREARE L'OGGETTO NEURONE OPPURE LO SI PUO' ASTRARRE? DA UN PUNTO DI VISTA DI MEMORIA
# TODO 3 [PROF]: LA FUNZIONE D'ERRORE E I PARAMETRI PER ADAM SONO IPER-PARAMETRI? (ABBIAMO SOLO CROSS-ENTROPY)
# TODO 4 [PROF]: I FILTRI PER ORA NON CERCANO NESSUN PATTERN SPECIFICO MA IMPARANO COSA ESTRARRE CON LE EPOCHE
# TODO 5 [PROF]: PARLARE DEL COMMENTO PRESENTE IN conv_layer.py
# TODO 6 [PROF]: ABBIAMO PENSATO AD UN DROP-OUT PER AUMENTARE LA GENERALIZZAZIONE
# TODO 7 [PROF]: ABBIAMO PENSATO AD UN WEIGHT-DECAY (TERMINE DI PENALITA' NELLA LOSS) NELL'AGGIORNAMENTO DEI PESI

# TODO 1 [CARMINE]: NELLA CONFIG DELLA FEED-FORWARD AGGIUNGERE NUOVE CONFIG CON PIU' E MENO LAYER
# TODO 2 [CARMINE]: SUDDIVIDERE IL DATASET DI TRAINING, IN TRAINING + VALIDATION (MODIFICARE IL TRAIN) E VEDERE COME MIGLIORA L'APPRENDIMENTO
# TODO 3 [CARMINE]: SULLA BASE DI QUELLO FATTO NEL TODO 2, IMPLEMENTA EARLY-STOPPING (E' FACILE, SERVE A RIDURRE L'OVER-FITTING)
# TODO 4 [CARMINE]: STABILIRE UN MODO PER CONFIGURARE GLI IPER-PARAMETRI DELLA RETE CONVOLUZIONALE

def show_results(nn, X_test, y_test_onehot):
    plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"])
    visualize_predictions(nn, X_test, y_test_onehot)

def run_ff_model():
    # IPER-PARAMETRI [K-FOLD]
    # 0 --> LAYERS
    # 1 --> NEURONS PER LAYER
    # 2 --> ACTIVATION FUNCTION
    # 3 --> TRAINING SET AND TEST SET DIMENSIONS
    # 4 --> EPOCHS
    # 5 --> BATCH SIZE

    configs = load_config('config/ffconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[FEED-FORWARD] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_ff_model_with_config(config)
        results.append(result)
        index += 1

    save_to_json(results)

def run_cnn_model():
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_cnn(60000, 10000)
    nn = build_cnn_model()

    # TODO [SIMONE]
    # IPER-PARAMETRI [K-FOLD]
    # 0  --> NUMERO DI CONV. LAYER E POOL. LAYER
    # 1  --> NUMERO DI NEURONI PER LAYER
    # 2  --> FUNZIONE DI ATTIVAZIONE
    # 3  --> DIMENSIONE DEI DATASET DI ALLENAMENTO E DI TEST
    # 4  --> EPOCHE
    # 5  --> BATCH SIZE
    # 6  --> LEARNING RATE
    # 7  --> KERNEL SIZE
    # 8  --> PADDING
    # 9  --> STRIDE
    # 10 --> DIMENSIONE DELLA FINESTRA (QUANTI PIXEL PRENDE OGNI NEURONE)
    # 11 --> NUMERO DI FILTRI
    # 12 --> BETA 1
    # 13 --> BETA 2
    # 14 --> EPSILON

    epochs = 15
    batch_size = 128
    learning_rate = 0.001

    # Using ADAM update rule
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning and test accuracy
    train_accuracy = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy  = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)

    print("\n[CONVOLUTIONAL] TRAIN ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(train_accuracy * 100))
    print("[CONVOLUTIONAL] TEST ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(test_accuracy * 100))
