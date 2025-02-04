# neura_nova/api.py

from .graphic_utils import visualize_predictions, plot_metrics
from .ff.setup import load_and_preprocess_data_for_ff, build_ff_model
from .cnn.setup import load_and_preprocess_data_for_cnn, build_cnn_model

# TODO [IMPORTANTE]: RISOLVERE LA QUESTIONE DEI NEURONI PER LA RETE CONVOLUZIONALE
# TODO [PROF]: MEDIA ARITMETICA OPPURE PRECISION PER ACCURACY
# TODO [PROF]: BISOGNA CREARE L'OGGETTO NEURONE OPPURE LO SI PUO' ASTRARRE? DA UN PUNTO DI VISTA DI MEMORIA
# TODO [PROF]: COSA BISOGNA METTERE A PARAGONE TRA LE DUE RETI?
# TODO [ALLA FINE]: USA GPU

# TODO [CARMINE]: STABILIRE UN MODO PER CONFIGURARE GLI IPER-PARAMETRI DELLE RETI


def build_and_train_ff_model():
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_ff(60000, 10000)
    nn = build_ff_model()

    # TODO [SIMONE]
    # IPER-PARAMETRI [K-FOLD]
    # 0 --> NUMERO DI LAYER
    # 1 --> NUMERO DI NEURONI PER LAYER
    # 2 --> FUNZIONE DI ATTIVAZIONE
    # 3 --> DIMENSIONE DEI DATASET DI ALLENAMENTO E DI TEST
    # 4 --> EPOCHE
    # 5 --> BATCH SIZE
    # 6 --> LEARNING RATE
    # 7 --> BETA 1
    # 8 --> BETA 2
    # 9 --> EPSILON
    # File .csv in results/ff

    epochs = 20
    batch_size = 128
    learning_rate = 0.001

    # Using ADAM update rule
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning and test accuracy
    train_accuracy = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy  = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)

    print("\n[FEED-FORWARD] TRAIN ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(train_accuracy * 100))
    print("[FEED-FORWARD] TEST ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(test_accuracy * 100))

    plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"])
    visualize_predictions(nn, X_test, y_test_onehot)

    # train_accuracy = nn.precision_accuracy(X_train, y_train_onehot, "----- TRAIN")
    # print("\n[INFO] TRAIN PRECISION_ACCURACY: ", train_accuracy)

    # test_accuracy = nn.precision_accuracy(X_test, y_test_onehot, "----- TEST")
    # print("\n[INFO] TEST PRECISION_ACCURACY: ", test_accuracy)


def build_and_train_cnn_model():
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

    plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"])
    visualize_predictions(nn, X_test, y_test_onehot)
