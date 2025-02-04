# neura_nova/api.py

from .graphic_utils import visualize_predictions, plot_metrics
from .ff.setup import load_and_preprocess_data_for_ff, build_ff_model
from .cnn.setup import load_and_preprocess_data_for_cnn, build_cnn_model

# TODO [IMPORTANTE]: USA GPU
# TODO [IMPORTANTE]: OTTIMIZZARE RETE CONVOLUZIONALE
# TODO [IMPORTANTE]: OTTIMIZZARE RETE FEED-FORWARD

# TODO [PROF]: MEDIA ARITMETICA OPPURE PRECISION PER ACCURACY
# TODO [PROF]: BISOGNA CREARE L'OGGETTO NEURONE OPPURE LO SI PUO' ASTRARRE? DA UN PUNTO DI VISTA DI MEMORIA

# TODO: NEURONI PER RETE CONVOLUZIONALE


def build_and_train_ff_model():
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_ff()
    nn = build_ff_model()

    epochs = 15
    batch_size = 128
    learning_rate = 0.001

    # Using ADAM update rule
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning and test accuracy
    train_accuracy = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy  = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)

    print("\n[INFO] TRAIN ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(train_accuracy * 100))
    print("[INFO] TEST ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(test_accuracy * 100))

    plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"])
    visualize_predictions(nn, X_test, y_test_onehot)

    # train_accuracy = nn.precision_accuracy(X_train, y_train_onehot, "----- TRAIN")
    # print("\n[INFO] TRAIN PRECISION_ACCURACY: ", train_accuracy)

    # test_accuracy = nn.precision_accuracy(X_test, y_test_onehot, "----- TEST")
    # print("\n[INFO] TEST PRECISION_ACCURACY: ", test_accuracy)


def build_and_train_cnn_model():
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_cnn()
    nn = build_cnn_model()

    epochs = 15
    batch_size = 128
    learning_rate = 0.001

    # Using ADAM update rule
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning and test accuracy
    train_accuracy = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy  = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)

    print("\n[INFO] TRAIN ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(train_accuracy * 100))
    print("[INFO] TEST ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(test_accuracy * 100))

    plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"])
    visualize_predictions(nn, X_test, y_test_onehot)
