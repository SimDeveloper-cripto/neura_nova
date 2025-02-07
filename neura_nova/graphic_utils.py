# neura_nova/graphic_utils.py

import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(nn, X_test, y_test_onehot, num_immagini=25):
    """
    :param nn           : neural network
    :param X_test       : test set of images (already processed)
    :param y_test_onehot: test set one-hot encoding labels
    :param num_immagini : how many images to visualize
    """
    assert X_test.shape[1] == y_test_onehot.shape[1]
    num_immagini = min(num_immagini, X_test.shape[1])

    indici             = np.random.choice(X_test.shape[1], num_immagini, replace=False)
    immagini_campione  = X_test[:, indici]         # shape: (784, num_immagini)
    etichette_campione = y_test_onehot[:, indici]  # shape: (10,  num_immagini)

    # Predictions
    logits     = nn.predict(immagini_campione)
    predizioni = np.argmax(logits, axis=0)
    truth      = np.argmax(etichette_campione, axis=0)

    # Display images in a grid (each row contains 5 elements)
    griglia_dim = int(np.ceil(np.sqrt(num_immagini)))
    plt.figure(figsize=(12, 12))

    for i in range(num_immagini):
        plt.subplot(griglia_dim, griglia_dim, i + 1)
        immagine = immagini_campione[:, i].reshape(28, 28)
        plt.imshow(immagine, cmap='gray')
        plt.title(f"Pred: {predizioni[i]}\nTrue: {truth[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.ioff()
    plt.show()

def plot_metrics(title, history1, history2, metric_names1, metric_names2):
    epochs1 = range(1, len(history1[metric_names1[0]]) + 1)
    epochs2 = range(1, len(history2[metric_names2[0]]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for metric in metric_names1:
        ax1.plot(epochs1, history1[metric], label=metric)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{title} - Set 1")
    ax1.legend()
    ax1.grid(True)

    for metric in metric_names2:
        ax2.plot(epochs2, history2[metric], label=metric)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(f"{title} - Set 2")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.ioff()
    plt.show()

def show_results(nn, X_test, y_train_onehot, X_val, y_val):
    plot_metrics("RESULTS", nn.getTrainHistory(), nn.getValidationHistory(), metric_names1=["train_loss", "train_accuracy"], metric_names2=["val_loss", "val_accuracy"])
    visualize_predictions(nn, X_test, y_train_onehot)
    visualize_predictions(nn, X_val, y_val)
