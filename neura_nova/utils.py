# neura_nova/utils.py

import numpy as np
import matplotlib.pyplot as plt


async def visualize_predictions(nn, X_test, y_test_onehot, num_immagini=25):
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

async def plot_metrics(title, history, metric_names):
    epochs = range(1, len(history[metric_names[0]]) + 1)

    plt.figure(figsize=(12, 6))
    for metric in metric_names:
        plt.plot(epochs, history[metric], label=metric)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ioff()
    plt.show()
