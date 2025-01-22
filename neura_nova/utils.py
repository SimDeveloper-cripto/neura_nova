# neura_nova/utils.py

import matplotlib.pyplot as plt
import numpy as np

# TODO: TRANSLATE TO ENGLISH
def visualize_predictions(nn, X_test, y_test_onehot, num_immagini=16):
    """
    :param nn: neural network used
    :param X_test: images in test set, already processed
    :param y_test_onehot: labels of test's one-hot encoding
    :param num_immagini: how many images to visualize
    """

    indici             = np.random.choice(X_test.shape[0], num_immagini, replace=False)
    immagini_campione  = X_test[indici]
    etichette_campione = y_test_onehot[indici]

    # Predictions
    logits     = nn.predict(immagini_campione)
    predizioni = np.argmax(logits, axis=1)
    verita     = np.argmax(etichette_campione, axis=1)

    # Show images in a grid
    griglia_dim = int(np.sqrt(num_immagini))
    if griglia_dim * griglia_dim < num_immagini:
        griglia_dim += 1
    plt.figure(figsize=(12, 12))
    for i in range(num_immagini):
        plt.subplot(griglia_dim, griglia_dim, i + 1)
        immagine = immagini_campione[i].reshape(28, 28)
        plt.imshow(immagine, cmap='gray')
        plt.title(f"Pred: {predizioni[i]}\nTrue: {verita[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
