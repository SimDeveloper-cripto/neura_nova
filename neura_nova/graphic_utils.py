# neura_nova/graphic_utils.py

import os
import numpy as np
import matplotlib.pyplot as plt


# Written for CNNs
def visualize_predictions(nn, X_test, y_test_onehot, num_immagini=25, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    if y_test_onehot.shape[0] == 10 and y_test_onehot.shape[1] == X_test.shape[0]:
        y_test_onehot = y_test_onehot.T

    assert X_test.shape[0] == y_test_onehot.shape[0], f"Dimension mismatch: X_test {X_test.shape}, y_test_onehot {y_test_onehot.shape}"

    num_immagini       = min(num_immagini, X_test.shape[0])
    indici             = np.random.choice(X_test.shape[0], num_immagini, replace=False)
    immagini_campione  = X_test[indici].reshape(-1, 28, 28)
    etichette_campione = y_test_onehot[indici]

    logits     = nn.predict(X_test[indici])  # shape (10, batch_size)
    predizioni = np.argmax(logits, axis=0)
    truth      = np.argmax(etichette_campione, axis=1)

    assert len(predizioni) == num_immagini, f"predictions has dimension {len(predizioni)}, expected {num_immagini}"
    assert len(truth) == num_immagini, f"truth has dimension {len(truth)}, expected {num_immagini}"

    griglia_dim = int(np.ceil(np.sqrt(num_immagini)))
    plt.figure(figsize=(12, 12))

    for i in range(num_immagini):
        plt.subplot(griglia_dim, griglia_dim, i + 1)
        plt.imshow(immagini_campione[i], cmap='gray')

        if i < len(predizioni) and i < len(truth):
            plt.title(f"Pred: {predizioni[i]}\nTrue: {truth[i]}")
        else:
            plt.title("Errore indice!")

        plt.axis('off')

    plt.tight_layout()
    predictions_file = os.path.join(save_dir, "predictions.png")
    plt.savefig(predictions_file, bbox_inches='tight', dpi=150)
    plt.show()

def plot_metrics(title, history1, history2, metric_names1, metric_names2, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

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
    plt.savefig(os.path.join(save_dir, "metrics.png"))
    plt.close()

def show_results(nn, X_train, y_train_onehot, model, index, save_dir="results"):
    result = os.path.join(os.path.dirname(__file__), save_dir, model, index)
    os.makedirs(result, exist_ok=True)

    plot_metrics("RESULTS", nn.getTrainHistory(), nn.getValidationHistory(),
                 metric_names1=["train_loss", "train_accuracy"],
                 metric_names2=["validation_loss", "validation_accuracy"],
                 save_dir=str(result))

    # visualize_predictions(nn, X_train, y_train_onehot, save_dir=str(result))
    # visualize_predictions(nn, X_val, y_val, save_dir=str(result))