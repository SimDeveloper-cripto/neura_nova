import numpy as np

from ..data import load_mnist
from .layer import DenseLayer
from .network import FeedForward
from ..loss import SoftmaxCrossEntropy
from ..graphic_utils import show_results


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

def load_and_preprocess_data_for_ff(train_limit, test_limit):
    (X_train, y_train), (X_test, y_test) = load_mnist(train_limit, test_limit)

    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    return X_train, y_train_onehot, X_test, y_test_onehot

def create_kfold_indices(n_samples, n_folds, seed=42):
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_size = n_samples // n_folds
    remainder = n_samples % n_folds

    folds = []
    start = 0
    for fold in range(n_folds):
        extra = 1 if fold < remainder else 0  # add an extra element to some folds if n_samples is not divisible by n_folds
        end = start + fold_size + extra

        val_indices = indices[start:end]

        train_mask            = np.ones(n_samples, dtype=bool)
        train_mask[start:end] = False
        train_indices         = indices[train_mask]

        folds.append((train_indices, val_indices))
        start = end
    return folds


def build_and_train_ff_model_with_config(config, index, loss_fun=SoftmaxCrossEntropy()):
    """
    - input_dim     = 784
    - output_dim    = 10
    - weights shape = (output_dim, input_dim) = (# neurons, # features)
    """
    test_dimension       = config['test_dimension']
    train_dimension      = config['train_dimension']
    validation_dimension = config['validation_dimension']

    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_ff(train_dimension, test_dimension)

    n_folds = max(2, train_dimension // validation_dimension)
    folds   = create_kfold_indices(X_train.shape[0], n_folds)

    fold_results      = []
    fold_count        = 1
    best_val_accuracy = 0
    best_model        = None

    X_test_T        = X_test.T
    y_test_onehot_T = y_test_onehot.T
    test_accuracies = []

    for train_idx, val_idx in folds:
        print(f"\n[FOLD {fold_count}/{n_folds}]")

        # SPLIT DATA FOR THE FOLD
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train_onehot[train_idx]
        X_val_fold   = X_train[val_idx]
        y_val_fold   = y_train_onehot[val_idx]

        X_train_fold = X_train_fold.T
        y_train_fold = y_train_fold.T
        X_val_fold   = X_val_fold.T
        y_val_fold   = y_val_fold.T

        nn      = FeedForward(loss_fun)
        lr      = config['learning_rate']
        beta1   = config['beta1']
        beta2   = config['beta2']
        epsilon = config['epsilon']

        for layer_config in config['layers']:
            nn.add_layer(DenseLayer(input_dim=784 if len(nn.layers) == 0 else nn.layers[-1].weights.shape[0],
                                    output_dim=layer_config['neurons'],
                                    activation=layer_config['activation'],
                                    learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))
        epochs     = config['epochs']
        batch_size = config['batch_size']

        nn.train(X_train_fold, y_train_fold, epochs, X_val_fold, y_val_fold, batch_size)

        val_accuracy  = nn.getAccuracy(X_val_fold, y_val_fold, X_val_fold.shape[1])
        test_accuracy = nn.getAccuracy(X_test_T, y_test_onehot_T, test_dimension)
        test_accuracies.append(test_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = nn

        fold_results.append({
            'fold_number': fold_count,
            'validation_size': X_val_fold.shape[1],
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy
        })
        fold_count += 1

    mean_test_accuracy = np.mean(test_accuracies)
    std_test_accuracy  = np.std(test_accuracies)

    # Show results using the best model
    show_results(best_model, X_test_T, y_test_onehot_T, "ff", index)

    avg_val_accuracy = np.mean([fold['val_accuracy'] for fold in fold_results])
    result = {
        'fold_results': fold_results,
        'avg_val_accuracy': "{:.2f}".format(avg_val_accuracy * 100),
        'avg_test_accuracy': "{:.2f}".format(mean_test_accuracy * 100),
        'std_test_accuracy': "{:.2f}".format(std_test_accuracy * 100),
    }
    return result