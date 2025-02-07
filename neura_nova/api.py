# neura_nova/api.py

# TODO: USE GPU TO COMPUTE

from .config import load_config
from .graphic_utils import visualize_predictions, plot_metrics
from .ff.setup import build_and_train_ff_model_with_config, save_to_json
from .cnn.setup import build_and_train_cnn_model_with_config, save_to_json

# TODO 1 [PROF]: MEDIA ARITMETICA OPPURE PRECISION PER ACCURACY
# TODO 2 [PROF]: BISOGNA CREARE L'OGGETTO NEURONE OPPURE LO SI PUO' ASTRARRE? DA UN PUNTO DI VISTA DI MEMORIA
# TODO 3 [PROF]: LA FUNZIONE D'ERRORE E I PARAMETRI PER ADAM SONO IPER-PARAMETRI? (ABBIAMO SOLO CROSS-ENTROPY)
# TODO 4 [PROF]: I FILTRI PER ORA NON CERCANO NESSUN PATTERN SPECIFICO MA IMPARANO COSA ESTRARRE CON LE EPOCHE
# TODO 5 [PROF]: PARLARE DEL COMMENTO PRESENTE IN conv_layer.py
# TODO 6 [PROF]: ABBIAMO PENSATO AD UN DROP-OUT PER AUMENTARE LA GENERALIZZAZIONE
# TODO 7 [PROF]: ABBIAMO PENSATO AD UN WEIGHT-DECAY (TERMINE DI PENALITA' NELLA LOSS) NELL'AGGIORNAMENTO DEI PESI
# TODO 8 [PROF]: PER LA LOSS DEL VALIDATION SET DEVE ESSERE DIVERSA DA QUELLA PER IL TRAINING SET?

# TODO 1 [CARMINE]: AGGIUGNERE ALLA CONFIG DELLA FF L'ATTRIBUTO "validation_dimension" AD OGNI MODELLO
# TODO 2 [CARMINE]: AGGIUGNERE ALLA CONFIG DELLA CNN ALTRI MODELLI
# TODO 3 [CARMINE]: IMPLEMENTA EARLY-STOPPING (PER RIDURRE L'OVER-FITTING)
# TODO 4 [CARMINE]: STABILIRE UN MODO PER CONFIGURARE GLI IPER-PARAMETRI DELLA RETE CONVOLUZIONALE

def show_results(nn, X_test, y_train_onehot, X_val, y_val):
    plot_metrics("RESULTS", nn.getTrainHistory(), nn.getValidationHistory(), metric_names1=["train_loss", "train_accuracy"], metric_names2=["val_loss", "val_accuracy"])
    visualize_predictions(nn, X_test, y_train_onehot)
    visualize_predictions(nn, X_val, y_val)

def run_ff_model():
    # IPER-PARAMETRI [K-FOLD]
    # 0 --> LAYERS
    # 1 --> NEURONS PER LAYER
    # 2 --> ACTIVATION FUNCTION
    # 3 --> DIMENSION OF TRAINING SET, VALIDATION SET AND TEST SET
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
    # IPER-PARAMETRI [K-FOLD]
    # 0  --> NUMERO DI CONV. LAYER, POOL. LAYER, FC LAYER
    # 1  --> FUNZIONE DI ATTIVAZIONE
    # 2  --> DIMENSIONE DEI DATASET DI ALLENAMENTO, VALIDAZIONE E DI TEST
    # 3  --> EPOCHE
    # 4  --> BATCH SIZE
    # 5  --> KERNEL SIZE
    # 6  --> PADDING
    # 7  --> STRIDE
    # TODO 1 [SIMONE, DOPO IL PROF] 8 --> DIMENSIONE DELLA FINESTRA (QUANTI PIXEL PRENDE OGNI NEURONE)

    configs = load_config('config/cnnconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[CONVOLUTIONAL] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_cnn_model_with_config(config)
        results.append(result)
        index += 1

    save_to_json(results)
