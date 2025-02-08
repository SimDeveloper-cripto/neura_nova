# neura_nova/api.py

# TODO: USE GPU TO COMPUTE

from .config import load_config
from .ff.setup import build_and_train_ff_model_with_config, save_ff_to_json
from .cnn.setup import build_and_train_cnn_model_with_config, save_cnn_to_json

# TODO 1 [PROF]: MEDIA ARITMETICA OPPURE PRECISION PER ACCURACY
# TODO 2 [PROF]: LA LOSS DEL VALIDATION SET DEVE ESSERE DIVERSA DA QUELLA PER IL TRAINING SET?
# TODO 3 [PROF]: LA FUNZIONE D'ERRORE E I PARAMETRI PER ADAM SONO IPER-PARAMETRI? (ABBIAMO SOLO CROSS-ENTROPY)
# TODO 4 [PROF]: BISOGNA CREARE L'OGGETTO NEURONE OPPURE LO SI PUO' ASTRARRE? DA UN PUNTO DI VISTA DI MEMORIA
# TODO 5 [PROF]: I FILTRI PER ORA NON CERCANO NESSUN PATTERN SPECIFICO MA IMPARANO COSA ESTRARRE CON LE EPOCHE
# TODO 6 [PROF]: PARLARE DEL COMMENTO PRESENTE IN conv_layer.py
# TODO 7 [PROF]: ABBIAMO PENSATO AD UN DROP-OUT PER AUMENTARE LA GENERALIZZAZIONE
# TODO 8 [PROF]: ABBIAMO PENSATO AD UN WEIGHT-DECAY (TERMINE DI PENALITA' NELLA LOSS) NELL'AGGIORNAMENTO DEI PESI
# TODO 9 [PROF]: EARLY STOPPING SOLO CONV LAYER?

# TODO 1 [SIMONE, DOPO IL PROF] 8 --> DIMENSIONE DELLA FINESTRA (QUANTI PIXEL PRENDE OGNI NEURONE)

def run_ff_model():
    configs = load_config('config/ffconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[FEED-FORWARD] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_ff_model_with_config(config)
        results.append(result)
        index += 1
    save_ff_to_json(results)

def run_cnn_model():
    configs = load_config('config/cnnconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[CONVOLUTIONAL] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_cnn_model_with_config(config)
        results.append(result)
        index += 1
    save_cnn_to_json(results)
