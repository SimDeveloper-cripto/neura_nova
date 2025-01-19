# import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# TODO: CREA LIBRERIA E SUDDIVIDI GIA' IN MODULI
# TODO: CREA STRUTTURA PER LO SVILUPPO DI UNA RETE
# TODO: PROVARE A STAMPARE UN'IMMAGINE

# DEBUG
def showImage(train_l, index):
    # Mostra un'immagine
    data_iter = iter(train_l)
    images, labels = next(data_iter)

    img = images[index]
    mean = 0.1307
    std = 0.3081

    # Riportiamo i pixel ai valori originali
    img = img * std + mean  # De-normalizza

    # Rimuovi la dimensione del canale per matplotlib (da (1, 28, 28) a (28, 28))
    img = img.squeeze()

    # Visualizza l'immagine
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {labels[0].item()}')
    plt.axis('off')  # Rimuove gli assi
    plt.show()


# Conversione delle immagini in tensori
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # TODO: DEFINIRE mean e std (deviazione standard)
])

data_path = './data'

# Carica il dataset MNIST per il training e il test
train_dataset = datasets.MNIST(
    root=data_path,
    train=True,          # Dataset di training
    download=True,       # Scarica il dataset se non già presente
    transform=transform  # Applica le trasformazioni definite
)

# Dopo la trasformazione, ogni immagine ha dimensioni (1, 28, 28)
# Il valore 1 indica la scala di grigi (ho reso le immagini in bianco e nero)

test_dataset = datasets.MNIST(
    root=data_path,
    train=False,         # Dataset di test
    download=True,
    transform=transform
)

# Crea dei DataLoader per gestire il batching e lo shuffling (per mescolare l'ordine dei campioni)
# Per evitare un adattamento troppo evidente alle sequenze fisse dei dati
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

showImage(train_loader, 0)

# Batch Learning
# Esempio di iterazione: stampa le forme di un batch di dati dal train_loader
for batch_idx, (data, target) in enumerate(train_loader):
    # Facciamo solo un primo batch per capire se lavoriamo sui dati corretti
    print("Batch index:", batch_idx)
    print("Data shape:", data.shape)      # Dovrebbe essere (64, 1, 28, 28)
    print("Target shape:", target.shape)  # Dovrebbe essere (64,)
    break