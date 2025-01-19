# main.py

from neura_nova import get_loaders
from neura_nova.trainer import Trainer
# from neura_nova.utils import show_image
from neura_nova.models import FeedForward


def main():
    train_loader, test_loader = get_loaders(dataset_path = '../data', batch_size = 64)

    # data_iter = iter(train_loader)
    # batch     = next(data_iter)
    # show_image(batch, 0)

    model = FeedForward()
    # model = Convolutional()

    trainer = Trainer(model, train_loader, test_loader, lr=0.001)

    for epoch in range(2):
        trainer.train()


if __name__ == '__main__':
    main()
