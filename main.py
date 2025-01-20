# main.py

from neura_nova import NeuraNova

def main():
    neura_nova = NeuraNova()
    train_loader, test_loader = neura_nova.getLoaders(dataset_path = '../data', batch_size = 64)

    # data_iter = iter(train_loader)
    # batch     = next(data_iter)
    # show_image(batch, 0)

    model = neura_nova.CreateFeedForwardNetwork()
    # model = Convolutional()

    trainer = neura_nova.InitTrainer(model, train_loader, test_loader, lr=0.001)

    for epoch in range(2):
        trainer.train()


if __name__ == '__main__':
    main()
