from model import *
from vocab import Dictionary
from ethicset import EthicSet

import torch

def getAccuracy(model, loader, device, short=False):
    correct = 0
    total = 0
    for i, (label, features) in enumerate(loader):
        label = label.to(device)
        features = features.to(device)
        with torch.no_grad():
            prediction = model(features)
            prediction = (prediction > .5).to(torch.int64).squeeze(1)
            correct += (prediction == label).sum().item()
            total += label.shape[0]
            if short and i == 500:
                return correct/total
    return correct/total

def train(model, trainloader, testloader, device, epochs, optim, crit):
    for epoch in range(epochs):
        total_loss = 0
        for i, (label, features) in enumerate(trainloader):
            label    = label.to(device).to(torch.float32)
            features = features.to(device)

            optim.zero_grad()

            pred = model(features).squeeze(1)
            loss = crit(pred, label)

            loss.backward()
            optim.step()

            total_loss += loss.item()

            if (i+1) % 250 == 0:
                print(f"Mid-Epoch[{epoch+1}] Loss: {total_loss/250}")
                total_loss = 0
            if (i+1) % 500 == 0:
                print(f"Train-Sample-Accuracy: {getAccuracy(model, trainloader, device, True)}")
                print(f"Test-Sample-Accuracy: {getAccuracy(model, testloader, device, True)}")


        print("=======================================")
        print(f"Full Train Accuracy: {getAccuracy(model, trainloader, device)}")
        print(f"Full Test Accuracy: {getAccuracy(model, testloader, device)}")
        torch.save(model, "Transformer3.pth")
        print("Model Saved")
        print("=======================================")

if __name__ == "__main__":
    dictionary = Dictionary("words.dict")
    trainset = EthicSet("data/train_moral.csv", dictionary, 100)
    testset = EthicSet("data/test_moral.csv", dictionary, 100)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                 shuffle=True, num_workers=0)

    device = torch.device('cuda')
    #model = SimpleModel(max_len=100, emb_dim=128, num_tokens=len(dictionary))
    model = Transformer(num_tokens=len(dictionary), emb_dim=128, max_seq=100, num_heads=16, num_hidden=4, num_layers=2, dropout=.4, device=device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr =3e-4)
    crit = torch.nn.BCEWithLogitsLoss()

    train(model, trainloader, testloader, device, 100, opt, crit)
    torch.save(model, "Transformer4.pth")


