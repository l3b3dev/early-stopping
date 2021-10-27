from functools import partial
import numpy as np
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import seaborn as sns

from cnn import CNN
from mlp import MLP


def load_data(data_dir="./data"):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.KMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.KMNIST(root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def train(config, checkpoint_dir=None, data_dir=None, num_epochs=200):
    #net = MLP(config["l1"], config["l2"], config["dr"])
    net = CNN()
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True, num_workers=8)

    for epoch in tqdm(range(1, num_epochs+1)):
        train_epoch_loss = 0
        train_epoch_acc = 0

        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            train_epoch_acc += torch.mean(equals.type(torch.FloatTensor))

        # Validation loss
        val_epoch_loss = 0
        val_epoch_acc = 0

        net.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)
                    val_epoch_acc += torch.mean(equals.type(torch.FloatTensor))

                    loss = criterion(outputs, labels)
                    val_epoch_loss += loss.item()

        loss_stats['train'].append(train_epoch_loss / len(trainloader))
        loss_stats['val'].append(val_epoch_loss / len(valloader))
        accuracy_stats['train'].append(train_epoch_acc.item() / len(trainloader))
        accuracy_stats['val'].append(val_epoch_acc.item() / len(valloader))
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)
        #
        # tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    torch.save(net.state_dict(), 'mdl.pth')

    print("Finished Training")

    return accuracy_stats, loss_stats


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False)

    val_epoch_acc = 0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            val_epoch_acc += torch.mean(equals.type(torch.FloatTensor))

    return val_epoch_acc.item() / len(testloader)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    # main(num_samples=10, max_num_epochs=200, gpus_per_trial=0)
    gpus_per_trial = 2
    data_dir = os.path.abspath("./data")
    config = {
        "l1": 128,
        "l2": 128,
        "lr": 0.001,  # Learning Rate
        "batch_size": 64,  # Batch Size
        "dr": 0.3,  # Dropout
        # "momentum": tune.uniform(0.1, 0.9)
    }
    accuracy_stats, loss_stats = train(config, data_dir=data_dir)

    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        'Train-Val Loss/Epoch')
    #plt.show()
    plt.savefig("./mlp-accuracy.png")

    model = CNN()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            model = nn.DataParallel(model)
    model.to(device)

    model.load_state_dict(torch.load('mdl.pth'))

    test_acc = test_accuracy(model, device)
    print("Best trial test set accuracy: {}".format(test_acc))