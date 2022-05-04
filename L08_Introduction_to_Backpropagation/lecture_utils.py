import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def alphaSmoothing(values, alpha):
    return (
        SimpleExpSmoothing(values)
        .fit(smoothing_level=alpha, optimized=False)
        .fittedvalues
    )


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


class Dataset:
    def __init__(self, batchSize=100):
        self.batchSize = batchSize
        self.train_data = datasets.MNIST(
            root="data", train=True, transform=ToTensor(), download=True,
        )
        self.test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                self.train_data, batch_size=self.batchSize, shuffle=True, num_workers=1
            ),
            "test": torch.utils.data.DataLoader(
                self.test_data, batch_size=self.batchSize, shuffle=True, num_workers=1
            ),
        }


class Training:
    def __init__(self, epochs=10, batchSize=100, learningRate=0.01):
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate

        self.dataset = Dataset(self.batchSize)
        self.net = CNN()
        self.net.cuda()
        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate)

    def train(self, maxIterations=None):
        print(
            f"Start training of {self.epochs} epochs with B = {self.batchSize}; lr = {self.learningRate}"
        )

        checkForMaxIter = False
        if maxIterations != None:
            checkForMaxIter = True

        self.net.train()

        epochLoss = []
        iterationLoss = []

        currIt = 0

        for epoch in tqdm(range(self.epochs)):

            itLoss = []
            for i, (images, labels) in enumerate(self.dataset.loaders["train"]):

                if checkForMaxIter:
                    if currIt > maxIterations:
                        print("reached max iterations")
                        return epochLoss, iterationLoss

                images = images.cuda()
                labels = labels.cuda()

                self.optimizer.zero_grad()
                # b_x = Variable(images)
                # b_y = Variable(labels)

                output = self.net(images)[0]
                loss = self.lossFunc(output, labels)

                # clear gradients for this training step

                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                self.optimizer.step()

                itLoss.append(loss.item())
                iterationLoss.append(loss.item())
                currIt += 1

            epochLoss.append(np.asarray(itLoss).mean())

        return epochLoss, iterationLoss
