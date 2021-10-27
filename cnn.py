from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    conv1 (channels = 32, kernel size= 3, stride = 1) -> Relu -> conv2 (channels = 64, kernel size= 3, stride = 1)
    -> Relu -> max pool (kernel size = 2x2) -> Dropout(.25) ->
    Linear (128) -> Relu -> Dropout(.5) -> -> Linear (10) -> LogSoftmax

    Need to reshape outputs from the last conv layer prior to feeding them into
    the linear layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 64 * 12 * 12)
        x = self.drop2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
