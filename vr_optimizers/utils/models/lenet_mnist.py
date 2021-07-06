from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc_1 = nn.Linear(256, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.pool(self.relu(self.conv_1(x)))
        out = self.pool(self.relu(self.conv_2(out)))
        out = self.flatten(out)
        out = self.relu(self.fc_1(out))
        out = self.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


def lenet(pretrained=False, num_classes=10):
    return LeNet(num_classes=num_classes)
