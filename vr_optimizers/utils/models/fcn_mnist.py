from torch import nn


class FCN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(FCN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.relu(self.linear_1(out))
        out = self.relu(self.linear_2(out))
        out = self.linear_3(out)
        return out


def fcn(pretrained=False, num_classes=10):
    return FCN(num_classes=num_classes)
