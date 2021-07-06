from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out


def log_reg_mushrooms(pretrained=False, num_classes=1):
    return LogisticRegression(112, num_classes=num_classes)


def log_reg_w8a(pretrained=False, num_classes=1):
    return LogisticRegression(300, num_classes=num_classes)


def log_reg_ijcnn1(pretrained=False, num_classes=1):
    return LogisticRegression(22, num_classes=num_classes)


def log_reg_a9a(pretrained=False, num_classes=1):
    return LogisticRegression(123, num_classes=num_classes)


def log_reg_phishing(pretrained=False, num_classes=1):
    return LogisticRegression(68, num_classes=num_classes)
