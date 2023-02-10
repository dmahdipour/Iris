import torch


class irisModel(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Linear(4,5)
        self.rel=torch.nn.ReLU()
        self.layer2=torch.nn.Linear(5,3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.rel(x)
        x = self.layer2(x)
        return x

    