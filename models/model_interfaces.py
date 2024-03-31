from torch import nn
import torch


class Case1ModelInterface(nn.Module):
    """
    The different cases for the PCA initialized models
    """
    def __init__(self, initialU1, initialU2, initialV, initialB1, initialB2):
        """
        Initialize the parameters that create the weights
        :param initialU1:
        :param initialU2:
        :param initialV:
        :param initialB1:
        :param initialB2:
        """
        super(Case1ModelInterface, self).__init__()
        self.U1 = nn.Parameter(torch.clone(initialU1), requires_grad=True)
        self.U2 = nn.Parameter(torch.clone(initialU2), requires_grad=True)
        self.V = nn.Parameter(torch.clone(initialV), requires_grad=True)
        self.B1 = nn.Parameter(torch.clone(initialB1), requires_grad=True)
        self.B2 = nn.Parameter(torch.clone(initialB2), requires_grad=True)

    def forward(self, x):
        return x

    def __str__(self):
        return "GenericCase1"


class Case2ModelInterface(nn.Module):
    def __init__(self, initialW1, initialW2, initialB1, initialB2):
        """
        Initialize the weights beforehand and do gradient descent on them instead
        :param initialW1:
        :param initialW2:
        :param initialB1:
        :param initialB2:
        """
        super(Case2ModelInterface, self).__init__()
        self.W1 = nn.Parameter(torch.clone(initialW1), requires_grad=True)
        self.W2 = nn.Parameter(torch.clone(initialW2), requires_grad=True)
        self.B1 = nn.Parameter(torch.clone(initialB1), requires_grad=True)
        self.B2 = nn.Parameter(torch.clone(initialB2), requires_grad=True)

    def forward(self, x):
        return x

    def __str__(self):
        return "GenericCase2"


class Case3ModelInterface(nn.Module):
    def __init__(self, model1, model2):
        """
        Initialize as 2 different models
        :param model1:
        :param model2:
        """
        super(Case3ModelInterface, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        return x

    def __str__(self):
        return "GenericCase3"

