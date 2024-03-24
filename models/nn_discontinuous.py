# define the NN architecture
import torch
from models.model_interfaces import Case1ModelInterface, Case2ModelInterface


class PiecewiseCase1(Case1ModelInterface):
    def forward(self, x):
        negative_factor = torch.add(x @ (self.U1 @ self.V).T, self.B1)
        positive_factor = torch.add(x @ (self.U2 @ self.V).T, self.B2)

        out = torch.where(x < 0, negative_factor, positive_factor)
        return out

    def __str__(self):
        return "PiecewiseCase1"


class PiecewiseCase2(Case2ModelInterface):
    def forward(self, x):
        negative_factor = torch.add(x @ self.W1.T, self.B1)
        positive_factor = torch.add(x @ self.W2.T, self.B2)

        out = torch.where(x < 0, negative_factor, positive_factor)
        return out

    def __str__(self):
        return "PiecewiseCase2"
