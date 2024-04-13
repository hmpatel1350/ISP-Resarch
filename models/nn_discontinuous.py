# define the NN architecture
import torch
from models.model_interfaces import Case1ModelInterface, Case2ModelInterface, Case3ModelInterface


class PiecewiseCase1(Case1ModelInterface):
    def forward(self, x):
        negative_factor = torch.add(x @ (self.U1 @ self.V).T, self.B1)
        positive_factor = torch.add(x @ (self.U2 @ self.V).T, self.B2)

        out = torch.where(x < 0, negative_factor, positive_factor)
        #out = torch.sigmoid(out)

        return out

    def __str__(self):
        return "PiecewiseCase1"


class PiecewiseCase2(Case2ModelInterface):
    def forward(self, x):
        negative_factor = torch.add(x @ self.W1.T, self.B1)
        positive_factor = torch.add(x @ self.W2.T, self.B2)

        out = torch.where(x < 0, negative_factor, positive_factor)
        #out = torch.sigmoid(out)

        return out

    def __str__(self):
        return "PiecewiseCase2"


class PiecewiseCase3(Case3ModelInterface):
    def forward(self, x):
        negative_factor = self.model1(x)
        negative_factor = torch.tanh(negative_factor)
        positive_factor = self.model2(x)
        positive_factor = torch.tanh(positive_factor)

        out = torch.where(x < 0, negative_factor, positive_factor)
        out = torch.sigmoid(out)

        return out

    def __str__(self):
        return "PiecewiseCase3"
