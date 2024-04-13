# define the NN architecture
import torch
from models.model_interfaces import Case1ModelInterface, Case2ModelInterface, Case3ModelInterface


class HadamardCase1(Case1ModelInterface):
    def forward(self, x):
        factor1 = torch.add(x @ (self.U1 @ self.V).T, self.B1)
        factor2 = torch.add(x @ (self.U2 @ self.V).T, self.B2)

        out = torch.mul(factor1, factor2)
        # out = torch.sigmoid(out)
        return out

    def __str__(self):
        return "HadamardCase1"


class HadamardCase2(Case2ModelInterface):
    def forward(self, x):
        factor1 = torch.add(x @ self.W1.T, self.B1)
        factor2 = torch.add(x @ self.W2.T, self.B2)

        out = torch.mul(factor1, factor2)
        # out = torch.sigmoid(out)

        return out

    def __str__(self):
        return "HadamardCase2"


class HadamardCase3(Case3ModelInterface):
    def forward(self, x):
        factor1 = self.model1(x)
        factor1 = torch.tanh(factor1)
        factor2 = self.model2(x)
        factor2 = torch.tanh(factor2)

        out = torch.mul(factor1, factor2)
        out = torch.sigmoid(out)

        return out

    def __str__(self):
        return "HadamardCase3"
