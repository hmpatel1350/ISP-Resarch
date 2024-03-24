from models import nn_continuous, nn_discontinuous
from problems.denoise_classifier import DenoiseClassifier
from problems.image_as_classification import ImageAsClassification
import sys
import torch
import torch.nn as nn


def main():
    model_number = int(sys.argv[1])
    wrong_hint = sys.argv[2]
    wrong_eval = sys.argv[3]
    max_loops = int(sys.argv[4])
    epochs_per_loop = int(sys.argv[5])
    problem_type = int(sys.argv[6])
    criterion = int(sys.argv[7])

    if problem_type == 1:
        principal_components = 200
        problem = DenoiseClassifier(wrong_hint, wrong_eval, max_loops, epochs_per_loop, principal_components)
    else:
        principal_components = 400
        problem = ImageAsClassification(wrong_hint, wrong_eval, max_loops, epochs_per_loop, principal_components)

    model = createModel(problem, principal_components, model_number)

    if criterion == 1:
        criterion_function = nn.MSELoss()
    else:
        criterion_function = nn.CrossEntropyLoss()


    learning_rate = 0.0001
    problem.setupWandb(model)
    problem.beginTraining(model, criterion_function, learning_rate)
    problem.evaluateTrainingModel(model)

def createModel(problem, principal_components, model_number):
    U = problem.getU()
    U1 = U[:, :principal_components]
    U1_zeros = torch.zeros(U1.size())

    W = U1 @ U1.T
    W_zeros = torch.zeros(W.size())
    B_ones = torch.ones(W.size()[0])
    B_zeros = torch.zeros(W.size()[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_number == 1:
        model = nn_continuous.HadamardCase1(U1, U1_zeros, U1.T, B_zeros, B_ones)
    elif model_number == 2:
        model = nn_continuous.HadamardCase2(W, W_zeros, B_zeros, B_ones)
    elif model_number == 3:
        model = nn_discontinuous.PiecewiseCase1(U1, U1, U1.T, B_zeros, B_zeros)
    else:
        model = nn_discontinuous.PiecewiseCase2(W, W, B_zeros, B_zeros)

    model.to(device)

    return model


if __name__ == "__main__":
    main()
