# -*- coding: utf-8 -*-

import torch
import numpy as np
import wandb
from torchvision import datasets
import torchvision.transforms as transforms
import sys

model_number = int(sys.argv[1])
wrong_hint = sys.argv[2]
wrong_eval = sys.argv[3]
max_loops = int(sys.argv[4])
epochs_per_loop = int(sys.argv[5])

n_epochs = max_loops * epochs_per_loop
project_name = ("Model_{}-Continuous-{}_Epochs-WrongHint_{}-WrongEval_{}"
                .format(model_number,n_epochs, wrong_hint, wrong_eval))

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# Create a list of tensors, where each tensor is an MNIST image
image_tensors = []
for image, label in train_data:
    image_crushed = image.view(image.size(0), -1)
    correct_label = torch.zeros(1, 10)
    correct_label[0][label] = 1
    resized_label = torch.cat([image_crushed, correct_label], dim=1)
    image_tensors.append(resized_label)

mnist_tensor = torch.stack(image_tensors, dim=1)
print(mnist_tensor.shape)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt

# %matplotlib inline

X = mnist_tensor.view(60000, 794).float().T
first_value = X[:, 0]
plt.imshow(first_value[0:784].view(28, 28), cmap='gray')

index_label = torch.argmax(first_value[784:794])

U, S, V = torch.svd(X)
plt.plot(S)

strangeI = U[:, :200] @ U[:, :200].T
plt.imshow(strangeI)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 100

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import torch.nn as nn
import torch


# define the NN architecture
class SimpleModelCase1(nn.Module):
    def __init__(self, initialU1, initialU2, initialV, initialB1, initialB2):
        super(SimpleModelCase1, self).__init__()
        self.U1 = nn.Parameter(torch.clone(initialU1), requires_grad=True)
        self.U2 = nn.Parameter(torch.clone(initialU2), requires_grad=True)
        self.V = nn.Parameter(torch.clone(initialV), requires_grad=True)
        self.B1 = nn.Parameter(torch.clone(initialB1), requires_grad=True)
        self.B2 = nn.Parameter(torch.clone(initialB2), requires_grad=True)

    def forward(self, x):
        factor1 = torch.add(x @ (self.U1 @ self.V).T, self.B1)
        factor2 = torch.add(x @ (self.U2 @ self.V).T, self.B2)

        out = torch.mul(factor1, factor2)
        return out
        # out = x @ self.W.T
        # return out


class SimpleModelCase2(nn.Module):
    def __init__(self, initialW1, initialW2, initialB1, initialB2):
        super(SimpleModelCase2, self).__init__()
        self.W1 = nn.Parameter(torch.clone(initialW1), requires_grad=True)
        self.W2 = nn.Parameter(torch.clone(initialW2), requires_grad=True)
        self.B1 = nn.Parameter(torch.clone(initialB1), requires_grad=True)
        self.B2 = nn.Parameter(torch.clone(initialB2), requires_grad=True)

    def forward(self, x):
        factor1 = torch.add(x @ self.W1.T, self.B1)
        factor2 = torch.add(x @ self.W2.T, self.B2)

        out = torch.mul(factor1, factor2)
        return out
        # out = x @ self.W.T
        # return out


# initialize the NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

W = U[:, :200] @ U[:, :200].T
W_zeros = torch.zeros(W.size())
B_ones = torch.ones(W.size()[0])
B_zeros = torch.zeros(W.size()[0])

model1 = SimpleModelCase1(U[:, :200], torch.zeros(U[:, :200].size()), U[:, :200].T, B_zeros, B_ones)
model1.to(device)

model2 = SimpleModelCase2(W, W_zeros, B_zeros, B_ones)
model2.to(device)

############
if model_number == 1:
    model = model1
else:
    model = model2
############

# specify loss function
criterion = nn.MSELoss()

lr = 0.0001
# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

import random

x = np.repeat(0.1, 10 * batch_size)
base_probability = torch.from_numpy(x)
base_probability = base_probability.resize(batch_size, 10)

size = 28

run = wandb.init(
    # Set the project where this run will be logged
    project="Continuous_PCA_Initialization",

    name=project_name,
    # Track hyperparameters and run metadata
    config={
        "learning_rate": f'Initial: {lr}, Annealing: lr/loops',
        "epochs": n_epochs,
        "max_loops": max_loops,
        "epochs_per_loop": epochs_per_loop,
        "model_description": "Continuous Non-linearity, Gradient on Initialized Weights",
        "parameter_description:": "W1 = U[200]*U.T, W2=U[200]*=U.T, B1 = Zeros, B2 = Zeros",
        "experiment_description": "Initialized as PCA projection of the manifold"
    },
)
wandb.define_metric("epoch_step")
wandb.define_metric("final_index_loss", step_metric="epoch_step")

model.to(device)
losses = []
average = []
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0
    loops = (epoch - 1) // epochs_per_loop + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 / loops)

    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        images, labels = data
        # flatten images
        images = images.view(images.size(0), -1)
        input_images = torch.clone(images)
        random_float = random.random()
        noise = [random_float, 1 - random_float]
        mask = np.random.choice([True, False], size=images.size(), replace=True, p=noise)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        # Invert the values based on the mask
        # Apply the mask and perform the subtraction
        tensor = images * (1 - mask_tensor) + mask_tensor * (torch.randn(images.size()))
        base_guess = torch.full((batch_size, 10), 0.1)
        s = torch.sum(mask_tensor, dim=1)
        for i in range(batch_size):
            if wrong_hint == "True":
                random_integer = random.randint(0, 9)
            else:
                random_integer = 0
            base_guess[i, labels[i]-random_integer] = 0.1 + 0.9 * s[i] / (size ** 2)

        correct_label = torch.zeros(batch_size, 10)
        for j, value in enumerate(labels):
            index = value.item()
            correct_label[j][index] = 1.0

        resized_input = torch.cat((tensor, base_guess), dim=1)
        resized_label = torch.cat([images, correct_label], dim=1)

        resized_input = resized_input.to(torch.float32).to(device)
        resized_label = resized_label.to(torch.float32).to(device)

        resized_input.to(device)
        resized_label.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = resized_input
        total_loss = 0
        for i in range(loops):
            outputs = model(outputs)
            # calculate the loss
            loss = criterion(outputs, resized_label)
            total_loss += loss * (i + 1)
        wandb.log({"final_loss": loss.item()})
        # backward pass: compute gradient of the loss with respect to model parameters
        total_loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        losses.append(total_loss.cpu().detach().numpy())
        wandb.log({"train_loss": total_loss.cpu().detach().numpy()})
        # update running training loss
        train_loss += loss.item()

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    average.append(train_loss)
    print('Epoch: {} \t Average Training Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))
    wandb.log({"epoch_step": epoch, "final_index_loss": train_loss})

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses)


def output_loops(loops, noise_percent):
    noise = [noise_percent, 1 - noise_percent]
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    input_images = images.view(images.size(0), -1)
    mask = np.random.choice([True, False], size=input_images.size(), replace=True, p=noise)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    # Invert the values based on the mask
    # Apply the mask and perform the subtraction
    tensor = input_images * (1 - mask_tensor) + mask_tensor * (torch.randn(input_images.size()))
    base_guess = torch.full((batch_size, 10), 0.1)
    s = torch.sum(mask_tensor, dim=1)
    if wrong_eval == "True":
        eval_hint = 1
    else:
        eval_hint = 0

    for i in range(batch_size):
        base_guess[i, labels[i] - eval_hint] = 0.1 + 0.9 * s[i] / (size ** 2)

    resized_input = torch.cat((tensor, base_guess), dim=1)
    resized_input = resized_input.to(torch.float32).to(device)

    # get sample outputs
    output = resized_input
    for i in range(loops):
        output = model(output)

    # prep images for display
    noise_image = tensor.view(images.size(0), 28, 28)
    noise_image = noise_image.numpy()
    images = images.numpy()
    output_labels = output[:, -10:]
    index_labels = torch.argmax(output_labels, dim=1)
    output = output[:, :784]
    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.cpu().detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=3, ncols=20, sharex=True, sharey=True, figsize=(25, 7))
    i = 0

    for ax in axes[:, 0]:
        ax.set_xlabel('hi', rotation=0, size='large')

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, noise_image, output], axes):
        i = i + 1
        j = 0
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            if i == 1:
                ax.set_title(f'{labels[j].item()}')
            elif i == 2:
                eval_label = labels[j].item()-eval_hint
                if eval_label == -1:
                    eval_label = 9
                ax.set_title(f'{eval_label}')
            else:
                ax.set_title('{}\n{:.4f}'
                             .format(index_labels[j].item(), output_labels[j].max()))
                # ax.set_title(f'{index_labels[j].item()}\n{output_labels[j].max()}')
            ax.get_yaxis().set_visible(False)
            j = j + 1
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = wandb.Image(data, caption=f"Loops: {loops}")
    return image


noise_list = [0.05, 0.25, 0.50, 0.75, 1.00]

for noise_item in noise_list:
    examples = [output_loops(1, noise_item), output_loops(3, noise_item), output_loops(5, noise_item),
                output_loops(10, noise_item),
                output_loops(20, noise_item), output_loops(100, noise_item)]
    run.log({f'Noise: {noise_item}': examples})
