import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import random
import numpy as np
import wandb
import matplotlib.pyplot as plt


class ProblemInterface:
    def __init__(self, wrong_hint, wrong_eval, max_loops, epochs_per_loop, principal_components):
        self.run = None
        self.U = None
        self.train_loader = None
        self.test_loader = None

        self.num_workers = 0
        self.batch_size = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.wrong_hint = wrong_hint
        self.wrong_eval = wrong_eval
        self.max_loops = max_loops
        self.epochs_per_loop = epochs_per_loop
        self.n_epochs = max_loops * epochs_per_loop
        self.principal_components = principal_components

        self.setup()

    def setup(self):
        train_data, test_data = self.getMNISTImages()
        train_images, train_labels = self.create_dataset(train_data)
        test_images, test_labels = self.create_dataset(test_data)

        train_custom_dataset = CustomImageDataset(train_images, train_labels)
        test_custom_dataset = CustomImageDataset(test_images, test_labels)

        self.train_loader = torch.utils.data.DataLoader(train_custom_dataset,
                                                        batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(test_custom_dataset,
                                                       batch_size=self.batch_size, num_workers=self.num_workers)

        self.U = self.createManifoldParamaters(train_custom_dataset)

    def setupWandb(self, model):
        self.run = self.wandbRunInfo(model)

    def beginTraining(self, model, criterion_function, learning_rate):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        x = np.repeat(0.1, 10 * self.batch_size)
        base_probability = torch.from_numpy(x)
        base_probability = base_probability.resize(self.batch_size, 10)
        losses = []
        average = []

        for epoch in range(1, self.n_epochs + 1):
            # monitor training loss
            train_loss = 0.0
            loops = (epoch - 1) // self.epochs_per_loop + 1

            ###################
            # train the model #
            ###################
            for data in self.train_loader:
                images, labels = data
                images = images.view(images.size(0), -1)
                images_only = images[:, 0:784]

                random_float = random.random()
                noise = [random_float, 1 - random_float]

                mask = np.random.choice([False, True], size=images_only.size(), replace=True, p=noise)
                mask_tensor = torch.tensor(mask, dtype=torch.float32)

                input_guess = self.getInputGuessesTraining(mask_tensor, labels)
                noisy_images = self.getNoisyImages(images, mask_tensor, input_guess)
                clean_images = images

                noisy_images = noisy_images.to(torch.float32).to(self.device)
                clean_images = clean_images.to(torch.float32).to(self.device)

                noisy_images.to(self.device)
                clean_images.to(self.device)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = noisy_images
                total_loss = 0
                for i in range(loops):
                    outputs = model(outputs)
                    # calculate the loss
                    loss = criterion_function(outputs, clean_images)
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
            train_loss = train_loss / len(self.train_loader)
            average.append(train_loss)
            print('Epoch: {} \t Average Training Loss: {:.6f}'.format(
                epoch,
                train_loss
            ))
            wandb.log({"epoch_step": epoch, "final_index_loss": train_loss})

    # OVERRIDE
    def getNoisyImages(self, images, mask_tensor, input_guess):
        return images

    def getInputGuessesTraining(self, mask_tensor, labels):
        base_guess = torch.full((self.batch_size, 10), 0.1)
        s = torch.sum(mask_tensor, dim=1)
        for i in range(self.batch_size):
            if self.wrong_hint == "True":
                random_integer = random.randint(0, 9)
            else:
                random_integer = 0

            base_guess[i][labels[i].item() - random_integer] = 0.1 + 0.9 * s[i] / 784
        return base_guess

    def getInputGuessesEvaluation(self, mask_tensor, labels):
        base_guess = torch.full((self.batch_size, 10), 0.1)
        s = torch.sum(mask_tensor, dim=1)
        for i in range(self.batch_size):
            if self.wrong_eval == "True":
                eval_hint = 1
            else:
                eval_hint = 0

            base_guess[i][labels[i].item() - eval_hint] = 0.1 + 0.9 * s[i] / 784
        return base_guess

    def getU(self):
        return self.U

    def getMNISTImages(self):
        transform = transforms.ToTensor()

        # load the training and test datasets
        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False,
                                   download=True, transform=transform)

        return train_data, test_data

    def createManifoldParamaters(self, train_dataset):
        train_images = train_dataset.images

        mnist_tensor = torch.stack(train_images, dim=1)

        X = mnist_tensor.view(len(mnist_tensor[0]), len(mnist_tensor[0][0])).float().T
        U, S, V = torch.svd(X)
        return U

    def create_dataset(self, dataset):
        data_images = []
        data_labels = []
        for image, label in dataset:
            new_image, new_label = self.create_custom_input_data(image, label)
            data_images.append(new_image)
            data_labels.append(new_label)
        return data_images, data_labels

    # OVERRIDE
    def create_custom_input_data(self, image, label):
        image_crushed = image.view(image.size(0), -1)
        return image_crushed, label

    # OVERRIDE
    def problem_name(self):
        return "GenericProblem"

    def run_name(self, model):
        return ("Model:{}-Epochs:{}-WrongHint:{}-WrongEval:{}"
                .format(str(model), self.n_epochs, self.wrong_hint, self.wrong_eval))

    def run_config(self, model):
        return {
            "model_class": str(model),
            "wrong_hint": self.wrong_hint,
            "wrong_eval": self.wrong_eval,
            "max_loops": self.max_loops,
            "epochs_per_loop": self.epochs_per_loop,
            "n_epochs": self.n_epochs,
            "principal_components": self.principal_components,
        }

    def wandbRunInfo(self, model):
        run = wandb.init(
            # Set the project where this run will be logged
            project=self.problem_name(),

            name=self.run_name(model),
            # Track hyperparameters and run metadata
            config=self.run_config(model),
        )
        wandb.define_metric("epoch_step")
        wandb.define_metric("final_index_loss", step_metric="epoch_step")
        return run

    def evaluateTrainingModel(self, model):
        noise_list = [0.05, 0.25, 0.50, 0.75, 1.00]

        output_loops = self.runEvaluation
        for noise_item in noise_list:
            examples = [output_loops(1, noise_item, model), output_loops(3, noise_item, model),
                        output_loops(5, noise_item, model),
                        output_loops(10, noise_item, model),
                        output_loops(20, noise_item, model), output_loops(100, noise_item, model)]
            self.run.log({f'Noise: {noise_item}': examples})

    def runEvaluation(self, loops, noise_percent, model):
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images = images.view(images.size(0), -1)
        images_only = images[:, 0:784]

        noise = [noise_percent, 1 - noise_percent]

        mask = np.random.choice([False, True], size=images_only.size(), replace=True, p=noise)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        input_guess = self.getInputGuessesTraining(mask_tensor, labels)

        noisy_images = self.getNoisyImages(images, mask_tensor, input_guess)

        noisy_images = noisy_images.to(torch.float32).to(self.device)

        noisy_images.to(self.device)

        # get sample outputs
        output = torch.clone(noisy_images)
        for i in range(loops):
            output = model(output)

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=3, ncols=20, sharex=True, sharey=True, figsize=(25, 7))
        i = 0

        image_set = [images, noisy_images, output]
        for row in axes:
            unformatted_images = image_set[i]
            i = i+1
            j = 0
            formatted_image, formatted_label = self.format_image(unformatted_images)
            formatted_image = formatted_image.cpu().detach().numpy()
            index_labels = torch.argmax(formatted_label, dim=1)

            for col in row:
                col.imshow(np.squeeze(formatted_image[j]), cmap='gray')
                col.get_xaxis().set_visible(False)
                col.set_title('{}\n{:.4f}'
                             .format(index_labels[j].item(), formatted_label[j].max()))
                col.get_yaxis().set_visible(False)
                j = j + 1

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = wandb.Image(data, caption=f"Loops: {loops}")
        return image

    # OVERRIDE
    def format_image(self, images):
        return images.view(self.batch_size, 28, 28), images[:, -10:]


class CustomImageDataset(Dataset):
    def __init__(self, image_tensors, image_labels):
        self.images = image_tensors
        self.labels = image_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
