import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import random
import numpy as np
import wandb
import matplotlib.pyplot as plt


class ProblemInterface:
    ## OVERRIDE ##
    def create_custom_input_data(self, image, label):
        """
        Creates the custom MNIST dataset using the image and label by combining them into one custom input
        :param image: The MNIST images
        :param label: The array of labels for every image
        :return: Torch tensor of flattened images followed by 10 values for label classification
        """
        image_crushed = image.view(image.size(0), -1)
        return image_crushed, label

    ## OVERRIDE ##
    def problem_name(self):
        """
        Name for the wandb project
        :return:
        """
        return "GenericProblem"

    ## OVERRIDE ##
    def getNoisyImages(self, images, mask_tensor, input_guess):
        """
        Creates a noisy image
        :param images: The images to add noise to
        :param mask_tensor: The mask for which pixels to turn noisy
        :param input_guess: The classification hint
        :return:
        """
        return images

    ## OVERRIDE ##
    def format_image(self, images):
        """
        Unflattened the images and converts it into a format that can be displayed as an image
        :param images: The unformatted images
        :return: The converted images and their labels
        """
        return images.view(self.batch_size, 28, 28), images[:, -10:]

    def __init__(self, wrong_hint, wrong_eval, max_loops, epochs_per_loop, principal_components):
        """
        Initializes a problem based on user parameters
        :param wrong_hint: Whether to use wrong hints in training
        :param wrong_eval: Whether to use wrong hints in evaluation
        :param max_loops: Number of loops the model should be trained to
        :param epochs_per_loop: Number of epochs before adding in a new loop
        :param principal_components: Number of principal components for the PCA initialization
        """
        self.run = None
        self.U = None
        self.X = None
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
        """
        Sets up the dataloaders to train as well as the PCA values
        :return:
        """

        # Gets the MNIST datasets
        train_data, test_data = self.getMNISTImages()
        train_images, train_labels = self.create_dataset(train_data)
        test_images, test_labels = self.create_dataset(test_data)

        # Converts the MNIST images into our custom dataset inputs
        train_custom_dataset = CustomImageDataset(train_images, train_labels)
        test_custom_dataset = CustomImageDataset(test_images, test_labels)

        # Sets the loaders so the training and evaluation functions can grab multiple images at once
        self.train_loader = torch.utils.data.DataLoader(train_custom_dataset,
                                                        batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(test_custom_dataset,
                                                       batch_size=self.batch_size, num_workers=self.num_workers)

        # Gets the U value from SVD
        self.U = self.createManifoldParamaters(train_custom_dataset)

    def setupWandb(self, model, learning_rate, criterion_function):
        """
        Creates the wandb run info
        :param criterion_function:
        :param learning_rate:
        :param model: The model being used for this run
        :return: The wandb run object
        """
        self.run = self.wandbRunInfo(model, learning_rate, criterion_function)

    def beginTraining(self, model, criterion_function, learning_rate):
        """
        Starts training the model on the problem
        :param model: Model to use
        :param criterion_function: The criterion loss function
        :param learning_rate: Learning rate for training
        :return:
        """

        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, self.n_epochs + 1):
            # monitor training loss
            train_loss = 0.0
            loops = (epoch - 1) // self.epochs_per_loop + 1
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / loops)

            ###################
            # train the model #
            ###################
            for data in self.train_loader:
                images, labels = data
                images = images.view(images.size(0), -1)  # Flattens the input images
                images_only = images[:, 0:784]

                random_float = random.random()
                noise = [random_float, 1 - random_float]  # Sets the noise value for the mask

                # Creates a random mask for noise
                mask = np.random.choice([False, True], size=images_only.size(), replace=True, p=noise)
                mask_tensor = torch.tensor(mask, dtype=torch.float32)

                # Creates the noisy image and classification hints
                input_guess = self.getInputGuessesTraining(mask_tensor, labels)
                noisy_images = self.getNoisyImages(images, mask_tensor, input_guess)

                clean_images = images

                # Makes sure the tensors are put to the proper size and device
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

                wandb.log({"final_index_loss": loss.item()})
                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                wandb.log({"train_loss": total_loss.cpu().detach().numpy()})
                # update running training loss
                train_loss += loss.item()
            # Logs the values to wandb
            train_loss = train_loss / len(self.train_loader)
            wandb.log({"epoch_step": epoch, "average_epoch_loss": train_loss})
        return model

    def getInputGuessesTraining(self, mask_tensor, labels):
        """
        Creates the hints for training input
        :param mask_tensor: Random noise to determine how much of a hint to give
        :param labels: Correct labels for the images
        :return: Noisy classification with a hint
        """
        base_guess = torch.full((self.batch_size, 10), 0.1)
        s = torch.sum(mask_tensor, dim=1)
        for i in range(self.batch_size):
            if self.wrong_hint == "True":
                random_integer = random.randint(0, 9)
            else:
                random_integer = 0

            base_guess[i][labels[i].item() - random_integer] = 0.1 + 0.9 * (1.0 - s[i] / 784)
        return base_guess

    def getInputGuessesEvaluation(self, mask_tensor, labels):
        """
        Creates the hints for the evaluation input
        :param mask_tensor: Random noise to determine how much of a hint to give
        :param labels: Correct labels for the images
        :return: Noisy classification with a hint
        """
        base_guess = torch.full((self.batch_size, 10), 0.1)
        s = torch.sum(mask_tensor, dim=1)
        for i in range(self.batch_size):
            if self.wrong_eval == "True":
                eval_hint = 1
            else:
                eval_hint = 0

            base_guess[i][labels[i].item() - eval_hint] = 0.1 + 0.9 * (1.0 - s[i] / 784)
        return base_guess

    def getU(self):
        """
        Gets the U value from SVD
        :return: tensor for U
        """
        return self.U

    def getMNISTImages(self):
        """
        Gets the MNIST datasets
        :return: the train and test dataset images
        """
        transform = transforms.ToTensor()

        # load the training and test datasets
        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False,
                                   download=True, transform=transform)

        return train_data, test_data

    def createManifoldParamaters(self, train_dataset):
        """
        Performs SVD on the answer dataset
        :param train_dataset: The correct training dataset
        :return: U from SVD
        """
        train_images = train_dataset.images

        mnist_tensor = torch.stack(train_images, dim=1)

        X = mnist_tensor.view(len(mnist_tensor[0]), len(mnist_tensor[0][0])).float().T
        self.X = X
        U, S, V = torch.svd(X)
        return U

    def create_dataset(self, dataset):
        """
        Creates images and labels for custom dataset
        :param dataset: MNIST dataset
        :return: lists for the images and labels
        """
        data_images = []
        data_labels = []
        for image, label in dataset:
            new_image, new_label = self.create_custom_input_data(image, label)
            data_images.append(new_image)
            data_labels.append(new_label)
        return data_images, data_labels

    def run_name(self, model, learning_rate, criterion_function):
        """
        Sets the current wandb run name
        :param learning_rate:
        :param model: Model to use for this run
        :return: String for the wandb name
        """
        return ("2Tanh  ToSigmoid-Model:{}-Epochs:{}-WrongHint:{}-WrongEval:{}-LearningRate:{}-Criterion:{}"
                .format(str(model), self.n_epochs, self.wrong_hint, self.wrong_eval,
                        learning_rate, str(criterion_function)))

    def run_config(self, model, learning_rate, criterion_function):
        """
        Sets the wandb run configs
        :param learning_rate:
        :param model: Model for this run
        :return: Dict for the run configs
        """
        return {
            "model_class": str(model),
            "wrong_hint": self.wrong_hint,
            "wrong_eval": self.wrong_eval,
            "max_loops": self.max_loops,
            "epochs_per_loop": self.epochs_per_loop,
            "n_epochs": self.n_epochs,
            "principal_components": self.principal_components,
            "learning_rate": learning_rate,
            "criterion_function": str(criterion_function),
        }

    def wandbRunInfo(self, model, learning_rate, criterion_function):
        """
        Creates the wandb run object
        :param criterion_function:
        :param learning_rate:
        :param model: Model to use
        :return: run object for wandb
        """
        run = wandb.init(
            # Set the project where this run will be logged
            project=self.problem_name(),

            name=self.run_name(model, learning_rate, criterion_function),
            # Track hyperparameters and run metadata
            config=self.run_config(model, learning_rate, criterion_function),
        )
        wandb.define_metric("epoch_step")
        wandb.define_metric("average_epoch_loss", step_metric="epoch_step")
        return run

    def evaluateTrainingModel(self, model):
        """
        Evaluates a model against different noise levels
        :param model: Model to evaluate
        :return: Returns the evaluation info in wandb
        """
        noise_list = [0.05, 0.25, 0.50, 0.75, 1.00]
        loop_list = [1, 3, 5, 10, 20, 50, 100]
        output_loops = self.runEvaluation
        for noise_item in noise_list:
            """
            examples = [output_loops(1, noise_item, model), output_loops(3, noise_item, model),
                        output_loops(5, noise_item, model),
                        output_loops(10, noise_item, model),
                        output_loops(20, noise_item, model), output_loops(100, noise_item, model)]
            """
            examples = [output_loops(loop_item, noise_item, model) for loop_item in loop_list]
            self.run.log({f'Noise: {noise_item}': examples})

    def runEvaluation(self, loops, noise_percent, model):
        """
        Run an evaluation test to save the results to wandb
        :param loops: Number of loops to test the model on
        :param noise_percent: Percent of noise in image
        :param model: Which model to use
        :return: Evaluation image to save in wandb
        """
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
            i = i + 1
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


class CustomImageDataset(Dataset):
    """
    Creates custom dataset for the images and their labels
    """

    def __init__(self, image_tensors, image_labels):
        """
        Create the dataset from images and labels
        :param image_tensors: Images
        :param image_labels: Labels
        """
        self.images = image_tensors
        self.labels = image_labels

    def __len__(self):
        """
        Get the number of images
        :return: int for number of images
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a specific image and label combination
        :param idx: Index in the array
        :return: The specific image/label
        """
        return self.images[idx], self.labels[idx]
