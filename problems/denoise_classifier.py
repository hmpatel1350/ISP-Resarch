import torch
from problems.problem_interface import ProblemInterface


class DenoiseClassifier(ProblemInterface):
    """
    Problem class for the regular denoiser classifier
    """
    def create_custom_input_data(self, image, label):
        # Simply gets the image and flattens them
        image_crushed = image.view(image.size(0), -1)

        # Creates correct classification
        correct_label = torch.zeros(1, 10)
        correct_label[0][label] = 1

        # Combines flattened images and classifications
        resized_label = torch.cat([image_crushed, correct_label], dim=1)
        return resized_label, label

    def problem_name(self):
        return "DenoiseClassifier"

    def getNoisyImages(self, images, mask_tensor, input_guess):
        # Adds noise to the image based on the mask and combines it with the input guess
        images_only = images[:, 0:784]
        noise_image = torch.where(mask_tensor == True, images_only, torch.rand(images_only.size()))
        noisy_input = torch.cat([noise_image, input_guess], dim=1)
        return noisy_input

    def format_image(self, images):
        # Formats the image portion
        images_only = images[:, 0:784]
        return images_only.view(self.batch_size, 28, 28), images[:, -10:]