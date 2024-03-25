import torch
from problems.problem_interface import ProblemInterface


class ImageAsClassification(ProblemInterface):
    def create_custom_input_data(self, image, label):
        # Creates two 28x28 images for classifications, one representing black probability
        # and the other white probability
        image_crushed = image.view(image.size(0), -1)
        black = torch.lt(image_crushed, 0.5)
        white = torch.ge(image_crushed, 0.5)

        # Creates the correct item classification
        correct_label = torch.zeros(1, 10)
        correct_label[0][label] = 1
        resized_label = torch.cat([black, white, correct_label], dim=1)
        return resized_label, label

    def problem_name(self):
        return "ImageAsClassification"

    def getNoisyImages(self, images, mask_tensor, input_guess):
        # Gets the correct black and white classifcations
        black_pixels = torch.clone(images[:, 0:784])
        white_pixels = torch.clone(images[:, 784:784 * 2])

        # Alters the probabilities based on the noise mask
        black_tensor = torch.where(mask_tensor == True, black_pixels, torch.rand(black_pixels.size()))
        white_tensor = torch.where(mask_tensor == True, white_pixels, torch.rand(white_pixels.size()))

        noisy_input = torch.cat([black_tensor, white_tensor, input_guess], dim=1)
        return noisy_input

    def format_image(self, images):
        # Image is white when white probability is higher, black when it instead is higher
        black_pixels = torch.clone(images[:, 0:784])
        white_pixels = torch.clone(images[:, 784:784 * 2])
        original_images = torch.lt(black_pixels, white_pixels)

        return original_images.view(self.batch_size, 28, 28), images[:, -10:]
