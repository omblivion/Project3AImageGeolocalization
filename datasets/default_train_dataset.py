from collections import defaultdict
from glob import glob

import numpy as np
import torch
import torchvision.transforms as tfm
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50

# Define a transformation pipeline to preprocess the images
default_transform = tfm.Compose([
    tfm.ToTensor(),  # Convert images to PyTorch tensors, making them suitable for model input
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Normalize images using predefined mean and std, aligning with common practice for models like ResNet
])


class FeatureExtractor:
    def __init__(self):
        self.model = resnet50(pretrained=True).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0)  # Add batch dimension
            return self.model(image).squeeze(0)  # Remove batch dimension for single image

# Define the TrainDataset class that inherits from PyTorch's Dataset class
class DefaultTrainDataset(Dataset):
    class DefaultTrainDataset(Dataset):
        """
        A PyTorch Dataset class for loading and transforming images from a structured directory,
        where images are grouped by place IDs, allowing for the random sampling of images per place.

        Attributes:
            dataset_folder (str): The path to the root directory containing the dataset images.
                                  The images are expected to be in subdirectories named after their
                                  respective place IDs.
            img_per_place (int): The number of images to randomly sample from each place for a given
                                 retrieval. This ensures diversity in the training samples.
            min_img_per_place (int): The minimum number of images required for a place to be included
                                     in the dataset. Places with fewer images are excluded.
            transform (callable, optional): A function/transform that takes in a PIL image and returns
                                            a transformed version. Typically, this includes conversions
                                            to tensor and normalization.
            images_paths (list): A list of all image paths in the dataset, sorted alphabetically.
            dict_place_paths (defaultdict(list)): A dictionary mapping place IDs to their respective
                                                  list of image paths.
            places_ids (list): A sorted list of valid place IDs that meet the `min_img_per_place`
                               requirement.
            total_num_images (int): The total number of images across all valid places in the dataset.

        Methods:
            __init__(self, dataset_folder, img_per_place=4, min_img_per_place=4, transform=default_transform):
                Initializes the dataset object, setting up the directory structure, and preparing
                the dataset for use.

            __getitem__(self, index):
                Allows indexed access to the dataset, randomly sampling `img_per_place` images
                for the place corresponding to the provided index. Returns a batch of images
                and their associated index.

            __len__(self):
                Returns the total number of valid places in the dataset, determining the iteration
                size of the dataset when used with a DataLoader.

        The class is designed to be used with PyTorch's DataLoader to enable efficient batching,
        shuffling, and parallel data loading in a training loop.
        """

    def __init__(
            self,
            dataset_folder,
            img_per_place=4,
            min_img_per_place=4,
            transform=default_transform,
    ):
        super().__init__()  # Call the constructor of the superclass (Dataset) to handle any necessary initialization
        self.dataset_folder = dataset_folder  # Path to the folder containing dataset images
        self.images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg",
                                        recursive=True))  # Retrieve all .jpg image paths within the dataset folder, sorted to maintain consistency

        if len(self.images_paths) == 0:
            # If no images are found in the specified folder, raise an error
            raise FileNotFoundError(f"There are no images under {dataset_folder}, you should change this path")

        self.dict_place_paths = defaultdict(list)  # Initialize a dictionary to map 'place ID' to a list of image paths
        for image_path in self.images_paths:
            place_id = image_path.split("@")[
                -2]  # Extract 'place ID' from the image path, assuming a specific naming convention
            self.dict_place_paths[place_id].append(image_path)  # Group images by their place ID

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than or equal to {min_img_per_place}"  # Ensure img_per_place does not exceed the minimum required images per place
        self.img_per_place = img_per_place  # Set the number of images to sample per place
        self.transform = transform  # Assign the transformation to apply to each image

        # Filter out places with fewer images than the specified minimum
        for place_id in list(self.dict_place_paths.keys()):
            if len(self.dict_place_paths[place_id]) < min_img_per_place:
                del self.dict_place_paths[place_id]  # Remove places not meeting the image count requirement
        self.places_ids = sorted(list(self.dict_place_paths.keys()))  # Create a sorted list of valid place IDs

        # Calculate the total number of images to process across all valid places
        self.total_num_images = sum(len(paths) for paths in self.dict_place_paths.values())

    def __getitem__(self, index):
        place_id = self.places_ids[index]
        all_paths_from_place_id = self.dict_place_paths[place_id]

        # Extract features for all images in this place
        features = [self.features[path] for path in all_paths_from_place_id]

        # Calculate pairwise differences (simplified example using Euclidean distance)
        diffs = np.zeros((len(features), len(features)))
        for i in range(len(features)):
            for j in range(len(features)):
                diffs[i, j] = torch.norm(features[i] - features[j]).item()

        # Select images based on these differences (example strategy: select two most dissimilar images)
        if len(all_paths_from_place_id) > 1:
            i, j = np.unravel_index(diffs.argmax(), diffs.shape)
            chosen_paths = [all_paths_from_place_id[i], all_paths_from_place_id[j]]
        else:
            chosen_paths = [all_paths_from_place_id[0]]

        images = [self.transform(Image.open(path).convert('RGB')) for path in chosen_paths]
        return torch.stack(images), torch.tensor(index).repeat(len(chosen_paths))


    def __len__(self):
        # Return the total number of places in the dataset, determining the dataset's iteration size
        return len(self.places_ids)
