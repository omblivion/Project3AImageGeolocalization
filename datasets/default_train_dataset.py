from collections import defaultdict
from glob import glob

import numpy as np
import torch
import torchvision.transforms as tfm
from PIL import Image
from torch.utils.data import Dataset

# Define a default transformation for the images
default_transform = tfm.Compose([
    tfm.ToTensor(),  # Convert images to PyTorch tensors
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
])


# Define the TrainDataset class inheriting from Dataset
class DefaultTrainDataset(Dataset):
    def __init__(
            self,
            dataset_folder,
            img_per_place=4,
            min_img_per_place=4,
            transform=default_transform,
    ):
        super().__init__()  # Initialize the superclass
        self.dataset_folder = dataset_folder  # Store the dataset folder path
        # Get all image paths in the dataset folder
        self.images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        # Check if there are any images in the dataset folder
        if len(self.images_paths) == 0:
            raise FileNotFoundError(f"There are no images under {dataset_folder}, you should change this path")

        # Create a dictionary to map place IDs to their image paths
        self.dict_place_paths = defaultdict(list)
        for image_path in self.images_paths:
            # Extract the place ID from the image path
            place_id = image_path.split("@")[-2]
            # Append the image path to the list of paths for this place ID
            self.dict_place_paths[place_id].append(image_path)

        # Ensure that the number of images per place does not exceed the minimum number of images per place
        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place  # Number of images to sample per place
        self.transform = transform  # Transformation to apply to the images

        # Filter out places that do not have the minimum required number of images
        for place_id in list(self.dict_place_paths.keys()):
            all_paths_from_place_id = self.dict_place_paths[place_id]
            if len(all_paths_from_place_id) < min_img_per_place:
                del self.dict_place_paths[place_id]
        self.places_ids = sorted(list(self.dict_place_paths.keys()))  # List of valid place IDs
        # Calculate the total number of images across all places
        self.total_num_images = sum([len(paths) for paths in self.dict_place_paths.values()])

    def __getitem__(self, index):
        # Get the place ID corresponding to the given index
        place_id = self.places_ids[index]
        # Get all image paths for this place ID
        all_paths_from_place_id = self.dict_place_paths[place_id]
        # Randomly select a specified number of images for this place
        chosen_paths = np.random.choice(all_paths_from_place_id, self.img_per_place)
        # Open, convert, and transform the selected images
        images = [Image.open(path).convert('RGB') for path in chosen_paths]
        images = [self.transform(img) for img in images]
        # Stack the images into a tensor and return it with the repeated index
        return torch.stack(images), torch.tensor(index).repeat(self.img_per_place)

    def __len__(self):
        """Denotes the total number of places (not images)"""
        # Return the number of valid places in the dataset
        return len(self.places_ids)
