import os
import pickle
from collections import defaultdict
from glob import glob

import gdown
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as tfm
import torchvision.transforms as transforms
from PIL import Image
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights

# Define a default transformation for the images
default_transform = tfm.Compose([
    tfm.ToTensor(),  # Convert images to PyTorch tensors
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
])

"""
Contextual Feature Extraction: Extract contextual features from images that are relevant to street scenes. 
These features could include the presence of specific objects (like vehicles, pedestrians, signs), the time of day (daylight, night), weather conditions (sunny, rainy), and urban vs. rural settings. 
This requires a pre-processing step where such features are either manually labeled or automatically detected using a pre-trained object detection or scene classification model.

Diversity Sampling Based on Contextual Features: Once you have these contextual features, the sampling strategy should aim to select images that represent a diverse set of these features for each place. 
This ensures that the model is trained on images that are not just visually diverse but also contextually varied.
"""


class ContextualDiverseTrainDataset(Dataset):

    def __init__(
            self,
            dataset_folder,
            img_per_place=4,
            min_img_per_place=4,
            transform=default_transform,
    ):
        super().__init__()  # Initialize the superclass
        self.device = None
        self.feature_transform = None
        self.feature_extractor = None
        self.dataset_folder = dataset_folder  # Store the dataset folder path
        self.img_per_place = img_per_place  # Number of images to sample per place
        self.transform = transform  # Transformation to apply to the images

        self.initialize_feature_extractor()

        # Get all image paths in the dataset folder
        self.images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(self.images_paths) == 0:
            # If no images are found, raise a FileNotFoundError
            raise FileNotFoundError(f"There are no images under {dataset_folder}, you should change this path")

        # Create a dictionary to map place IDs to their image paths
        self.dict_place_paths = defaultdict(list)
        for image_path in self.images_paths:
            # Extract the place ID from the image path
            place_id = image_path.split("@")[-2]
            # Append the image path to the list of paths for this place ID
            self.dict_place_paths[place_id].append(image_path)

        # Filter out places that do not have the minimum required number of images
        for place_id in list(self.dict_place_paths.keys()):
            if len(self.dict_place_paths[place_id]) < min_img_per_place:
                # Remove places with insufficient images
                del self.dict_place_paths[place_id]
        self.places_ids = sorted(list(self.dict_place_paths.keys()))  # List of valid place IDs

        # Extract contextual features for each image
        self.contextual_features = self.extract_contextual_features()
        self.cached_chosen_paths = defaultdict(list)

        # Calculate pairwise distances between images based on contextual features
        # print("Calculating pairwise distances...")
        # self.pairwise_distances = self.calculate_pairwise_distances()

    def extract_contextual_features(self):
        print("ContextualDiverseTrainDataset starting up!")
        features_file = os.path.join(os.getcwd(), "contextual_features.pkl")
        google_drive_url = 'https://drive.google.com/uc?id=1vUP4M5GLUf3ZIGleZBdfOpbhcIvROhKm'

        # Check if the features have already been extracted and saved
        if os.path.exists(features_file):
            print("Loading previously contextual_features.pkl from disk...")
            try:
                with open(features_file, 'rb') as file:
                    file = pickle.load(file)
                    print("Loaded extracted features successfully!")
                    return file
            except (EOFError, pickle.UnpicklingError):
                print("The file is not valid, proceeding to download or calculate...")

        print("Downloading previously calculated contextual_features.pkl file from Google Drive...")
        gdown.download(google_drive_url, features_file, quiet=False)

        # Check if the downloaded file is valid
        if os.path.exists(features_file):
            try:
                with open(features_file, 'rb') as file:
                    file = pickle.load(file)
                    print("Downloaded and loaded features successfully!")
                    return file
            except (EOFError, pickle.UnpicklingError):
                print("Downloaded file is not valid, proceeding to calculate...")

        # Initialize a dictionary to store the contextual features
        contextual_features = {}

        # Log the start of the feature extraction process
        print("Starting feature extraction...")
        print(f"Total images to iterate: {len(self.images_paths)}")
        for i, path in enumerate(self.images_paths):
            # Extract features for each image
            features = self.get_features_for_image(path)
            contextual_features[path] = features

            # Log the progress every 10%
            if (i + 1) % (len(self.images_paths) // 10) == 0 or i == len(self.images_paths) - 1:
                percentage_complete = (i + 1) / len(self.images_paths) * 100
                print(f"Feature extraction progress: {percentage_complete:.2f}% complete")

        # Save the extracted features to disk in the current working directory
        with open(features_file, 'wb') as file:
            pickle.dump(contextual_features, file)
            print("Saved extracted features in contextual_features.pkl successfully!")

        return contextual_features

    def initialize_feature_extractor(self):
        # Check if CUDA (GPU support) is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load a pre-trained ResNet50 model with the new weights parameter
        weights = ResNet50_Weights.DEFAULT
        self.feature_extractor = models.resnet50(weights=weights).eval()

        # Move the model to the specified device (GPU or CPU)
        self.feature_extractor = self.feature_extractor.to(self.device)

        # Disable gradient calculations
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Define the image preprocessing transformations
        self.feature_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features_for_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self.feature_transform(image)

        # Add an extra batch dimension
        image = image.unsqueeze(0).to(self.device)

        # Extract features using the pre-trained model
        with torch.no_grad():
            features = self.feature_extractor(image)

        # Apply adaptive average pooling and move the tensor back to CPU for numpy conversion
        if features.ndim == 4:
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            return pooled_features.view(-1).cpu().numpy()
        else:
            if features.ndim == 2:
                return features.view(-1).cpu().numpy()
            else:
                raise ValueError("Feature tensor does not have the expected dimensions")

    def calculate_pairwise_distances(self, batch_size=1000):
        distances_file = os.path.join(self.dataset_folder, "pairwise_distances.pkl")

        # Check if the distances have already been calculated and saved
        if os.path.exists(distances_file):
            print("Loading previously calculated pairwise distances from disk...")
            with open(distances_file, 'rb') as file:
                file = pickle.load(file)
                print("Loaded pairwise distances successfully!")
                return file
        else:
            print("No file to load from disk, performing pairwise distances calculation...")

        feature_vectors = list(self.contextual_features.values())
        n_samples = len(feature_vectors)

        temp_files = []

        print("Starting pairwise distance calculation...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = feature_vectors[start_idx:end_idx]

            batch_distances = pairwise_distances(batch, feature_vectors, metric='euclidean')

            # Save batch_distances to a temporary file
            temp_file = os.path.join(self.dataset_folder, f"temp_distances_{start_idx}.pkl")
            with open(temp_file, 'wb') as file:
                pickle.dump(batch_distances, file)
            temp_files.append(temp_file)

            percentage_complete = (end_idx / n_samples) * 100
            print(f"Pairwise distance calculation progress: {percentage_complete:.2f}% complete")

        # Load and merge all the batch distances
        distances = lil_matrix((n_samples, n_samples))
        for temp_file in temp_files:
            with open(temp_file, 'rb') as file:
                batch_distances = pickle.load(file)
                start_idx = int(os.path.basename(temp_file).split('_')[-1].split('.')[0])
                distances[start_idx:start_idx + batch_distances.shape[0], :] = batch_distances

            # Delete temporary file
            os.remove(temp_file)

        print("Pairwise distance calculation completed.")

        # Save the final distances to disk
        with open(distances_file, 'wb') as file:
            pickle.dump(distances, file)
            print("Saved pairwise distances successfully!")

        return distances

    def __getitem__(self, index):
        # Retrieve the unique identifier for the place corresponding to the given index.
        place_id = self.places_ids[index]

        # Check if the set of diverse images for this place_id has already been selected and cached.
        if not self.cached_chosen_paths[place_id]:
            # If not cached, retrieve all image paths associated with this place_id.
            all_paths_from_place_id = self.dict_place_paths[place_id]
            # Select a subset of images that are diverse based on their contextual features.
            chosen_paths = self.select_diverse_images(all_paths_from_place_id)
            # Cache the selected paths for future access to avoid re-computation.
            self.cached_chosen_paths[place_id] = chosen_paths
        else:
            # If the diverse set of images is already cached, retrieve it directly.
            chosen_paths = self.cached_chosen_paths[place_id]

        # Load, convert to RGB, and apply the specified transformations to each selected image.
        images = [Image.open(path).convert('RGB') for path in chosen_paths]
        images = [self.transform(img) for img in images]

        # Stack the images into a single tensor and repeat the index for each image to indicate their origin.
        return torch.stack(images), torch.tensor(index).repeat(self.img_per_place)


    def select_diverse_images(self, image_paths):
        # Determine the total number of images available for the current place.
        num_images = len(image_paths)

        # If the number of available images is less than or equal to the desired number per place, return all images.
        if num_images <= self.img_per_place:
            return image_paths

        # For each image, retrieve precomputed contextual features from a dictionary.
        features = np.array([self.contextual_features[path] for path in image_paths])

        # The number of clusters for K-Means is the smaller of the desired images per place or the total available images.
        num_clusters = min(self.img_per_place, len(features))

        # Execute K-Means clustering on the features to group images into clusters based on similarity.
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
        # Assuming 'features' is your array of image features and 'kmeans' is your fitted K-Means model
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Calculate distances of all features to their respective cluster centroids
        distances_to_centroids = np.linalg.norm(features - centroids[labels], axis=1)

        # Assuming distances_to_centroids is computed and labels are available
        unique_labels = np.unique(labels)
        # Initialize an array to hold the index of the minimum distance image for each cluster
        selected_indices = np.zeros_like(unique_labels)

        for i, label in enumerate(unique_labels):
            # Create a mask for the current cluster
            mask = labels == label
            # Use this mask to filter distances and identify the index of the minimum value
            cluster_indices = np.arange(len(labels))[mask]
            min_index = cluster_indices[np.argmin(distances_to_centroids[mask])]
            selected_indices[i] = min_index

        # Now, selected_indices contains the indices of the images closest to each cluster's centroid
        selected_image_paths = [image_paths[i] for i in selected_indices]

        # Return the paths of the selected images, ensuring a diverse representation of the place.
        return selected_image_paths

    def __sizeof__(self):
        # Return the number of valid places in the dataset
        return len(self.places_ids)

    def __len__(self):
        # Return the number of valid places in the dataset
        return len(self.places_ids)
