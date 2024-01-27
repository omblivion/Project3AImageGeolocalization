# Importing custom dataset classes
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.models
from pytorch_metric_learning import losses, miners
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tfm

import utils
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


# Defining a custom model class that inherits from pytorch_lightning.LightningModule
class CustomLightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True, loss_name='contrastive', loss_params=''):
        super().__init__()  # Calling the superclass initializer
        self.val_dataset = val_dataset  # Validation dataset
        self.test_dataset = test_dataset  # Test dataset
        self.num_preds_to_save = num_preds_to_save  # Number of predictions to save
        self.save_only_wrong_preds = save_only_wrong_preds  # Flag to save only wrong predictions
        # Initializing a pre-trained ResNet-18 model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Modifying the fully connected layer of the ResNet model to match the desired descriptor dimensions
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
        self.miner_fn = None #overwritten for TripletMargin
        # Setting the loss function
        loss_params = [float(param) for param in loss_params.split(',') if param]

        if loss_name == 'multisimilarity':
            alpha = loss_params[0] if len(loss_params) > 0 else 2
            beta = loss_params[1] if len(loss_params) > 1 else 50
            base = loss_params[2] if len(loss_params) > 2 else 1
            self.loss_fn = losses.MultiSimilarityLoss(alpha = alpha, beta = beta, base = base)
        
        elif loss_name == "tripletmargin":
            margin = loss_params[0] if len(loss_params) > 0 else 0.05
            # Set miner with the same margin
            self.miner_fn = miners.TripletMarginMiner(margin=margin)
            self.loss_fn = losses.TripletMarginLoss(margin=margin)

        elif loss_name == "fastap":
            num_bins = loss_params[0] if len(loss_params) > 0 else 10
            self.loss_fn = losses.FastAPLoss(num_bins = num_bins)

        else:
            #default setting of loss to the Contrastive Loss
            self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    def forward(self, images):  # Forward pass method
        descriptors = self.model(images)  # Pass images through the model to get descriptors
        return descriptors  # Return the descriptors

    def configure_optimizers(self):  # Method to configure optimizers
        # Using Stochastic Gradient Descent as the optimizer
        optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        return optimizers

    def loss_function(self, descriptors, labels):  # Method to compute loss
        
        if self.miner_fn is not None:
            miner_output = self.miner_fn(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_output)
        else:
            loss = self.loss_fn(descriptors, labels)  # Compute loss
        return loss

    def training_step(self, batch, batch_idx):  # Method for a single training step
        images, labels = batch  # Unpack batch into images and labels
        num_places, num_images_per_place, C, H, W = images.shape  # Get the shape details of the images
        # Reshape images and labels to match the expected input dimensions for the model
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        descriptors = self(images)  # Forward pass to get descriptors
        loss = self.loss_function(descriptors, labels)  # Compute loss

        self.log('loss', loss.item(), logger=True)  # Log the loss value
        # print(f'Training Step {batch_idx}, Loss: {loss.item()}')
        return {'loss': loss}  # Return the loss value

    def training_epoch_end(self, outputs):
        print(f'Epoch {self.current_epoch + 1} of {self.trainer.max_epochs} complete.')

    def inference_step(self, batch):  # Method for a single inference step
        images, _ = batch  # Unpack batch into images and discard labels
        descriptors = self(images)  # Forward pass to get descriptors
        return descriptors.cpu().numpy().astype(np.float16)  # Convert descriptors to numpy array and return

    # Methods for validation and test steps, which call the inference_step method
    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    # Methods for handling the end of validation and test epochs
    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset)

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, num_preds_to_save=0):
        # Concatenate all descriptors into one array
        all_descriptors = np.concatenate(all_descriptors)
        # Separate query descriptors from database descriptors
        queries_descriptors = all_descriptors[inference_dataset.database_num:]
        database_descriptors = all_descriptors[: inference_dataset.database_num]

        # Compute recalls for the dataset
        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            self.trainer.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        print(recalls_str)  # Print recall values
        # Log recall values
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)

    @classmethod
    def load_model_from_checkpoint(cls, checkpoint_path, val_dataset, test_dataset):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Instantiate the model with the checkpoint hyperparameters
        model = cls(val_dataset=val_dataset, test_dataset=test_dataset)

        # Apply the loaded state_dict to the model instance
        model.load_state_dict(checkpoint['state_dict'])

        return model


# Define a function to create datasets and data loaders for training, validation, and testing
def get_datasets_and_dataloaders(args):
    # Define the transformation pipeline for the training dataset
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),  # Apply RandAugment with 3 random operations
        tfm.ToTensor(),  # Convert images to PyTorch tensors
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Normalize tensors with given mean and std
    ])
    # Create the training dataset
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform  # Apply defined transformations
    )
    # Create the validation and testing datasets without additional transformations
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)

    # Create data loaders for each dataset to handle batching, shuffling, and parallel loading
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Return datasets and data loaders
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
