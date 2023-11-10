# Importing necessary libraries and modules
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tfm

import parser  # Argument parser
import utils  # Custom utility functions
# Importing custom dataset classes
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


# Defining a custom model class that inherits from pytorch_lightning.LightningModule
class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True):
        super().__init__()  # Calling the superclass initializer
        self.val_dataset = val_dataset  # Validation dataset
        self.test_dataset = test_dataset  # Test dataset
        self.num_preds_to_save = num_preds_to_save  # Number of predictions to save
        self.save_only_wrong_preds = save_only_wrong_preds  # Flag to save only wrong predictions
        # Initializing a pre-trained ResNet-18 model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Modifying the fully connected layer of the ResNet model to match the desired descriptor dimensions
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
        # Setting the loss function to ContrastiveLoss
        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    def forward(self, images):  # Forward pass method
        descriptors = self.model(images)  # Pass images through the model to get descriptors
        return descriptors  # Return the descriptors

    def configure_optimizers(self):  # Method to configure optimizers
        # Using Stochastic Gradient Descent as the optimizer
        optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        return optimizers

    def loss_function(self, descriptors, labels):  # Method to compute loss
        loss = self.loss_fn(descriptors, labels)  # Compute Contrastive loss
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
        return {'loss': loss}  # Return the loss value

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
            trainer.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        print(recalls_str)  # Print recall values
        # Log recall values
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)


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


# Main execution block
if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_arguments()

    # Get datasets and data loaders
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)

    # Instantiate a Lightning model with given parameters
    model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save,
                           args.save_only_wrong_preds)

    # Define a model checkpointing callback to save the best 3 models based on Recall@1 metric
    # The model will be saved whenever there is an improvement in the R@1 metric. If during an epoch the R@1 metric is among the top 3 values observed so far, the model's state will be saved.
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename='_epoch({epoch:02d})_step({step:04d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max'
    )

    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1  # Assuming you want to use 1 GPU
        precision = 16  # Use 16-bit precision on GPU
    else:
        accelerator = 'cpu'
        devices = None  # No device IDs are needed for CPU training
        precision = 32  # Use full precision on CPU

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir='./LOGS',
        num_sanity_val_steps=0,
        precision=precision,  # Set precision based on whether GPU is available
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )

    # Validate the model using the validation data loader
    trainer.validate(model=model, dataloaders=val_loader)
    # Train the model using the training data loader and validate using the validation data loader
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test the model using the testing data loader
    trainer.test(model=model, dataloaders=test_loader)
