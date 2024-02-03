# Importing custom dataset classes
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.models
from pytorch_metric_learning import losses
from torch.optim import ASGD, SGD, Adam, AdamW, lr_scheduler  # type: ignore
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tfm

import utils
from datasets.contextual_train_dataset import ContextualDiverseTrainDataset
from datasets.default_train_dataset import DefaultTrainDataset
from datasets.test_dataset import TestDataset
from p_adam import PAdam  # type: ignore


# Defining a custom model class that inherits from pytorch_lightning.LightningModule
class CustomLightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, args):
        super().__init__()
        self.args = args
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Use values from args or defaults
        descriptors_dim = getattr(args, 'descriptors_dim', 512)
        num_preds_to_save = getattr(args, 'num_preds_to_save', 0)
        save_only_wrong_preds = getattr(args, 'save_only_wrong_preds', True)

        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds

        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)

        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    def forward(self, images):  # Forward pass method
        descriptors = self.model(images)  # Pass images through the model to get descriptors
        return descriptors  # Return the descriptors

    def configure_optimizers(self):
        # Helper function to get attributes
        def get_attribute(attr_name, default=None):
            return getattr(self.args, attr_name, default)

        # Helper function to parse parameters
        def parse_params(params_str, default=[]):
            return [float(param) for param in params_str.split(',') if param] if params_str else default

        # Default optimizer
        default_optimizer = AdamW(self.parameters(), lr=1e-04, betas=(0.9, 0.999), weight_decay=1e-3)

        # Get optimizer details
        optimizer_name = get_attribute('optimizer_name')
        optimizer_params_str = get_attribute('optimizer_params', '')
        optimizer_params = parse_params(optimizer_params_str)

        # Optimizer selection
        if optimizer_name:
            optimizer_name = optimizer_name.lower()
            if optimizer_name == 'adamw':
                lr = optimizer_params[0] if len(optimizer_params) > 0 else 1e-03
                betas = (optimizer_params[1], optimizer_params[2]) if len(optimizer_params) > 2 else (0.9, 0.999)
                weight_decay = optimizer_params[3] if len(optimizer_params) > 3 else 0
                optimizer = AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            elif optimizer_name == 'asgd':
                lr = optimizer_params[0] if len(optimizer_params) > 0 else 1e-02
                weight_decay = optimizer_params[1] if len(optimizer_params) > 1 else 0
                optimizer = ASGD(self.parameters(), lr=lr, weight_decay=weight_decay)
                print("ASGD will not work with ReduceLROnPlateau scheduler, using default scheduler")
                return optimizer
            elif optimizer_name == 'adam':
                lr = optimizer_params[0] if len(optimizer_params) > 0 else 1e-03
                betas = (optimizer_params[1], optimizer_params[2]) if len(optimizer_params) > 2 else (0.9, 0.999)
                eps = optimizer_params[3] if len(optimizer_params) > 3 else 1e-08
                weight_decay = optimizer_params[4] if len(optimizer_params) > 4 else 0
                optimizer = Adam(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            elif optimizer_name == 'sgd':
                lr = optimizer_params[0] if len(optimizer_params) > 0 else 1e-02
                momentum = optimizer_params[1] if len(optimizer_params) > 1 else 0
                weight_decay = optimizer_params[2] if len(optimizer_params) > 2 else 0
                optimizer = SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_name == 'padam':
                print("CURRENTLY BROKEN")
                lr = optimizer_params[0] if len(optimizer_params) > 0 else 1e-02
                betas = (optimizer_params[1], optimizer_params[2]) if len(optimizer_params) > 2 else (0.9, 0.999)
                eps = optimizer_params[3] if len(optimizer_params) > 3 else 1e-08
                weight_decay = optimizer_params[4] if len(optimizer_params) > 4 else 0
                lambda_p = optimizer_params[5] if len(optimizer_params) > 5 else 1e-02
                p_norm = optimizer_params[6] if len(optimizer_params) > 6 else 1
                optimizer = PAdam(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                              lambda_p=lambda_p, p_norm=p_norm)
            else:
                print("Using default optimizer!")
                optimizer = default_optimizer
        else:
            print("Using default optimizer!")
            optimizer = default_optimizer

        # Default scheduler
        default_scheduler = lr_scheduler.ReduceLROnPlateau(default_optimizer, mode='min', patience=3, factor=0.1,
                                                           verbose=True)

        # Get scheduler details
        scheduler_name = get_attribute('scheduler_name')
        scheduler_params_str = get_attribute('scheduler_params', '')
        scheduler_params = parse_params(scheduler_params_str)

        # Scheduler selection
        if scheduler_name:
            scheduler_name = scheduler_name.lower()
            if scheduler_name == 'reduce_lr_on_plateau':
                mode = 'min' if len(scheduler_params) == 0 or scheduler_params[0] == 0 else 'max'
                patience = 3 if len(scheduler_params) < 2 else int(scheduler_params[1])
                factor = 0.1 if len(scheduler_params) < 3 else scheduler_params[2]
                verbose = True if len(scheduler_params) < 4 else bool(scheduler_params[3])
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor,
                                                           verbose=verbose)
            elif scheduler_name == 'cosine_annealing':
                T_max = 10 if len(scheduler_params) == 0 else scheduler_params[0]
                eta_min = 0 if len(scheduler_params) < 2 else scheduler_params[1]
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            elif scheduler_name == 'cosine_annealing_warm':
                T_0 = 8 if len(scheduler_params) == 0 else scheduler_params[0]
                eta_min = 1e-5 if len(scheduler_params) < 2 else scheduler_params[1]
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
            # Add other schedulers as needed...
            else:
                print("Using default scheduler!")
                scheduler = default_scheduler
        else:
            print("Using default scheduler!")
            scheduler = default_scheduler

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

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

    # Depending on the sampling strategy, create different samplers
    if args.sampling_str == 'contextual':
        train_dataset = ContextualDiverseTrainDataset(
            # dataset_folder=args.train_path,
            # img_per_place=args.img_per_place,
            # min_img_per_place=args.min_img_per_place,
            # transform=train_transform  # Apply defined transformations
        )
    else:
        train_dataset = DefaultTrainDataset(
            dataset_folder=args.train_path,
            img_per_place=args.img_per_place,
            min_img_per_place=args.min_img_per_place,
            transform=train_transform  # Apply defined transformations
        )

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    # Create the validation and testing datasets without additional transformations
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)

    # Create data loaders for validation and testing datasets
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Return datasets and data loaders
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
