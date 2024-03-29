# Import necessary libraries and modules
import datetime
import glob
import logging
import os
import re
import time
from typing import Tuple

import faiss
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset

# Import custom visualization module
import visualizations
from lightning_model import CustomLightningModel

# Define a list of recall values for evaluation
RECALL_VALUES = [1, 5, 10, 20]


def compute_recalls(eval_ds: Dataset, queries_descriptors: np.ndarray, database_descriptors: np.ndarray,
                    output_folder: str = None, num_preds_to_save: int = 0,
                    save_only_wrong_preds: bool = True) -> Tuple[np.ndarray, str]:
    """
    Compute the recalls given the queries and database descriptors.
    The dataset is needed to know the ground truth positives for each query.
    """

    # Instantiate a FAISS index for nearest neighbor search
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    # Add the database descriptors to the FAISS index
    faiss_index.add(database_descriptors)
    # Delete the database descriptors to free up memory
    del database_descriptors

    # Log the start of recall calculation (for debugging purposes)
    logging.debug("Calculating recalls")
    # Perform a k-NN search to find the nearest neighbors in the database for each query
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    # Get the ground truth positives for each query from the evaluation dataset
    positives_per_query = eval_ds.get_positives()
    # Initialize an array to hold the recall values
    recalls = np.zeros(len(RECALL_VALUES))

    # Iterate through the queries and their corresponding predictions
    for query_index, preds in enumerate(predictions):
        # Check the recall at each specified value (e.g., R@1, R@5, R@10, R@20)
        for i, n in enumerate(RECALL_VALUES):
            # If any of the top-n predictions are in the ground truth positives for this query, increment the recall counters
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break  # Break early once a true positive is found

    # Normalize the recall values by the number of queries and multiply by 100 to convert to percentage
    recalls = recalls / eval_ds.queries_num * 100
    # Create a string representation of the recall values for easy logging/output
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])

    # Save visualizations of the predictions if specified
    if num_preds_to_save != 0:
        # Save the top-n predictions for each query to the output folder (optional)
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)

    # Return the recall values and their string representation
    return recalls, recalls_str


def print_divider(title):
    """
    Print a fancy divider with a title centered.
    """
    divider_line = '+' + '-' * 78 + '+'
    title_line = f"| {' ':<{(76 - len(title)) // 2}}{title}{' ':>{(76 - len(title)) // 2}} |"
    print(divider_line)
    print(title_line)
    print(divider_line)


def print_weights_summary(initial_weights, final_weights):
    print_divider("Model Weights Summary")

    header = f"{'Layer':<40} {'Status':<10} {'Initial Mean':<15} {'Initial Std':<15} {'Final Mean':<15} {'Final Std':<15}"
    print(header)
    print("-" * len(header))

    changed_weights = 0
    unchanged_weights = 0
    initial_means = []
    initial_stds = []
    final_means = []
    final_stds = []

    for name in initial_weights.keys():
        initial_weight = initial_weights[name]
        final_weight = final_weights[name]
        initial_mean, initial_std = initial_weight.mean().item(), initial_weight.std().item()
        final_mean, final_std = final_weight.mean().item(), final_weight.std().item()

        initial_means.append(initial_mean)
        initial_stds.append(initial_std)
        final_means.append(final_mean)
        final_stds.append(final_std)

        if not torch.equal(initial_weight, final_weight):
            weights_status = "CHANGED"
            changed_weights += 1
        else:
            weights_status = "UNCHANGED"
            unchanged_weights += 1

        row = f"{name:<40} {weights_status:<10} {initial_mean:<15.4e} {initial_std:<15.4e} {final_mean:<15.4e} {final_std:<15.4e}"
        print(row)

    general_initial_mean = sum(initial_means) / len(initial_means)
    general_initial_std = sum(initial_stds) / len(initial_stds)
    general_final_mean = sum(final_means) / len(final_means)
    general_final_std = sum(final_stds) / len(final_stds)

    summary = f"\nTotal changed weights: {changed_weights}\nTotal unchanged weights: {unchanged_weights}"
    general_summary = f"General Initial Mean / Std: {general_initial_mean:.4e} / {general_initial_std:.4e}\n"
    general_summary += f"General Final Mean / Std: {general_final_mean:.4e} / {general_final_std:.4e}"

    print("-" * len(header))
    print(summary)
    print(general_summary)
    print_divider("End of Model Weights Summary")


def print_program_config(args):
    current_time_rome = ((
                                 datetime.datetime.utcnow() +
                                 datetime.timedelta(hours=2 if time.localtime().tm_isdst else 1))
                         .strftime('%Y-%m-%d %H:%M:%S CET/CEST'))

    print_divider("Program Configuration")
    print(f"Current Date and Time: {current_time_rome}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Training Path: {args.train_path}")
    print(f"Validation Path: {args.val_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Descriptor Dimension: {args.descriptors_dim}")
    print(f"Number of Predictions to Save: {args.num_preds_to_save}")
    print(f"Save Only Wrong Predictions: {args.save_only_wrong_preds}")
    print(f"Image per Place: {args.img_per_place}")
    print(f"Minimum Image per Place: {args.min_img_per_place}")

    mining_str = args.mining_str if args.mining_str else "Random mining"
    print(f"Mining Strategy: {mining_str}")

    # Check if the testing argument is provided and print relevant information
    if hasattr(args, 'test') and args.test:
        testing_status = f"Model tested from checkpoint: {args.test}" if args.test != 'latest' else "Model tested from the latest checkpoint."
    else:
        testing_status = "Model not in testing mode, training from scratch."

    print(testing_status)

    print_divider("End of Configuration")


def print_model_configuration(model_instance):
    print_divider("Model Configuration")
    print(f"Model Architecture: {model_instance.model.__class__.__name__}")
    print(f"Pretrained: {'Yes' if model_instance.model.fc.in_features else 'No'}")

    optimizer_name = getattr(model_instance.args, 'optimizer_name', 'adamw')
    optimizer_params_str = getattr(model_instance.args, 'optimizer_params', '1e-04,0.9,0.999,1e-3')
    scheduler_name = getattr(model_instance.args, 'scheduler_name', 'reduce_lr_on_plateau')
    scheduler_params_str = getattr(model_instance.args, 'scheduler_params', 'min,3,0.1,True')

    if optimizer_name:
        print(f"Optimizer: {optimizer_name.title()}")
        optimizer_params = optimizer_params_str.split(',') if optimizer_params_str else []
        if optimizer_name == 'adamw':
            print(f"  - Learning rate: {optimizer_params[0] if len(optimizer_params) > 0 else '1e-03'}")
            print(f"  - Betas: {tuple(optimizer_params[1:3]) if len(optimizer_params) > 2 else '(0.9, 0.999)'}")
            print(f"  - Weight decay: {optimizer_params[3] if len(optimizer_params) > 3 else '0'}")
        elif optimizer_name == 'adam':
            print(f"  - Learning rate: {optimizer_params[0] if len(optimizer_params) > 0 else '1e-03'}")
            print(f"  - Betas: {tuple(optimizer_params[1:3]) if len(optimizer_params) > 2 else '(0.9, 0.999)'}")
            print(f"  - Epsilon: {optimizer_params[3] if len(optimizer_params) > 3 else '1e-08'}")
            print(f"  - Weight decay: {optimizer_params[4] if len(optimizer_params) > 4 else '0'}")
        elif optimizer_name == 'asgd':
            print(f"  - Learning rate: {optimizer_params[0] if len(optimizer_params) > 0 else '1e-02'}")
            print(f"  - Weight decay: {optimizer_params[1] if len(optimizer_params) > 1 else '0'}")
        elif optimizer_name == 'sgd':
            print(f"  - Learning rate: {optimizer_params[0] if len(optimizer_params) > 0 else '1e-02'}")
            print(f"  - Weight decay: {optimizer_params[1] if len(optimizer_params) > 1 else '0'}")
            print(f"  - Momentum: {optimizer_params[2] if len(optimizer_params) > 2 else '0'}")
        elif optimizer_name == 'padam':
            print(f"  - Learning rate: {optimizer_params[0] if len(optimizer_params) > 0 else '1e-02'}")
            print(f"  - Betas: {tuple(optimizer_params[1:3]) if len(optimizer_params) > 2 else '(0.9, 0.999)'}")
            print(f"  - Epsilon: {optimizer_params[3] if len(optimizer_params) > 3 else '1e-08'}")
            print(f"  - Weight decay: {optimizer_params[4] if len(optimizer_params) > 4 else '0'}")
            print(f"  - Lambda: {optimizer_params[5] if len(optimizer_params) > 5 else '1e-02'}")
            print(f"  - P Norm: {optimizer_params[6] if len(optimizer_params) > 6 else '1'}")
    else:
        print("Default Optimizer: AdamW with lr=1e-4, betas=(0.9, 0.999), weight decay=1e-3")

    if scheduler_name:
        print(f"Scheduler: {scheduler_name.replace('_', ' ').title()}")
        scheduler_params = scheduler_params_str.split(',') if scheduler_params_str else []
        if scheduler_name == 'reduce_lr_on_plateau':
            mode = 'min' if len(scheduler_params) == 0 or scheduler_params[0] == '0' else 'max'
            print(f"  - Mode: {mode}")
            print(f"  - Patience: {scheduler_params[1] if len(scheduler_params) > 1 else '3'}")
            print(f"  - Factor: {scheduler_params[2] if len(scheduler_params) > 2 else '0.1'}")
            print(f"  - Verbose: {'True' if len(scheduler_params) < 4 or scheduler_params[3] == '1' else 'False'}")
        elif scheduler_name == 'cosine_annealing':
            print(f"  - T_max: {scheduler_params[0] if len(scheduler_params) > 0 else '10'}")
            print(f"  - Eta_min: {scheduler_params[1] if len(scheduler_params) > 1 else '0'}")
        else:
            print("Default Scheduler: ReduceLRonPlateau with mode='min', patience=3, factor=0.1, verbose=True")

    print(f"Loss Function: {model_instance.loss_fn.__class__.__name__}")


def load_latest_checkpoint_model(val_dataset, test_dataset):
    # Specify the directory where checkpoints are saved
    checkpoint_dir = './LOGS/lightning_logs/'
    # Find all the version directories
    version_dirs = glob.glob(os.path.join(checkpoint_dir, 'version_*'))

    # Make sure there is at least one version directory
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found in {checkpoint_dir}")

    # Get the most recent version directory based on creation time
    latest_version_dir = max(version_dirs, key=os.path.getctime)
    # Get the path to the checkpoint directory within the latest version directory
    checkpoint_subdir = os.path.join(latest_version_dir, 'checkpoints')
    # List all the .ckpt files in the checkpoints subdirectory
    list_of_files = glob.glob(os.path.join(checkpoint_subdir, '*.ckpt'))

    # If there are no checkpoints, raise an informative error
    if not list_of_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_subdir}")

    # Get the most recent checkpoint file
    latest_checkpoint = max(list_of_files, key=os.path.getctime)

    # Load the model from the latest checkpoint
    model = CustomLightningModel.load_model_from_checkpoint(latest_checkpoint, val_dataset, test_dataset)

    return model, latest_checkpoint


def load_model_from_checkpoint(checkpoint_path, val_dataset, test_dataset):
    """
    Load a model from the specified checkpoint file.

    :param checkpoint_path: Path to the checkpoint file provided by the user.
    :param val_dataset: The validation dataset to use with the model.
    :param test_dataset: The test dataset to use with the model.
    :return: A tuple of the loaded model and the checkpoint path used.
    """
    # Ensure the checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the model from the specified checkpoint
    model = CustomLightningModel.load_model_from_checkpoint(checkpoint_path, val_dataset, test_dataset)

    return model, checkpoint_path


def checkpoint_setup(args):
    formatted_train_path = args.train_path.replace('/', '_').replace('.', '')
    formatted_val_path = args.val_path.replace('/', '_').replace('.', '')
    formatted_test_path = args.test_path.replace('/', '_').replace('.', '')
    formatted_train_path = re.sub(r'[^A-Za-z0-9_]+', '_', formatted_train_path)
    formatted_val_path = re.sub(r'[^A-Za-z0-9_]+', '_', formatted_val_path)
    formatted_test_path = re.sub(r'[^A-Za-z0-9_]+', '_', formatted_test_path)
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename=f'train{formatted_train_path}-val{formatted_val_path}-test{formatted_test_path}-epoch{{epoch:02d}}-step{{global_step:04d}}-R1{{val/R@1:.4f}}_R@5{{val/R@5:.4f}}',
        save_weights_only=True,
        save_top_k=3,
        mode='max',
        verbose=True,
        auto_insert_metric_name=False,
    )
    return checkpoint_cb
