# Import necessary libraries and modules
import datetime
import glob
import logging
import os
import time
from typing import Tuple

import faiss
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

# Import custom visualization module
import visualizations
from main import LightningModel

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


def print_program_config(args, model):
    current_time_rome = (
            datetime.datetime.utcnow() + datetime.timedelta(hours=2 if time.localtime().tm_isdst else 1)).strftime(
        '%Y-%m-%d %H:%M:%S CET/CEST')
    """
    Print the configuration settings for the program, including model details.
    """

    print_divider("Program Configuration")
    print(f"Current Date and Time: {current_time_rome}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Training Path: {args.train_path}")
    print(f"Validation Path: {args.val_path}")
    print(f"Test Path: {args.test_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Descriptor Dimension: {args.descriptors_dim}")
    print(f"Number of Predictions to Save: {args.num_preds_to_save}")
    print(f"Save Only Wrong Predictions: {args.save_only_wrong_preds}")
    print(f"Image per Place: {args.img_per_place}")
    print(f"Minimum Image per Place: {args.min_img_per_place}")
    print_divider("Model Configuration")
    print(f"Model Architecture: {model.model.__class__.__name__}")
    print(f"Pretrained: {torchvision.models.ResNet18_Weights.DEFAULT is not None}")
    print(f"Optimizer: SGD with lr=0.001, weight_decay=0.001, momentum=0.9   *this is a static print statement")
    print(f"Loss Function: {model.loss_fn.__class__.__name__}")
    print_divider("End of Configuration")


def load_latest_checkpoint_model():
    # Specify the directory where checkpoints are saved
    checkpoint_dir = './LOGS'
    # List all the .ckpt files in the directory
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    # Get the most recent checkpoint file
    latest_checkpoint = max(list_of_files, key=os.path.getctime)
    # Load the model from the latest checkpoint
    model = LightningModel.load_from_checkpoint(latest_checkpoint)
    return model, latest_checkpoint
