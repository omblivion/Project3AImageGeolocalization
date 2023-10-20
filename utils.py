# Import necessary libraries and modules
import logging
from typing import Tuple

import faiss
import numpy as np
from torch.utils.data import Dataset

# Import custom visualization module
import visualizations

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
