import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of places to use per iteration (one place is N images)")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="The effective batch size is (batch_size * img_per_place)")
    parser.add_argument("--min_img_per_place", type=int, default=4,
                        help="places with less than min_img_per_place are removed")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="stop when training reaches max_epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of processes to use for data loading / preprocessing")
    # Architecture parameters
    parser.add_argument("--descriptors_dim", type=int, default=512,
                        help="dimensionality of the output descriptors")

    # Visualizations parameters
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                             "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                             "preds for difficult queries, i.e. with uncorrect first prediction")
    # Paths parameters
    parser.add_argument("--train_path", type=str, default="data/gsv_xs/train",
                        help="path to train set")
    parser.add_argument("--val_path", type=str, default="data/sf_xs/val",
                        help="path to val set (must contain database and queries)")
    parser.add_argument("--test_path", type=str, default="data/sf_xs/test",
                        help="path to test set (must contain database and queries)")
    parser.add_argument("--test", type=str, default=None,
                        help="Path to a model checkpoint to load for evaluation, use 'latest' to load the most recent checkpoint. Leave empty for training.")
    # Model parameters
    parser.add_argument("--mining_str", type=str, default=None,
                        help="Specifies the mining strategy to use during training. Options are 'per_class' for contextual or None for random mining.")
    parser.add_argument("--optimizer_name", type=str, default=None,
                        help="Name of the optimizer to use ('adamw', 'asgd', 'adam', 'sgd')")
    parser.add_argument("--optimizer_params", type=str, default=None,
                        help="Comma-separated list of parameters for the optimizer")
    parser.add_argument("--scheduler_name", type=str, default='',
                        help="Name of the scheduler to use ('reduce_lr_on_plateau', 'cosine_annealing')")
    parser.add_argument("--scheduler_params", type=str, default='',
                        help="Comma-separated list of parameters for the scheduler")
    parser.add_argument("--loss_name", type=str, default='',
                        help="Specifies the loss function to be used ('contrastive', 'multisimilarity', 'tripletmargin', 'fastap')")
    parser.add_argument("--loss_params", type=str, default='',
                        help="Comma-separated list of parameters for the loss")
    args = parser.parse_args()
    return args
