# Importing necessary libraries and modules
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import parser  # Argument parser
import utils  # Custom utility functions
from lightning_model import LightningModel, get_datasets_and_dataloaders

# Main execution block
if __name__ == '__main__':
    print("""
 .----------------. .----------------. .----------------. .----------------. .----------------. 
| .--------------. | .--------------. | .--------------. | .--------------. | .--------------. |
| | ____    ____ | | |   _____      | | |    ___       | | |  ________    | | |   _____      | |
| ||_   \  /   _|| | |  |_   _|     | | |  .' _ '.     | | | |_   ___ `.  | | |  |_   _|     | |
| |  |   \/   |  | | |    | |       | | |  | (_) '___  | | |   | |   `. \ | | |    | |       | |
| |  | |\  /| |  | | |    | |   _   | | |  .`___'/ _/  | | |   | |    | | | | |    | |   _   | |
| | _| |_\/_| |_ | | |   _| |__/ |  | | | | (___)  \_  | | |  _| |___.' / | | |   _| |__/ |  | |
| ||_____||_____|| | |  |________|  | | | `._____.\__| | | | |________.'  | | |  |________|  | |
| |              | | |              | | |              | | |              | | |              | |
| '--------------' | '--------------' | '--------------' | '--------------' | '--------------' |
 '----------------' '----------------' '----------------' '----------------' '----------------' 
    """)
    print("Welcome to the ML&DL Project! Please wait while the program is starting up...\n")

    # Parse command line arguments
    args = parser.parse_arguments()

    model = None
    should_train = True

    # Check if a checkpoint path was provided for evaluation
    if args.test == 'latest':
        # Load the latest checkpoint from the logs directory
        model, checkpoint_path = utils.load_latest_checkpoint_model()
        print(f"Loaded model from latest checkpoint: {checkpoint_path}")
        should_train = False
    elif args.test:
        # Load the model from the specified checkpoint
        model = LightningModel.load_from_checkpoint(args.checkpoint_path)
        print(f"Loaded model from checkpoint: {args.checkpoint_path}")
        should_train = False
    else:
        # Initialize the model for training from scratch
        print("No checkpoint provided, initializing model for training...")
        print("Preparing datasets and dataloaders...")
        # Get datasets and data loaders
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(
            args)
        print("Datasets and dataloaders ready.")

        print("Initializing the model...")
        # Instantiate a Lightning model with given parameters
        model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save,
                               args.save_only_wrong_preds)
        initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    print("Model loaded successfully")
    utils.print_program_config(args, model)

    # Format the paths to include them in the filename (remove slashes and dots)
    formatted_train_path = args.train_path.replace('/', '_').replace('.', '')
    formatted_val_path = args.val_path.replace('/', '_').replace('.', '')
    formatted_test_path = args.test_path.replace('/', '_').replace('.', '')
    # Define a model checkpointing callback to save the best 3 models based on Recall@1 metric
    # The model will be saved whenever there is an improvement in the R@1 metric. If during an epoch the R@1 metric is among the top 3 values observed so far, the model's state will be saved.
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename=f'{formatted_train_path}-{formatted_val_path}-{formatted_test_path}_epoch({{epoch:02d}})_step({{global_step:04d}})_R@1[{{R@1:.4f}}]',
        save_weights_only=True,
        save_top_k=3,
        mode='max',
        every_n_train_steps=1,  # Save the checkpoint at every training step
    )

    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1  # Assuming you want to use 1 GPU
        precision = 16  # Use 16-bit precision on GPU
        print("Trainer configured with GPU.")
    else:
        accelerator = 'cpu'
        devices = None  # No device IDs are needed for CPU training
        precision = 32  # Use full precision on CPU
        print("Trainer configured with CPU.")

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
        enable_progress_bar=False
    )
    print("Trainer initialized, all ready.")

    print("Starting validation...")
    # Validate the model using the validation data loader
    trainer.validate(model=model, dataloaders=val_loader)
    print("Validation completed.")

    if should_train:
        training_start_time = time.time()
        print("Starting training...")
        # Train the model using the training data loader and validate using the validation data loader
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # Calculate and print training time
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        final_weights = {name: param.clone() for name, param in model.named_parameters()}
        print("Training completed.")

        # Test the model and print the summary
        print("Starting testing...")
        trainer.test(model=model, dataloaders=test_loader)
        testing_end_time = time.time()
        testing_duration = testing_end_time - training_end_time
        print(f"Testing completed in {testing_duration:.2f} seconds.")

        # Print a summary of the model's performance
        print("\nModel Performance Summary:")
        print(f"Training Duration: {training_duration:.2f} seconds")
        print(f"Testing Duration: {testing_duration:.2f} seconds")
        utils.print_program_config(args, model)
        utils.print_weights_summary(initial_weights, final_weights)
    else:
        # Evaluate the model
        print("Evaluating the model...")
        trainer.validate(model=model, dataloaders=val_loader)
        print("Validation completed.")
        print("Starting testing...")
        trainer.test(model=model, dataloaders=test_loader)
        print("Testing completed.")

        # Print a summary of the model's performance
        print("\nModel Performance Summary:")
        utils.print_program_config(args, model)
