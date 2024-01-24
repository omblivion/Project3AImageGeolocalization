# Importing necessary libraries and modules
import time

import pytorch_lightning as pl
import torch

import parser  # Argument parser
import utils  # Custom utility functions
from lightning_model import CustomLightningModel, get_datasets_and_dataloaders

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
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(
        args)
    # Check if a checkpoint path was provided for evaluation
    if args.test == 'latest':
        # Load the latest checkpoint from the logs directory
        model, checkpoint_path = utils.load_latest_checkpoint_model(val_dataset, test_dataset)
        print(f"Loaded model from latest checkpoint: {checkpoint_path}")
        should_train = False
    elif args.test:
        # Load the model from the specified checkpoint
        model, checkpoint_path = utils.load_model_from_checkpoint(args.test, val_dataset, test_dataset)
        print(f"Loaded model from checkpoint: {args.test}")
        should_train = False
    else:
        # Initialize the model for training from scratch
        print("No checkpoint provided, initializing model for training...")
        print("Preparing datasets and dataloaders...")
        # Get datasets and data loaders

        print("Datasets and dataloaders ready.")

        print("Initializing the model...")
        # Instantiate a Lightning model with given parameters
        model = CustomLightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save,
                                     args.save_only_wrong_preds, args.loss_name, args.loss_params)
        initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    print("Model loaded successfully")
    utils.print_program_config(args, model)

    checkpoint_cb = utils.checkpoint_setup(args)

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
