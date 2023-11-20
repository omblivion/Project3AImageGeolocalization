# Simple_VPR_codebase

This repository is a starting point for implementing a Visual Place Recognition (VPR) pipeline. It includes a simple
ResNet-18 model trained on the GSV dataset, utilizing
the [pytorch_metric_learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) library.

## Download Datasets

**Note**: If you are using Google Colab, you can skip this section.

To download reduced versions of the GSVCities, SF-XL, Tokyo247 datasets (GSV_xs, SF_xs, Tokyo_xs), use the script below:

```shell
python download_datasets.py
```

## Install Dependencies

**Note**: If you are using Google Colab, you can skip this section.

Install the required packages with the following command:

```shell
pip install -r requirements.txt
```

## Running Experiments

Train and test the model on various datasets using the provided instructions.

### Training and Validation

To train and validate on the London dataset and test on the Tokyo_xs dataset, use the command:

```shell
python main.py --train_path data/gsv_xs/train/london \
--val_path data/tokyo_xs/test \
--test_path data/tokyo_xs/test \
--num_workers 2
```

### Testing a Pre-trained Model

To test a pre-trained model, use the command below, ensuring to replace the example path with your specific checkpoint
file:

```shell
python Project3AImageGeolocalization/main.py \
--train_path data/gsv_xs/train/osaka \
--val_path data/sf_xs/test \
--test_path data/sf_xs/val \
--num_workers 4 \
--test \
./LOGS/lightning_logs/version_0/checkpoints/traindata_gsv_xs_train_osaka-valdata_sf_xs_test-testdata_sf_xs_val-epoch06-step0000-R10.0000_R@50.0000.ckpt
```

For testing with the latest checkpoint, use the `--test latest` option:

```shell
python Project3AImageGeolocalization/main.py \
--train_path data/gsv_xs/train/osaka \
--val_path data/sf_xs/test \
--test_path data/sf_xs/val \
--num_workers 4 \
--test latest
```

This will automatically find and use the most recent checkpoint from the `/LOGS/lightning_logs` directory.

```

This formatted version includes:

- A clear title.
- Sections with headers for downloads, installation, and running experiments.
- Code blocks for commands.
- Notes for users on Google Colab.
- Subsections for training, validation, and testing instructions.

## Command Line Arguments

The following are the command line arguments you can use with `main.py` to train, validate, and test the image geolocalization model:

- **`--batch_size`**: Integer.
    - The number of unique places to use per training iteration. One place corresponds to `N` images. Default is 64.
- **`--img_per_place`**: Integer.
    - Defines the effective batch size as `(batch_size * img_per_place)`. This is the total number of images per batch considering all places. Default is 4.
- **`--min_img_per_place`**: Integer.
    - Places with fewer images than this number will be excluded from training. Default is 4.
- **`--max_epochs`**: Integer.
    - The training process will stop once this number of epochs is reached. Default is 20.
- **`--num_workers`**: Integer.
    - The number of subprocesses to use for data loading. More workers can increase the loading speed. Default is 8.
- **`--descriptors_dim`**: Integer.
    - The dimensionality of the output descriptors from the model. Default is 512.
- **`--num_preds_to_save`**: Integer.
    - Specifies the number of predictions to save for each query at the end of training. Useful for analysis and debugging. Default is 0 (none are saved).
- **`--save_only_wrong_preds`**:
    - Flag (no value needed). If set, only incorrect predictions are saved. This is useful for focusing on difficult queries.
- **`--train_path`**: String.
    - The file path to the training dataset. It should contain subdirectories for each place with associated images. Default is `data/gsv_xs/train`.
- **`--val_path`**: String.
    - The file path to the validation dataset. It must contain a `database` and `queries` subdirectory. Default is `data/sf_xs/val`.
- **`--test_path`**: String.
    - The file path to the test dataset. Like the validation dataset, it must contain a `database` and `queries` subdirectory. Default is `data/sf_xs/test`.
- **`--test`**: String.
    - The file path to a specific model checkpoint file to load for evaluation. If set to 'latest', the most recent checkpoint will be used. Leave this argument empty if you wish to train a new model.

To use these arguments, add them to your command line call like so:

```bash
python main.py --batch_size 32 --img_per_place 5 --max_epochs 100 --num_workers 4 --descriptors_dim 256
```

Customize the values according to your computational resources and dataset specifics.

---

## Usage on Colab

We provide you with the notebook `colab_example.ipynb`.
It shows you how to attach your GDrive file system to Colab, unzip the datasets, install packages and run your first experiment.

NB: BEFORE running this notebook, you must copy the datasets zip into your GDrive. You can use the [link](https://drive.google.com/drive/folders/1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf?usp=sharing) that we provided and simply click 'create a copy'. Make sure that you have enough space (roughly 8 GBs)

NB^2: you can ignore the dataset `robotcar_one_every_2m`.