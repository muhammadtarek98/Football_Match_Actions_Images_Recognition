# Football Match Actions Recognition

This project implements a deep learning model to recognize various actions in football matches from images. It uses a Convolutional Neural Network (CNN) based on the VGG16 architecture, trained using PyTorch Lightning.

## Dataset

The dataset used for this project was created by **Muhammad** and is freely available on Kaggle.

*   **Source:** [Kaggle - Football Match Actions Dataset](https://www.kaggle.com/datasets/itarek898/football-match-dataset-first-version)
*   **Classes:** The dataset currently supports the following action classes:
    *   `goals` (Label: 0)
    *   `red_card` (Label: 1)
    *   `tackles` (Label: 2)

## Project Structure

*   `main.py`: The entry point for training the model. Handles argument parsing, data loading, and initializing the PyTorch Lightning trainer.
*   `DataSet.py`: Contains the `FootBallDataSet` class, a custom PyTorch Dataset that handles image loading and preprocessing using Albumentations.
*   `Model.py`: Defines the neural network architecture. It uses a pre-trained VGG16 backbone followed by custom MLP blocks for classification.
*   `PL_module.py`: The PyTorch Lightning Module (`PLModule`) that defines the training, validation, and testing steps, as well as metric logging (Accuracy, F1 Score, Precision, Recall).

## Requirements

To run this project, you need the following Python libraries:

*   `torch`
*   `torchvision`
*   `pytorch_lightning`
*   `albumentations`
*   `opencv-python` (cv2)
*   `torchmetrics`
*   `torchinfo`
*   `argparse`

You can install them using pip:

```bash
pip install torch torchvision pytorch-lightning albumentations opencv-python torchmetrics torchinfo
```

## How to Train

You can train the model using the `main.py` script. It accepts several command-line arguments to configure the training process.

### Usage

```bash
python main.py --data_dir /path/to/your/dataset [OPTIONS]
```

### Arguments

*   `--data_dir` (Required): Path to the root directory of the dataset containing the images.
*   `--lr`: Learning rate (default: `1e-4`).
*   `--epochs`: Number of training epochs (default: `20`).
*   `--bs`: Batch size (default: `32`).
*   `--num_workers`: Number of data loading workers (default: `4`).

### Example

```bash
python main.py --data_dir ./football_match_dataset --bs 16 --epochs 10 --lr 0.001
```

## Model Architecture

The model (`Model.py`) consists of:
1.  **Backbone:** VGG16 with Batch Normalization (pretrained on ImageNet).
2.  **Feature Extractor:** The first few layers of VGG16 are used to extract features.
3.  **Classifier Head:**
    *   Adaptive Average Pooling.
    *   Two MLP blocks (Linear -> BatchNorm -> LeakyReLU -> Dropout).
    *   Final Linear layer mapping to the number of classes (3).

## Training Details

*   **Optimizer:** Adam
*   **Scheduler:** Cosine Annealing Warm Restarts
*   **Loss Function:** Cross Entropy Loss
*   **Metrics:** Accuracy, F1 Score, Precision, Recall are logged during training and validation.
*   **Callbacks:**
    *   `ModelCheckpoint`: Saves the top 3 models based on validation loss.
    *   `EarlyStopping`: Stops training if validation loss doesn't improve for 5 epochs.
