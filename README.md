# MNIST Classification Project

This repository contains a PyTorch implementation for classifying MNIST digits using a 3-layer MLP (Multi-Layer Perceptron). The project includes downloading and preprocessing the MNIST dataset, training the MLP model, and evaluating its performance.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Tasks](#tasks)
- [Results](#results)
- [References](#references)

## Introduction

The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits, each of size 28x28 pixels, with corresponding labels. This project uses an MLP model to classify these digits, demonstrating key deep learning concepts such as data augmentation, model training, and evaluation.

## Setup

### Prerequisites

- Python 3.11
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- argparse
- numpy
- tqdm

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/mnist-classification.git
    cd mnist-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Script

To train the model and evaluate its performance, run the following command:

```bash
python mnist_classification.py --epochs 5 --batch_size 64 --hidden_layers 256 512 --activation relu --loss_function cross_entropy --optimizer adam

```markdown
## Command-line Arguments

- `--epochs`: Number of epochs for training (default: 5).
- `--batch_size`: Batch size for training, validation, and testing (default: 64).
- `--hidden_layers`: List of hidden layer sizes (default: [256, 512]).
- `--activation`: Activation function to use ('relu' or 'sigmoid', default: 'relu').
- `--loss_function`: Loss function to use ('cross_entropy' or 'mse', default: 'cross_entropy').
- `--optimizer`: Optimizer to use ('adam' or 'sgd', default: 'adam').

## Tasks

### Task 0: Data Preparation and Augmentation

- Download the MNIST dataset using `torchvision`.
- Split the data into training, validation, and test sets.
- Apply augmentations: `RandomRotation`, `RandomCrop`, `ToTensor`, and `Normalize`.

### Task 1: Data Visualization and DataLoader Creation

- Plot a few images from each class.
- Create DataLoaders for training, validation, and test datasets.

### Task 2: Model Implementation

- Implement a 3-layer MLP using PyTorch with linear layers.
- Print the number of trainable parameters in the model.

### Task 3: Model Training

- Train the model for the specified number of epochs using Adam as the optimizer and CrossEntropyLoss as the loss function.
- Evaluate the model on the validation set after each epoch.
- Save the best model based on validation loss.
- Log the accuracy and loss of the model on training and validation data at the end of each epoch.

### Task 4: Results Visualization

- Visualize correct and incorrect predictions.
- Plot loss vs. epoch and accuracy vs. epoch graphs for both training and validation datasets.

## Results

### Visualizing Predictions

Correct and incorrect predictions are visualized along with their corresponding images.

### Loss and Accuracy Graphs

Loss and accuracy graphs for training and validation datasets are plotted to show the model's performance over epochs.
```
