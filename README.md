# CNN Implementation Project

A convolutional neural network implementation from scratch for image classification using PyTorch.

## Project Description

This project implements a CNN model for classifying images from the MNIST/CIFAR-10 dataset. The implementation focuses on understanding the fundamentals of convolutional neural networks, including:

- Building CNN architectures with convolutional, pooling, and fully-connected layers
- Training neural networks using backpropagation and gradient descent
- Evaluating model performance on image classification tasks

## Setup Instructions

### Environment Setup

1. Clone this repository:
   `git clone https://github.com/sahith-M/cnn-project.git`
    `cd cnn-project`

2. Create and activate a conda environment:
   `conda create -n cnn_env python=3.9`
   `conda activate cnn_env`

3. Install required packages:
   `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
   `conda install jupyter matplotlib scikit-learn pandas`

4. Run the project

## Model Architecture

A simple CNN with:

- 2 convolutional layers
- 2 pooling layers
- 2 fully connected layers
- Output layer for classification

## Implementation Timeline

- Data preprocessing and exploration
- Model architecture design
- Training implementation
- Evaluation and visualization
- Fine-tuning and optimization

## Acknowledgments

Based on PyTorch tutorials and CS231n course materials
