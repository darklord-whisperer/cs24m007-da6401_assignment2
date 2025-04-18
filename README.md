# CS6910 Fundamentals of Deep Learning - Assignment 2

This repository contains all files for the second assignment of the CS6910 - Fundamentals of Deep Learning course at IIT Madras.

## Contents

- [PART A](#part-a)  
  - [Description](#description)  
  - [Arguments](#arguments)  
  - [Usage](#usage)  
  - [Results](#results)  
- [PART B](#part-b)  
  - [Description](#description-1)  
  - [Arguments](#arguments-1)  
  - [Usage](#usage-1)  
  - [Results](#results-1)  

# PART A

## Description

This part contains code to train a Convolutional Neural Network (CNN) model from scratch using PyTorch. The code demonstrates how to tune hyperparameters and visualize filters to improve the performance of the CNN model.

## Arguments
| Argument             | Description                                                | Choices                                                  |
|----------------------|------------------------------------------------------------|----------------------------------------------------------|
| num_filters          | Number of filters in the convolutional layers              | 32, 64, 128                                              |
| activation           | Activation function for the model                          | ReLU, GELU, SiLU, Mish                                   |
| filter_size_1        | Size of the filter for the first convolutional layer       | 2, 3, 5                                                  |
| filter_size_2        | Size of the filter for the second convolutional layer      | 2, 3, 5                                                  |
| filter_size_3        | Size of the filter for the third convolutional layer       | 2, 3, 5                                                  |
| filter_size_4        | Size of the filter for the fourth convolutional layer      | 2, 3, 5                                                  |
| filter_size_5        | Size of the filter for the fifth convolutional layer       | 2, 3, 5                                                  |
| filter_multiplier    | Multiplier to increase the number of filters               | 1, 2, 0.5                                                |
| data_augmentation    | Whether to use data augmentation                           | Yes, No                                                  |
| batch_normalization  | Whether to use batch normalization                         | Yes, No                                                  |
| dropout              | Dropout rate                                               | 0.2, 0.3                                                 |
| dense_neurons        | Number of neurons in the fully connected layer             | 128, 256, 512, 1024                                      |
| epoch                | Number of epochs for training                              | 5, 10                                                    |
| learning_rate        | Learning rate for optimizer                                | 0.001, 0.0001                                            |

## Usage
- Take script of train_eval_a.py from partA.
  
```bash
!python train_eval_a.py --train_dir "/content/drive/MyDrive/nature_12K/inaturalist_12K/train" --test_dir "/content/drive/MyDrive/nature_12K/inaturalist_12K/val" --num_filters 64 --filter_size_1 3 --filter_size_2 3 --filter_size_3 3 --filter_size_4 3 --filter_size_5 3 --activation SiLU --dense_neurons 1024 --dropout 0.3 --batch_norm Yes --filter_multiplier 1 --learning_rate 0.001 --epochs 20 --data_augmentation Yes
```
- Take script of train_eval_a.py and run it in local machine.
- 
```bash
>python train_eval_a.py --train_dir nature_12K\inaturalist_12K\train --test_dir nature_12K\inaturalist_12K\val --num_filters 64 --filter_size_1 3 --filter_size_2 3 --filter_size_3 3 --filter_size_4 3 --filter_size_5 3 --activation SiLU --dense_neurons 1024 --dropout 0.3 --batch_norm Yes --filter_multiplier 1 --learning_rate 0.001 --epochs 20 --data_augmentation Yes
```


## Results
- Training logs and metrics are tracked using WandB (Weights & Biases).
- Evaluation results on the test set are printed after training completion.
  
# PART B

## Description
This part focuses on fine-tuning pre-trained convolutional neural network (CNN) models using PyTorch for image classification tasks. The goal is to classify images from the iNaturalist 12K dataset into 10 different classes representing various species of organisms. Fine-tuning a pre-trained model is a common practice in many real-world applications of deep learning, where instead of training a model from scratch, a pre-trained model is used as a starting point and then fine-tuned on a specific dataset to adapt it to the new task.

## Arguments
| Argument             | Description                                                | Choices                               |
|----------------------|------------------------------------------------------------|---------------------------------------|
| activation           | Activation function for the model                          | ReLU, GELU, SiLU, Mish                |
| data_augmentation    | Whether to use data augmentation                           | Yes, No                               |
| batch_normalization  | Whether to use batch normalization                         | Yes, No                               |
| dropout              | Dropout rate                                               | 0.2, 0.3                              |
| dense_neurons        | Number of neurons in the fully connected layer             | 128, 256, 512, 1024                   |
| epoch                | Number of epochs for training                              | 5, 10                                 |
| learning_rate        | Learning rate for optimizer                                | 0.001, 0.0001                         |
| strategy             | Training strategy for model                                | feature_extraction, fine_tuning, fine_tuning_partial, progressive_unfreezing |


## Usage
- Take script of train_B.py from partB
```bash
python train_B.py --activation ReLU --data_augmentation Yes --batch_normalization Yes --dropout 0.2 --dense_neurons 512 --epoch 10 --learning_rate 0.0001 --strategy fine_tuning
```
## Results
- Training logs and metrics are tracked using WandB (Weights & Biases).
- Evaluation results on the test set are printed after training completion.

Wandb Report Link :- https://api.wandb.ai/links/cs23m063/9dfo5c1f
