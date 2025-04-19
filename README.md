# DA6401 : Fundamentals of Deep Learning - Assignment 2

This repository contains all files for the second assignment of the DA6401 - Fundamentals of Deep Learning course at IIT Madras.

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

## 📂 Dataset Setup

1. **Download the dataset**  
   Download the `nature_12K.zip` from:  
   `https://storage.googleapis.com/wandb_datasets/nature_12K.zip`

2. **Extract the archive**  
   Unzip to produce:
   nature_12K/ └── inaturalist_12K/ ├── train/ └── val/

3. **Set dataset paths**  
- **Google Colab**  
  Upload the `inaturalist_12K` folder to Google Drive, then run:  
  ```bash
  --train_dir "/content/drive/MyDrive/nature_12K/inaturalist_12K/train"
  --test_dir  "/content/drive/MyDrive/nature_12K/inaturalist_12K/val"
  ```
- **Local machine**  
  If the folders reside in your project directory:
  ```bash
  --train_dir "nature_12K/inaturalist_12K/train"
  --test_dir  "nature_12K/inaturalist_12K/val"
  ```

## Usage
- Take script of train_eval_a.py from google collab.
  
```bash
!python train_eval_a.py --train_dir "/content/drive/MyDrive/nature_12K/inaturalist_12K/train" --test_dir "/content/drive/MyDrive/nature_12K/inaturalist_12K/val" --num_filters 64 --filter_size_1 3 --filter_size_2 3 --filter_size_3 3 --filter_size_4 3 --filter_size_5 3 --activation SiLU --dense_neurons 1024 --dropout 0.3 --batch_norm Yes --filter_multiplier 1 --learning_rate 0.001 --epochs 20 --data_augmentation Yes
```
- Take script of train_eval_a.py and run it in local machine.
```bash
python train_eval_a.py --train_dir nature_12K\inaturalist_12K\train --test_dir nature_12K\inaturalist_12K\val --num_filters 64 --filter_size_1 3 --filter_size_2 3 --filter_size_3 3 --filter_size_4 3 --filter_size_5 3 --activation SiLU --dense_neurons 1024 --dropout 0.3 --batch_norm Yes --filter_multiplier 1 --learning_rate 0.001 --epochs 20 --data_augmentation Yes
```


## Results
- Training logs and metrics are tracked using WandB (Weights & Biases).
- Evaluation results on the test set are printed after training completion.
  
# PART B

## Description
This part focuses on fine-tuning pre-trained convolutional neural network (CNN) models using PyTorch for image classification tasks. The goal is to classify images from the iNaturalist 12K dataset into 10 different classes representing various species of organisms. Fine-tuning a pre-trained model is a common practice in many real-world applications of deep learning, where instead of training a model from scratch, a pre-trained model is used as a starting point and then fine-tuned on a specific dataset to adapt it to the new task.

## Arguments
| Argument                | Description                                            | Default     |
|-------------------------|--------------------------------------------------------|-------------|
| `pretrain_model`        | Pre‑trained model architecture to use                  | InceptionV3 |
| `epoch`                 | Number of training epochs                              | 9           |
| `batch_size`            | Number of samples per gradient update                  | 16          |
| `augmentation`          | Whether to apply data augmentation                     | True        |
| `fc_size`               | Number of neurons in the fully‑connected layer         | 256         |
| `droprate`              | Dropout rate                                           | 0.4         |
| `batch_normalization`   | Whether to use batch normalization                     | True        |
| `num_of_trainable_layers` | Number of layers to fine‑tune in the pre‑trained model | 1           |

## Applicable for Google collab:-
## 📂 Dataset Setup

1. **Download the dataset**  
   Download the `nature_12K.zip` from:  
   `https://storage.googleapis.com/wandb_datasets/nature_12K.zip`

2. **Extract the archive**  
   Unzip to produce:
   nature_12K/ └── inaturalist_12K/ ├── train/ └── val/

3. **Set dataset paths**  
- **Google Colab**  
  Upload the `inaturalist_12K` folder to Google Drive, then set:  
  ```bash
  trainset_dir = "/content/drive/MyDrive/nature_12K/inaturalist_12K/train"
  testset_dir =  "/content/drive/MyDrive/nature_12K/inaturalist_12K/val"
  ```
- **Local machine**  
  If the folders reside in your project directory:
  ```bash
  trainset_dir = "nature_12K/inaturalist_12K/train"
  testset_dir =  "nature_12K/inaturalist_12K/val"
  ```
- **remove "%%writefile train_b.py" 1st line from the python script and then run in Google collab:**
  ```bash
  python train_part_b.py

## Usage
- Open this in kaggle and collab
- Take script of train_part_b.py from partB
```bash
python train_part_b.py
```
## Results
- Training logs and metrics are tracked using WandB (Weights & Biases).
- Evaluation results on the test set are printed after training completion.

Wandb Report Link :- https://wandb.ai/cs24m007-iit-madras/Alik_Final_DA6401_DeepLearning_Assignment2/reports/DA6401-Assignment-2-Alik-Sarkar--VmlldzoxMjM0NDQzOA
