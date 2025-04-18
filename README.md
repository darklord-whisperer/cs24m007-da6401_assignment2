# DA6401 Fundamentals of Deep Learning – Assignment 2

This repository contains all files for the second assignment of the CS6910 – Fundamentals of Deep Learning course at IIT Madras.

## Contents

- [Task](#task)
- [Submission](#submission)
- [Dataset](#dataset)
- [Usage](#usage)  
  - [Running Manually](#running-manually)  
  - [Running a Sweep using WandB](#running-a-sweep-using-wandb)  
  - [Customization](#customization)

---

# Task

*(Describe the assignment task here. E.g., “Train a CNN from scratch and fine‑tune a pre‑trained model on the iNaturalist 12K dataset.”)*

# Submission

*(Explain how to package and submit your solution. E.g., “Upload your `.py` scripts, the `README.md`, and the WandB link as a single zip to Canvas.”)*

# Dataset

The dataset used in this assignment is the **iNaturalist 12K** subset, consisting of images of 10 different species.  
- **Train**: `/content/drive/MyDrive/inaturalist_12K/train`  
- **Val**: `/content/drive/MyDrive/inaturalist_12K/val`  

---

# Usage

### Running Manually
- **Part A** (train from scratch):  
  ```bash
  python train_A.py \
    --num_filters 64 \
    --activation GELU  \
    --filter_size_1 3 \
    --filter_size_2 3 \
    --filter_size_3 3 \
    --filter_size_4 3 \
    --filter_size_5 3 \
    --filter_multiplier 1 \
    --data_augmentation No \
    --batch_normalization No \
    --dropout 0.2 \
    --dense_neurons 512 \
    --epoch 10 \
    --learning_rate 0.0001
