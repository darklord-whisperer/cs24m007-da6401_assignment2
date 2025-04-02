<div style="text-align: justify;">

# CS6910 Fundamentals of Deep Learning - Assignment 2

This repository contains all files for the second assignment of the CS6910 - Fundamentals of Deep Learning course at IIT Madras.

## Contents

- [Task](#task)
- [Submission](#submission)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Running Manually](#running-manuallyn)
  - [Running a Sweep using Wandb](#running-a-sweep-using-wandb)
  - [Customization](#customization)

## Task

The task is to build and experiment with CNN based image classifiers.

## Submission

My WandB project: https://wandb.ai/ed23s037/CS6910_AS2/overview

My WandB report: https://wandb.ai/ed23s037/CS6910_AS2/reports/-CS6910-Assignment-2--Vmlldzo3NDM1Njcy

## Dataset

The dataset utilized for this assignment is `iNaturalist` dataset and can be used in the training like

```sh
wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
unzip nature_12K.zip
```

## Usage

### Running Manually

To train the model, use the following command (where x can be a and b):

```sh
$ python train_partx.py -wp <wandb_project_name> -we <wandb_entity_name>
```

You can also modify the following list of available options. To get the brief information about each:

```sh
$ python train.py -h
```

Similarly to evaluate the models use,

```sh
$ python eval_partx.py
```

The above code will generate the confusion matrix and will save the plot as a image.

### Customization

Refer to the `src` folder in both partA and partB for customizing the code. The `notebooks` folder has all the jupyter notebooks enlisting the different trails and base code for this implmentation.

</div>
