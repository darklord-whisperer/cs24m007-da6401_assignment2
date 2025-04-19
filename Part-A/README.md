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
