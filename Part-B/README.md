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
- **remove "%%writefile train_part_b.py" 1st line from the python script and then run in Google collab:**
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
