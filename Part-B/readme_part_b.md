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



## Usage
- Open this in kaggle and collab
- Take script of train_part_b.py from partB
```bash
python train_part_b.py
```
## Results
- Training logs and metrics are tracked using WandB (Weights & Biases).
- Evaluation results on the test set are printed after training completion.
