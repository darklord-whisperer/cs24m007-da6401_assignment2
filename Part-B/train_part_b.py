%%writefile train_b.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl           # original import for potential matplotlib backends
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# plotting & confusion matrix utilities
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# pretrained model applications
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet_v2 import MobileNetV2

print("TF version:", tf.__version__)

# ─── Dataset download and directory setup ───────────────────────────────────────
dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
dataset_dir = tf.keras.utils.get_file(
    fname="nature_12K",
    origin=dataset_url,
    cache_dir='.',
    extract=True
)

trainset_dir = './datasets/inaturalist_12K/train/'
testset_dir  = './datasets/inaturalist_12K/val/'
classlist    = [
    d for d in os.listdir(trainset_dir)
    if os.path.isdir(os.path.join(trainset_dir, d))
]

# ─── Function: prepare train/validation generators ─────────────────────────────
def generate_batch_train_val(path, augmentation, batch_size, image_size):
    """
    Creates ImageDataGenerators for training and validation splits.
    Applies data augmentation if specified.
    """
    if augmentation:
        datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=30,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            validation_split=0.1,
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.1
        )

    train_gen = datagen.flow_from_directory(
        path,
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
        seed=0,
        subset="training"
    )
    val_gen = datagen.flow_from_directory(
        path,
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
        seed=0,
        subset="validation"
    )

    labels = list(train_gen.class_indices.keys())
    return train_gen, val_gen, labels

# ─── Function: prepare test generator ──────────────────────────────────────────
def generate_batch_test(path, batch_size, image_size):
    """
    Creates a test ImageDataGenerator without shuffling.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        path,
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
        seed=0
    )
    return test_gen

# ─── Main training function ────────────────────────────────────────────────────
def train(config=None):
    """
    Builds, trains, fine-tunes, and plots metrics for the model.
    """
    # unpack configuration
    batch_size   = config['batch_size']
    augmentation = config['augmentation']
    base_name    = config['pretrain_model']
    drop_rate    = config['droprate']
    use_bn       = config['batch_normalization']
    epochs_total = config['epoch']
    fc_units     = config['fc_size']
    n_unfreeze   = config['num_of_trainable_layers']

    # ─── Select and freeze the pretrained base model layers ────────────────────
    if base_name == 'InceptionV3':
        img_size = (299, 299)
        base = InceptionV3(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    elif base_name == 'InceptionResNetV2':
        img_size = (299, 299)
        base = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    elif base_name == 'ResNet50':
        img_size = (224, 224)
        base = ResNet50(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    elif base_name == 'Xception':
        img_size = (299, 299)
        base = Xception(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    else:  # MobileNetV2
        img_size = (224, 224)
        base = MobileNetV2(include_top=False, weights='imagenet', input_shape=img_size + (3,))

    base.trainable = False  # freeze all pretrained layers

    # ─── Build classifier head on top of base ──────────────────────────────────
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=img_size + (3,)))
    model.add(base)
    model.add(Flatten())
    model.add(Dense(fc_units, activation='relu'))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(Dense(fc_units, activation='relu'))
    model.add(Dropout(drop_rate))

    # ─── Prepare data generators ────────────────────────────────────────────────
    train_gen, val_gen, class_labels = generate_batch_train_val(
        trainset_dir, augmentation, batch_size, img_size
    )
    model.add(Dense(len(class_labels), activation='softmax'))

    # ─── Compile and run initial training (feature extraction) ────────────────
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # decide split between feature extraction and fine‑tuning
    if n_unfreeze > 0:
        initial_epochs = epochs_total // 2
    else:
        initial_epochs = epochs_total

    hist1 = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen
    )

    # ─── Fine‑tuning block: feature extraction with frozen layers, full fine‑tuning of the entire model, selective fine‑tuning up to a specific layer, and progressive unfreezing—each consistently ─────────
    if n_unfreeze > 0:
        # progressively unfreeze the last n_unfreeze layers
        layers_to_unfreeze = n_unfreeze + (len(model.layers) - len(base.layers))
        for layer in reversed(model.layers):
            if layers_to_unfreeze > 0:
                layer.trainable = True
                layers_to_unfreeze -= 1

        # recompile with lower learning rate for fine‑tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        hist2 = model.fit(
            train_gen,
            epochs=epochs_total - initial_epochs,
            validation_data=val_gen
        )

        # merge histories so plots reflect all epochs
        for key in hist1.history:
            hist1.history[key].extend(hist2.history[key])
        history = hist1
    else:
        history = hist1

    # ─── Plot Train & Validation Accuracy ─────────────────────────────────────
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, history.history['accuracy'],    label='Train Acc')
    plt.plot(epochs_range, history.history['val_accuracy'],label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('train_val_accuracy.png')
    plt.close()

    # ─── Plot Train & Validation Loss ─────────────────────────────────────────
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, history.history['loss'],    label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'],label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('train_val_loss.png')
    plt.close()

    # ─── Plot Epoch Progression Curve ─────────────────────────────────────────
    plt.figure(figsize=(8,5))
    steps = np.arange(len(epochs_range))
    plt.plot(steps, list(epochs_range), marker='o')
    plt.title('Epoch Progression')
    plt.xlabel('Step (Epoch Index)'); plt.ylabel('Epoch Number')
    plt.grid(True); plt.tight_layout()
    plt.savefig('epoch_curve.png')
    plt.close()

    # ─── Plot Normalized Confusion Matrix (non‑shuffled) ───────────────────────
    cm_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        testset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False,
        seed=0
    )
    preds = model.predict(cm_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = cm_gen.classes

    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    plt.figure(figsize=(12,9))
    sns.heatmap(
        cm,
        annot=True, fmt='.1f', cmap='magma',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label':'Percent (%)'}
    )
    plt.title('Normalized Confusion Matrix (%)')
    plt.xlabel('Predicted Class'); plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# ─── Configuration & execution ────────────────────────────────────────────────
config = {
    'pretrain_model': 'InceptionV3',
    'epoch': 9,
    'batch_size': 16,
    'augmentation': True,
    'fc_size': 256,
    'droprate': 0.4,
    'batch_normalization': True,
    'num_of_trainable_layers': 1
}
train(config)
