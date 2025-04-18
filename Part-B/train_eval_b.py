#!/usr/bin/env python3
"""
Fine-tune and evaluate an InceptionV3 model on the iNaturalist 12K dataset,
with live logging of metrics and model artifacts to Weights & Biases.

Usage:
    python inception_train_eval.py --mode train [options]
    python inception_train_eval.py --mode eval  [options]

Examples:
    # Initial training phase
    python train_eval_b.py --mode train \
        --epochs 100 --batch-size 32 \
        --learning-rate 5e-5 --weight-decay 1e-2 \
        --fc1 1024 --fc2 512 --classes 10 \
        --model-path models/fine_tuned_inception.pth

    # Evaluation on test set
    python train_eval_b.py --mode eval \
        --batch-size 16 \
        --model-path models/fine_tuned_inception.pth
"""
import os
import ssl
import certifi

# On Windows or environments with missing certs, use certifi bundle:
os.environ['SSL_CERT_FILE'] = certifi.where()
# Fallback: disable SSL verification if necessary:
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import gc
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import wandb

# Default dataset locations
TRAIN_DIR = "inaturalist_12K/train/"
VAL_DIR   = "inaturalist_12K/val/"

# Default WandB settings
WANDB_SETTINGS = {
    'project': 'Alik_Final_DA6401_DeepLearning_Assignment2',
    'entity':  'cs24m007-iit-madras'
}

class Config:
    def __init__(self, args):
        self.mode         = args.mode
        self.epochs       = args.epochs
        self.batch_size   = args.batch_size
        self.lr           = args.learning_rate
        self.weight_decay = args.weight_decay
        self.fc1_size     = args.fc1
        self.fc2_size     = args.fc2
        self.num_classes  = args.classes
        self.model_path   = args.model_path or 'models/fine_tuned_inception.pth'


def build_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def prepare_dataloaders(cfg):
    transform = build_transforms()
    full_ds   = datasets.ImageFolder(TRAIN_DIR, transform)
    train_len = int(0.8 * len(full_ds))
    val_len   = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def configure_model(cfg, device):
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

    # Replace the classifier
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, cfg.fc1_size),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.fc1_size, cfg.fc2_size),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.fc2_size, cfg.num_classes)
    )

    # Strategy 2: only train new classifier and Mixed_7c
    for name, param in model.named_parameters():
        param.requires_grad = ('fc' in name or 'Mixed_7c' in name)

    return model.to(device)


def train_one_phase(model, loaders, cfg, device, phase_name):
    train_loader, val_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for ep in range(cfg.epochs):
        # Training
        model.train()
        running_loss = running_correct = total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            logits = outs.logits if hasattr(outs, 'logits') else outs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = 100 * running_correct / total

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outs = model(images)
                logits = outs.logits if hasattr(outs, 'logits') else outs
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc  = 100 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log to Weights & Biases
        wandb.log({
            'phase':        phase_name,
            'epoch':        ep + 1,
            'train/loss':   train_loss,
            'train/accuracy': train_acc,
            'val/loss':     val_loss,
            'val/accuracy': val_acc
        }, step=ep + 1)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{phase_name}] Epoch {ep+1}/{cfg.epochs} — train_loss: {train_loss:.4f}, "
              f"train_acc: {train_acc:.2f}% | val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")

    return history


def train(cfg, device):
    # Initialize WandB
    wandb.init(
        project=WANDB_SETTINGS['project'],
        entity=WANDB_SETTINGS['entity'],
        config={
            'epochs':        cfg.epochs,
            'batch_size':    cfg.batch_size,
            'learning_rate': cfg.lr,
            'weight_decay':  cfg.weight_decay,
            'fc1_size':      cfg.fc1_size,
            'fc2_size':      cfg.fc2_size,
            'num_classes':   cfg.num_classes
        }
    )
    train_loader, val_loader = prepare_dataloaders(cfg)
    model = configure_model(cfg, device)
    wandb.watch(model, log='all', log_freq=50)

    # Phase 1: fc + Mixed_7c
    print("Phase 1: training classifier + Mixed_7c")
    hist1 = train_one_phase(model, (train_loader, val_loader), cfg, device, phase_name='phase1')

    # Phase 2: unfreeze Mixed_7b
    print("Phase 2: unfreezing Mixed_7b for fine-tuning")
    for name, param in model.named_parameters():
        if 'Mixed_7b' in name:
            param.requires_grad = True
    cfg.epochs = 5
    hist2 = train_one_phase(model, (train_loader, val_loader), cfg, device, phase_name='phase2')

    # Save and log model
    torch.save(model.state_dict(), cfg.model_path)
    artifact = wandb.Artifact('inceptionv3-finetuned', type='model')
    artifact.add_file(cfg.model_path)
    wandb.log_artifact(artifact)
    print(f"Model saved to {cfg.model_path} and logged to W&B.")


def evaluate(cfg, device):
    transform = build_transforms()
    test_set = datasets.ImageFolder(VAL_DIR, transform)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    # Rebuild model and load weights
    model = models.inception_v3(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, cfg.fc1_size),
        nn.ReLU(),
        nn.Linear(cfg.fc1_size, cfg.fc2_size),
        nn.ReLU(),
        nn.Linear(cfg.fc2_size, cfg.num_classes)
    )
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            logits = outs.logits if hasattr(outs, 'logits') else outs
            preds = logits.argmax(1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = sum(t==p for t,p in zip(y_true,y_pred)) / len(y_true)
    print(f"Test accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm / cm.sum(axis=1)[:, None],
                         index=test_set.classes,
                         columns=test_set.classes)
    plt.figure(figsize=(10, 8))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',         choices=['train','eval'], required=True)
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch-size',   type=int,   default=128)
    parser.add_argument('--learning-rate',type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--fc1',          type=int,   default=1024)
    parser.add_argument('--fc2',          type=int,   default=512)
    parser.add_argument('--classes',      type=int,   default=10)
    parser.add_argument('--model-path',   type=str,   default=None,
                        help='Path to save or load the model state_dict')
    args = parser.parse_args()

    cfg = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.mode == 'train':
        train(cfg, device)
    else:
        evaluate(cfg, device)

if __name__ == '__main__':
    main()
