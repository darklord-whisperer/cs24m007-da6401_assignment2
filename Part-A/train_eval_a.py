import os
import sys
import random
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import wandb

# Device configuration: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable benchmark mode in cuDNN for potentially faster runtime
torch.backends.cudnn.benchmark = True

# ---------------------------
# Data Loading and Augmentation
# ---------------------------
def load_data(train_dir, test_dir, augment=False, train_batch_size=16, num_workers=8):
    """
    Loads images from train and test directories, splits train into train/val,
    applies transformations and returns DataLoaders.
    """
    # Check directories exist
    if not os.path.isdir(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist.")
        sys.exit(1)

    # Basic transforms: resize, center crop, to tensor, normalize
    basic_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Augmentation transforms: add rotation and flip
    augment_transforms = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset to split manually
    dataset = datasets.ImageFolder(root=train_dir, transform=basic_transforms)
    # Group indices by class for stratified split
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
    # Shuffle and split each class
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split = int(0.2 * len(indices))
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    # Create subsets for train and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # If augmentation enabled, create augmented dataset and concat
    if augment:
        augmented_dataset = datasets.ImageFolder(root=train_dir, transform=augment_transforms)
        train_aug_subset = Subset(augmented_dataset, train_indices)
        combined_train = ConcatDataset([train_subset, train_aug_subset])
        train_loader = DataLoader(
            combined_train, batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
    else:
        # No augmentation: just use train subset
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )

    # Validation loader (shuffled)
    val_loader = DataLoader(
        val_subset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    # Test loader (no shuffle)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=basic_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader, test_loader

# ---------------------------
# CNN Model Definition
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, filter_sizes,
                 activation, dense_neurons, dropout_rate, use_batchnorm, filter_multiplier):
        super(SimpleCNN, self).__init__()
        # Map string to activation layer
        activations = {
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU(),
            'SiLU': nn.SiLU(),
            'Mish': nn.Mish()
        }
        act_layer = activations[activation]

        layers = []
        in_ch, out_ch = input_channels, num_filters
        # Build 5 convolutional blocks
        for i in range(5):
            layers.append(nn.Conv2d(
                in_ch, out_ch,
                kernel_size=filter_sizes[i],
                stride=1, padding=1
            ))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act_layer)
            layers.append(nn.MaxPool2d(2, 2))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            # Increase channels by multiplier for next block
            in_ch, out_ch = out_ch, int(out_ch * filter_multiplier)

        self.feature_extractor = nn.Sequential(*layers)

        # Determine flattened feature size by a dummy forward pass
        dummy = torch.randn(1, input_channels, 32, 32)
        with torch.no_grad():
            feat = self.feature_extractor(dummy)
        flat_dim = feat.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(flat_dim, dense_neurons)
        self.act_fc = act_layer
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.act_fc(self.fc1(x))
        return self.fc2(x)

# ---------------------------
# Training and Evaluation
# ---------------------------
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(loader, desc="Training", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        # Mixed precision for speed
        with torch.cuda.amp.autocast():
            outs = model(imgs)
            loss = criterion(outs, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = outs.argmax(dim=1)
        total += lbls.size(0)
        correct += (preds == lbls).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def validate_epoch(model, loader, criterion):
    model.eval()
    v_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            v_loss += criterion(outs, lbls).item()
            preds = outs.argmax(dim=1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()
    avg_loss = v_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def test_model(model, loader):
    """
    Runs final test set evaluation and prints test accuracy.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def plot_predictions(model, loader, class_names):
    """
    Visualizes up to 3 correct/incorrect predictions per class.
    """
    model.eval()
    fig, axes = plt.subplots(10, 3, figsize=(12, 40))
    count = {i: 0 for i in range(len(class_names))}
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1)
            for img, actual, pred in zip(imgs, lbls, preds):
                if count[actual] < 3:
                    ax = axes[actual, count[actual]]
                    ax.imshow(np.transpose(img.cpu(), (1, 2, 0)))
                    ax.set_title(f"Act: {class_names[actual]}\nPred: {class_names[pred]}")
                    ax.axis('off')
                    count[actual] += 1
                # Stop once we have 3 examples per class
                if all(v >= 3 for v in count.values()):
                    plt.tight_layout()
                    plt.show()
                    return

# ---------------------------
# Aggregation Function for Sweep Results
# ---------------------------
def aggregate_sweep_results(sweep_id, project, entity):
    """
    Fetches all runs belonging to a sweep, aggregates metrics,
    and logs combined plots back to WandB as an artifact.
    """
    run = wandb.init(project=project, entity=entity,
                     job_type="aggregation", reinit=True)
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}",
                    filters={"sweep": sweep_id})
    if not runs:
        print("No runs found for this sweep.")
        run.finish()
        return

    records = []
    for r in runs:
        sum_json = r.summary._json_dict
        cfg = r.config
        # Try to get final validation accuracy first
        final_acc = sum_json.get("Validation Accuracy",
                                 sum_json.get("Train Accuracy"))
        records.append({
            "run_id": r.id,
            "num_filters": cfg.get("num_filters"),
            "activation": cfg.get("activation"),
            "dense_neurons": cfg.get("dense_neurons"),
            "dropout": cfg.get("dropout"),
            "batch_norm": cfg.get("batch_norm"),
            "filter_multiplier": cfg.get("filter_multiplier"),
            "learning_rate": cfg.get("learning_rate"),
            "epochs": cfg.get("epochs"),
            "data_augmentation": cfg.get("data_augmentation"),
            "Validation Accuracy": final_acc
        })

    df = pd.DataFrame(records)
    df.sort_values("Validation Accuracy", ascending=False,
                   inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Experiment"] = df.index + 1

    # Plot 1: Accuracy vs Experiment
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Experiment"], df["Validation Accuracy"],
             marker='o')
    ax1.set(xlabel="Experiment",
            ylabel="Validation Accuracy",
            title="Accuracy vs Experiment")
    plt.tight_layout()

    # Plot 2: Parallel Coordinates for hyperparameters
    df_p = df.copy()
    df_p["Validation Accuracy"] = df_p[
        "Validation Accuracy"].round(2).astype(str)
    from pandas.plotting import parallel_coordinates
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    parallel_coordinates(
        df_p,
        class_column="Validation Accuracy",
        cols=[
            "num_filters", "dense_neurons", "dropout",
            "filter_multiplier", "learning_rate", "epochs",
            "Validation Accuracy"
        ],
        ax=ax2
    )
    plt.title("Parallel Coordinates")
    plt.tight_layout()

    # Plot 3: Correlation Heatmap of parameters vs accuracy
    num_cols = [
        "num_filters", "dense_neurons", "dropout",
        "filter_multiplier", "learning_rate", "epochs",
        "Validation Accuracy"
    ]
    corr = df[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sn.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    ax3.set_title("Correlation Heatmap")
    plt.tight_layout()

    # Save and log as WandB artifact
    fig1.savefig("accuracy_vs_experiment.png")
    fig2.savefig("parallel_coordinates.png")
    fig3.savefig("correlation_heatmap.png")
    art = wandb.Artifact("sweep_aggregated_results", type="plots")
    art.add_file("accuracy_vs_experiment.png")
    art.add_file("parallel_coordinates.png")
    art.add_file("correlation_heatmap.png")
    wandb.log_artifact(art)
    print("Aggregated results logged.")
    run.finish()

# ---------------------------
# Main Training Function
# ---------------------------
def main(args):
    """
    Initializes WandB run, loads data, builds model,
    runs training/validation epochs, evaluates test set,
    and plots sample predictions.
    """
    if wandb.run is None:
        wandb.init(
            project="Alik_Final_DA6401_DeepLearning_Assignment2",
            entity="cs24m007-iit-madras",
            config=vars(args)
        )

    # Load data loaders
    train_loader, val_loader, test_loader = load_data(
        args.train_dir, args.test_dir,
        augment=(args.data_augmentation == "Yes")
    )
    print(f"Train len: {len(train_loader.dataset)}, "
          f"Val len: {len(val_loader.dataset)}, "
          f"Test len: {len(test_loader.dataset)}")

    # Instantiate model and move to device
    model = SimpleCNN(
        input_channels=3, num_classes=10,
        num_filters=args.num_filters,
        filter_sizes=[
            args.filter_size_1, args.filter_size_2,
            args.filter_size_3, args.filter_size_4,
            args.filter_size_5
        ],
        activation=args.activation,
        dense_neurons=args.dense_neurons,
        dropout_rate=args.dropout,
        use_batchnorm=(args.batch_norm == "Yes"),
        filter_multiplier=args.filter_multiplier
    )
    model.to(device)

    # If multiple GPUs, wrap model for parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss, optimizer, scaler for mixed precision
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Optionally script model for speed
    model = torch.jit.script(model)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion
        )
        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss={train_loss:.4f}, "
            f"Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Acc={val_acc:.2f}%"
        )
        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc
        })

    # Final test evaluation and sample plots
    test_model(model, test_loader)
    class_names = {i: f"Class{i}" for i in range(10)}
    plot_predictions(model, test_loader, class_names)
    wandb.finish()

# ---------------------------
# Sweep Run Function
# ---------------------------
def sweep_run():
    """
    Callback for wandb.agent to execute a single run
    with parameters sampled from the sweep.
    """
    run = wandb.init(
        project="Alik_Final_DA6401_DeepLearning_Assignment2",
        entity="cs24m007-iit-madras"
    )
    config = run.config
    args = argparse.Namespace(**vars(config))
    main(args)

# ---------------------------
# Sweep Configuration
# ---------------------------
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
    'parameters': {
        'num_filters': {'values': [32, 64, 128]},
        'filter_size_1': {'value': 3}, 'filter_size_2': {'value': 3},
        'filter_size_3': {'value': 3}, 'filter_size_4': {'value': 3},
        'filter_size_5': {'value': 3},
        'activation': {'values': ['ReLU', 'GELU', 'SiLU', 'Mish']},
        'dense_neurons': {'values': [64, 128, 256]},
        'dropout': {'values': [0, 0.2, 0.3]},
        'batch_norm': {'values': ['Yes', 'No']},
        'filter_multiplier': {'values': [1, 2, 0.5]},
        'learning_rate': {'values': [0.001, 0.0001]},
        'epochs': {'values': [10, 20]},
        'train_dir': {'value': None}, 'test_dir': {'value': None},
        'data_augmentation': {'values': ['Yes', 'No']}
    }
}

# ---------------------------
# Argument Parser
# ---------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train CNN with wandb sweep and aggregation"
    )
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--filter_size_1', type=int, default=3)
    parser.add_argument('--filter_size_2', type=int, default=3)
    parser.add_argument('--filter_size_3', type=int, default=3)
    parser.add_argument('--filter_size_4', type=int, default=3)
    parser.add_argument('--filter_size_5', type=int, default=3)
    parser.add_argument(
        '--activation', type=str, default='ReLU',
        choices=['ReLU', 'GELU', 'SiLU', 'Mish']
    )
    parser.add_argument('--dense_neurons', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument(
        '--batch_norm', type=str, default='No',
        choices=['Yes', 'No']
    )
    parser.add_argument('--filter_multiplier', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir',  type=str, required=True)
    parser.add_argument(
        '--data_augmentation', type=str, default='No',
        choices=['Yes', 'No']
    )
    parser.add_argument('--sweep',      action='store_true')
    parser.add_argument('--aggregate',  action='store_true')
    parser.add_argument('--sweep_id',   type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.aggregate:
        if not args.sweep_id:
            print("Please provide --sweep_id for aggregation.")
        else:
            aggregate_sweep_results(args.sweep_id,
                                    project="Alik_Final_DA6401_DeepLearning_Assignment2",
                                    entity="cs24m007-iit-madras")
    else:
        if args.sweep:
            # Set data paths for sweep
            sweep_config['parameters']['train_dir']['value'] = args.train_dir
            sweep_config['parameters']['test_dir']['value']  = args.test_dir
            sweep_id = wandb.sweep(sweep_config,
                                   project="Alik_Final_DA6401_DeepLearning_Assignment2")
            print(f"Sweep ID: {sweep_id}")
            wandb.agent(sweep_id, function=sweep_run)
        else:
            main(args)
