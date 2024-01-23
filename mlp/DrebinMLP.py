import os
import shutil

import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DrebinMLP(nn.Module):
    def __init__(self, input_size):
        super(DrebinMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        return self.layers(x)


def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer=None):
    """
    Load a model and (optionally) an optimizer from a checkpoint.

    Args:
    - checkpoint_fpath (str): Path to the saved checkpoint.
    - model (torch.nn.Module): Model to load the checkpoint into.
    - optimizer (torch.optim.Optimizer, optional): Optimizer to load from the checkpoint. Default is None.

    Returns:
    - model, optimizer (if provided), epoch value
    """
    # Check for the checkpoint file's existence
    if not os.path.exists(checkpoint_fpath):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_fpath}'!")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_fpath,
                            map_location=lambda storage, loc: storage)  # Ensure the loading is device agnostic

    # Validate the checkpoint keys
    if 'state_dict' not in checkpoint:
        raise KeyError("No state_dict found in checkpoint file!")
    if optimizer and 'optimizer' not in checkpoint:
        raise KeyError("No optimizer state found in checkpoint file!")

    # Load model state
    model.load_state_dict(checkpoint['state_dict'])

    # If an optimizer is provided, and its state is saved, load it
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Get epoch number if it's available, else return None
    epoch = checkpoint.get('epoch', None)

    return model, optimizer, epoch


def mlp_predict(test_loader, model, best):
    # Load Best Model Weights
    _, _, _ = load_checkpoint(best, model, None)

    model.eval()
    predictions = []
    labels_full = []
    X = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            X.extend(inputs)
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            predictions.extend(prediction.tolist())
            labels_full.extend(labels.tolist())

    return predictions, labels_full, X


def mlp_train_model(train_loader, cal_loader, model, optimizer, criterion, epochs, best, last):
    # Metrics initialization

    accuracy = Accuracy().to(device=device)
    f1score = F1Score(num_classes=2, average='macro').to(device=device)
    precision = Precision(num_classes=2, average='macro').to(device=device)
    recall = Recall(num_classes=2, average='macro').to(device=device)

    valid_max_f1score = 0.0

    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predictions = torch.max(outputs.data, 1)

            accuracy(predictions, labels)
            f1score(predictions, labels)
            precision(predictions, labels)
            recall(predictions, labels)

        accuracy.reset()
        f1score.reset()
        precision.reset()
        recall.reset()

        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in cal_loader:
                inputs, labels = inputs.to(device=device), labels.to(device=device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                accuracy(predictions, labels)
                f1score(predictions, labels)
                recall(predictions, labels)
                precision(predictions, labels)

            # Save model if it's the best
            checkpoint = {
                'epoch': epoch + 1,
                'accuracy': accuracy.compute(),
                'f1score': f1score.compute(),
                'precision': precision.compute(),
                'recall': recall.compute(),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            is_best_model = False
            if checkpoint['f1score'] > valid_max_f1score:
                is_best_model = True
                valid_max_f1score = checkpoint['f1score']

            save_checkpoint(checkpoint, is_best_model, last, best)
            print(f'Epoch: {epoch + 1}, '
                  f'Accuracy: {accuracy.compute()}, '
                  f'F1: {f1score.compute()}, '
                  f'Precision: {precision.compute()}, '
                  f'Recall: {recall.compute()}')
