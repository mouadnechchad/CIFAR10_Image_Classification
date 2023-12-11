import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt


def import_and_merge_cifar10():
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
    ])

    # Import CIFAR-10 training set
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Import CIFAR-10 validation set
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Merge training and validation sets
    merged_dataset = ConcatDataset([train_dataset, val_dataset])

    return merged_dataset

def visualize_images(images, labels, class_names=None, num_samples=5):
    """
    Visualize a set of images along with their labels.

    Parameters:
    - images: List of image tensors or numpy arrays.
    - labels: List of corresponding labels.
    - class_names: List of class names (optional).
    - num_samples: Number of samples to visualize.

    Returns:
    None
    """
    num_samples = min(num_samples, len(images))

    plt.figure(figsize=(15, 3))

    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        image = images[i]
        label = labels[i]

        # If the image is a torch tensor, convert it to a numpy array
        if torch.is_tensor(image):
            image = image.numpy()

        # If the image has three channels, remove the channel dimension for display
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        # Normalize pixel values to the range [0, 1] for floats or [0, 255] for integers
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.float32:
            image = np.clip(image, 0.0, 1.0)

        plt.imshow(image)
        plt.title(f"Label: {label}")

        if class_names:
            plt.xlabel(f"Class: {class_names[label]}")

        plt.axis("off")

    plt.show()

def kfold_data_split(dataset, batch_size=32, num_folds=6, shuffle=True, random_seed=None):
    """
    Split a dataset into training and validation sets using K-Fold method.

    Parameters:
    - dataset: PyTorch dataset object.
    - batch_size: Batch size for DataLoader.
    - num_folds: Number of folds for K-Fold cross-validation.
    - shuffle: Whether to shuffle the indices before splitting.
    - random_seed: Random seed for reproducibility.

    Returns:
    - List of DataLoader objects for each fold: [train_loader_1, val_loader_1, ..., train_loader_6, val_loader_6]
    """
    # Get the indices for the dataset
    indices = list(range(len(dataset)))

    # Initialize KFold with the specified number of folds
    kfold = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_seed)

    # Initialize a list to store DataLoader objects for each fold
    dataloaders = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        # Create SubsetRandomSampler for training and validation indices
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Create DataLoader for training and validation sets
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        dataloaders.extend([train_loader, val_loader])

    return dataloaders
def create_cross_entropy_loss():
    """
    Create Cross Entropy Loss function.

    Returns:
    - Cross Entropy Loss function
    """
    return nn.CrossEntropyLoss()
def create_adam_optimizer(model, learning_rate=0.001, weight_decay=0.01):
    """
    Create Adam optimizer with optional L2 regularization.

    Parameters:
    - model: PyTorch model for optimization.
    - learning_rate: Learning rate for the optimizer (default: 0.001).
    - weight_decay: L2 regularization strength (default: 0.01).

    Returns:
    - Adam optimizer
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def train_and_evaluate(model, train_dataloader, val_dataloader, num_episodes, criterion, optimizer):
    """
    Train and evaluate a PyTorch model over a fixed number of episodes.

    Parameters:
    - model: PyTorch model to be trained and evaluated.
    - train_dataloader: DataLoader for the training dataset.
    - val_dataloader: DataLoader for the validation dataset.
    - num_episodes: Number of training episodes.
    - learning_rate: Learning rate for the optimizer (default: 0.001).
    - weight_decay: L2 regularization strength (default: 0.01).

    Returns:
    - List of mean training losses per episode.
    - List of mean training accuracies per episode.
    - List of mean validation losses per episode.
    - List of mean validation accuracies per episode.
    """

    mean_train_losses = []
    mean_train_accuracies = []
    mean_val_losses = []
    mean_val_accuracies = []

    for episode in range(1, num_episodes + 1):
        # Training
        model.train()
        train_losses = []
        correct_train = 0
        total_train = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        mean_train_loss = sum(train_losses) / len(train_losses)
        mean_train_accuracy = correct_train / total_train

        mean_train_losses.append(mean_train_loss)
        mean_train_accuracies.append(mean_train_accuracy)

        # Validation
        model.eval()
        val_losses = []
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            mean_val_loss = sum(val_losses) / len(val_losses)
            mean_val_accuracy = correct_val / total_val

            mean_val_losses.append(mean_val_loss)
            mean_val_accuracies.append(mean_val_accuracy)

        # Print the progress for each episode
        print(f"Episode {episode}/{num_episodes}: "
              f"Train Loss: {mean_train_loss:.4f}, Train Accuracy: {mean_train_accuracy:.4f}, "
              f"Val Loss: {mean_val_loss:.4f}, Val Accuracy: {mean_val_accuracy:.4f}")

    return mean_train_losses, mean_train_accuracies, mean_val_losses, mean_val_accuracies

def plot_training_progress(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Plot the training progress (loss and accuracy) over episodes.

    Parameters:
    - train_losses: List of mean training losses per episode.
    - train_accuracies: List of mean training accuracies per episode.
    - val_losses: List of mean validation losses per episode.
    - val_accuracies: List of mean validation accuracies per episode.

    Returns:
    None
    """
    episodes = range(1, len(train_losses) + 1)

    # Plot Training Loss and Validation Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, train_losses, label='Training Loss')
    plt.plot(episodes, val_losses, label='Validation Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Training Accuracy and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(episodes, train_accuracies, label='Training Accuracy')
    plt.plot(episodes, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Episodes')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()