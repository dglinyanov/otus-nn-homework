import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_dataset_mean_std(dataset):
    loader = utils.data.DataLoader(
        dataset,
        batch_size=1000,
        num_workers=4,
        shuffle=False
    )


    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, data_y in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    return mean, std


mnist_pre_transform = transforms.Compose([
    transforms.ToTensor()
])

def mnist(batch_size=50, shuffle=True, transform=mnist_pre_transform, path='./MNIST_data'):
    train_data = datasets.MNIST(path, train=True, download=True, transform=transform)
    train_mean, train_std = get_dataset_mean_std(train_data)
    print('Train mean and std', train_mean, train_std)
    mnist_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std,)
    ])
    # Using same transform for both sets
    train_data = datasets.MNIST(path, train=True, download=True, transform=mnist_train_transform)
    test_data = datasets.MNIST(path, train=False, download=True, transform=mnist_train_transform)
    
    train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader

def plot_mnist(images, shape):
    fig = plt.figure(figsize=shape[::-1], dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[0], shape[1], j)
        ax.matshow(images[j - 1, 0, :, :], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()