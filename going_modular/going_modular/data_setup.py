"""
Contains functions for creating PyTorch Datasets and DataLoaders for the
Face Landmarks dataset.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


NUM_WORKERS = os.cpu_count()

BATCH_SIZE = 1

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
    ):

  """ Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path.

  Returns training and testing DataLoaders.

  Args:
    train_dir : Path to training directory.
    test_dir : Path to testing directory.
    transform : torchvision transforms to perform on the images.
    batch_size : Number of samples per batch in the training and testing DataLoaders.
    test_batch_size : Number of samples per batch in the testing DataLoader.
  """
  # Use ImageFolder to create dataset
  train_data = datasets.ImageFolder(root=train_dir,
                                    transform=transform)

  test_data = datasets.ImageFolder(root=test_dir,
                                   transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into dataloader
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                shuffle=True,
                                pin_memory=True)

  test_dataloader = DataLoader(dataset=test_data,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               shuffle=False,
                               pin_memory=True)

  return train_dataloader, test_dataloader , class_names
