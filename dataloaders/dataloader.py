from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100(batch_size=64, data_dir='./data', num_workers=4, pin_memory=False):
    """
    Load CIFAR-100 training and test DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory to download/load the data.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): If True, DataLoader will copy Tensors into CUDA pinned memory.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    # CIFAR-100 normalization statistics
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Only normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Create datasets
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, test_loader

def get_cifar10(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def get_mnist(batch_size=128, num_workers=2, data_root='./data'):
    """Load MNIST train/test dataloaders."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST mean/std
    ])
    
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader
