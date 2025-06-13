from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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