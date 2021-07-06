import torchvision
from torchvision import transforms

from .libsvm import LibSVM, DOWNLOAD_LINKS

libsvm_keys = list(DOWNLOAD_LINKS.keys())


def load_data(path, dataset, load_train_set=True, download=True):
    dataset = dataset.lower()
    train_set = None

    if dataset.startswith("cifar"):  # CIFAR-10/100
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        if dataset == "cifar10":
            if load_train_set:
                train_set = torchvision.datasets.CIFAR10(
                    root=path, train=True, download=download, transform=transform_train)
            test_set = torchvision.datasets.CIFAR10(
                root=path, train=False, download=download, transform=transform_test)
        elif dataset == "cifar100":
            if load_train_set:
                train_set = torchvision.datasets.CIFAR100(
                    root=path, train=True, download=download, transform=transform_train)
            test_set = torchvision.datasets.CIFAR100(
                root=path, train=False, download=download, transform=transform_test)
        else:
            raise NotImplementedError(f'{dataset} is not implemented.')
    elif dataset == 'mnist':
        transform = transforms.ToTensor()
        if load_train_set:
            train_set = torchvision.datasets.MNIST(
                root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root=path, train=False, download=True, transform=transform)

    elif dataset in libsvm_keys:
        test_set = LibSVM(root=path, dataset_name=dataset)
        if load_train_set:
            train_set = test_set
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    return train_set, test_set


def get_num_classes(dataset):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'mnist':
        num_classes = 10
    elif dataset in libsvm_keys:
        num_classes = 1
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    return num_classes
