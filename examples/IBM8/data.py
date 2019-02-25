import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def get_data(dataset, data_path, batch_size, val_ratio, num_workers):
    assert dataset in ["CIFAR10", "IMAGENET12"], "dataset not supported {}".format(dataset)
    assert val_ratio >= 0, "invalid validation ratio: {}".format(val_ratio)
    print('Loading dataset {} from {}'.format(dataset, data_path))
    if dataset=="CIFAR10":
        ds = getattr(datasets, dataset)
        path = os.path.join(data_path, dataset.lower())
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = ds(path, train=True, download=True, transform=transform_train)
        val_set = ds(path, train=True, download=True, transform=transform_test)
        test_set = ds(path, train=False, download=True, transform=transform_test)
        train_sampler = None
        val_sampler = None
        num_classes = 10
    elif dataset=="IMAGENET12":
        traindir = os.path.join(data_path, dataset.lower(), 'train')
        valdir = os.path.join(data_path, dataset.lower(), 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        test_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        train_sampler = None
        val_sampler = None
        num_classes = 1000
    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    return loaders
