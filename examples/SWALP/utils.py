import os
import torch
import tabulate
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from time import perf_counter


def set_seed(seed, cuda):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def schedule(epoch, lr_init, swa_start, swa_lr):
    if epoch < swa_start:
        t = (epoch) / swa_start
        lr_ratio = 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return lr_init * factor
    else:
        return swa_lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase == "train":
        model.train()
    elif phase == "eval":
        model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ttl = 0
    start = perf_counter()
    with torch.autograd.set_grad_enabled(phase == "train"):
        for i, (input, target) in enumerate(loader):
            input = input.to(device=device)
            target = target.to(device=device)
            output = model(input)
            loss = criterion(output, target)

            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    elapse = perf_counter() - start
    correct = correct.cpu().item()
    res = {
        "loss": loss_sum / float(ttl),
        "accuracy": correct / float(ttl) * 100.0,
    }
    if phase == "train":
        res["train time"] = elapse

    return res


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def print_table(columns, values, epoch):
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


num_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
}


def get_data(dataset, data_path, batch_size, num_workers):
    assert dataset in ["CIFAR10", "CIFAR100"]
    print("Loading dataset {} from {}".format(dataset, data_path))
    if dataset in ["CIFAR10", "CIFAR100"]:
        ds = getattr(datasets, dataset.upper())
        path = os.path.join(data_path, dataset.lower())
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_set = ds(path, train=True, download=True, transform=transform_train)
        val_set = ds(path, train=True, download=True, transform=transform_test)
        test_set = ds(path, train=False, download=True, transform=transform_test)
        train_sampler = None
        val_sampler = None
    else:
        raise Exception("Invalid dataset %s" % dataset)

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
