import os
import torch
import tabulate
import torch.nn as nn


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def print_table(values, columns, epoch):
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train", "val", "test"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase == "train":
        model.train()
    elif phase == "val" or phase == "test":
        model.eval()

    ttl = 0
    with torch.autograd.set_grad_enabled(phase == "train"):
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase == "train":
                optimizer.zero_grad()
                loss = loss * 1000  # grad scaling
                loss.backward()
                optimizer.step()

    correct = correct.cpu().item()
    return {
        "loss": loss_sum / float(ttl),
        "accuracy": correct / float(ttl) * 100.0,
    }
