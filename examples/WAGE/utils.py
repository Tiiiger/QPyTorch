import os
import torch
import models


def set_seed(seed, cuda):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out = 0.5 * ((logits - target) ** 2).sum()
    return out


def train_epoch(
    loader,
    model,
    criterion,
    weight_quantizer,
    grad_quantizer,
    epoch,
    quant_bias=True,
    quant_bn=True,
    log_error=False,
    wage_quantize=False,
    wage_grad_clip=None,
):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    ttl = 0

    for i, (input_v, target) in enumerate(loader):
        step = i + epoch * len(loader)
        # input is [0-1], scale to [-1,1]
        input_v = input_v.cuda()
        input_v = input_v * 2 - 1
        target = target.cuda()

        # WAGE quantize 8-bits accumulation into ternary before forward
        for name, param in model.named_parameters():
            param.data = weight_quantizer(
                model.weight_acc[name], model.weight_scale[name]
            )

        output = model(input_v)
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()

        # gradient quantization
        for name, param in list(model.named_parameters())[::-1]:
            param.grad.data = grad_quantizer(param.grad.data).data

            # WAGE accumulate weight in gradient precision
            # assume no batch norm
            w_acc = wage_grad_clip(model.weight_acc[name])
            w_acc -= param.grad.data
            model.weight_acc[name] = w_acc

        loss_sum += loss.cpu().item() * input_v.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        ttl += input_v.size(0)

    correct = correct.cpu().item()
    return {
        "loss": loss_sum / float(ttl),
        "accuracy": correct / float(ttl) * 100.0,
    }


def eval(loader, model, criterion, wage_quantizer=None):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    # WAGE quantize 8-bits accumulation into ternary before forward
    # assume no batch norm
    for name, param in model.named_parameters():
        param.data = wage_quantizer(model.weight_acc[name], model.weight_scale[name])

    cnt = 0
    with torch.no_grad():
        for i, (input_v, target) in enumerate(loader):
            input_v = input_v.cuda()
            input_v = input_v * 2 - 1
            target = target.cuda()

            output = model(input_v)
            loss = criterion(output, target)

            loss_sum += loss.data.cpu().item() * input_v.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            cnt += input_v.size(0)

    correct = correct.cpu().item()

    return {
        "loss": loss_sum / float(cnt),
        "accuracy": correct / float(cnt) * 100.0,
    }
