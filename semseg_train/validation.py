import numpy as np
import utils
from torch import nn


def validation_binary(model: nn.Module, criterion, valid_loader):
    model.eval()
    losses = []

    jaccard = []
    dice = []

    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        #invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
        #dataに0番目の配列がない、代わりに.item()
        #losses.append(loss.data[0])
        losses.append(loss.data.item())
        outputs = (outputs > 0).float()
        jaccard += get_jaccard(targets, outputs)
        dice += get_dice(targets, outputs)

    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard).astype(np.float64)
    valid_dice = np.mean(dice).astype(np.float64)

    print('Valid loss: {:.5f}, jaccard: {:.5f}, dice: {:.5f}'.format(valid_loss, valid_jaccard, valid_dice))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard, 'dice_loss': valid_dice}
    return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list((intersection / (union + epsilon - intersection)).data.cpu().numpy())


def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)
    return list((2 * intersection / (union + epsilon)).data.cpu().numpy())
