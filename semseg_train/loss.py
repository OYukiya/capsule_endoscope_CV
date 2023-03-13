import torch
from torch import nn
import torch.nn.functional as F

def soft_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_target = (targets == 1).float()
    jaccard_output = torch.sigmoid(outputs)

    intersection = (jaccard_output * jaccard_target).sum()
    union = jaccard_output.sum() + jaccard_target.sum()
    return intersection / (union - intersection + eps)

def structure_loss(pred, mask):
    #重み付きloss
    #ref:https://github.com/DengPingFan/PraNet
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()#the sum of the output will be divided by the number of elements(sum data) in the output
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            loss += self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
        return loss

class LossBCE:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss(reduction='sum')#1ピクセル当たりのBCE（実数1つを返す）、バッチ数で割り画像サイズ縦*横で割る
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        return loss

class LossJaccard:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
        return loss

class LossWeightBCE:
    """
    Loss defined as WeightBCE

    Deng-Ping Fan, Ge-Peng Ji, Tao Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, and Ling Shao,
    PraNet: Parallel Reverse Attention Network for Polyp Segmentation
    """

    #def __init__(self, jaccard_weight=0):
        #self.nll_loss = nn.BCEWithLogitsLoss()
        #self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = structure_loss(outputs, targets)
        return loss