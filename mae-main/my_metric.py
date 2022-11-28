import numpy as np
import torch
def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)

import numpy as np
from numba import jit
# jit编译器编译执行python代码，不使用python解析器，编译执行加速计算速度
# @jit(nopython=True)
def cal_mIoU(seg, gt, classes=2, background_id=-1):
    channel_iou = []
    for i in range(classes):
        if i == background_id:
            continue
        cond = i ** 2
        # 计算相交部分
        inter = len(np.where(seg * gt == cond)[0])
        union = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0]) - inter
        if union == 0:
            iou = 0
        else:
            iou = inter / union
        channel_iou.append(iou)
    res = np.array(channel_iou).mean()
    return res

# @jit(nopython=True)
def cal_dice(seg, gt, classes=2, background_id=-1):
    channel_dice = []
    for i in range(classes):
        if i == background_id:
            continue
        cond = i ** 2
        # 计算相交部分
        inter = len(np.where(seg * gt == cond)[0])
        total_pix = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0])
        if total_pix == 0:
            dice = 0
        else:
            dice = (2 * inter) / total_pix
        channel_dice.append(dice)
    res = np.array(channel_dice).mean()
    return res
