import torch.nn as nn
import torch.nn.functional as F
import torch


# class SoftDiceLoss(nn.Module):
#     def init(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).init()

#         def forward(self, logits, targets):
#             num = targets.size(0)
#             smooth = 1e-9
#             probs = F.sigmoid(logits)
#             m1 = probs.view(num, -1)
#             m2 = targets.view(num, -1)
#             intersection = (m1*m2)
#             score = 2.*(intersection.sum(1) + smooth) / \
#                 (m1.sum(1) + m2.sum(1) + smooth)

#             score = 1 - score.sum()/num

#             return score
# import torch
# from torch import Tensor
# import torch.nn.functional as F


# def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all batches, or for a single mask
#     assert input.size() == target.size()
#     if input.dim() == 2 and reduce_batch_first:
#         raise ValueError(
#             f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input.dim() == 2 or reduce_batch_first:
#         inter = torch.dot(input.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input) + torch.sum(target) 
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
#     else:
#         # compute and average metric for each batch element
#         dice = 0
#         for i in range(input.shape[0]):
#             dice += dice_coeff(input[i, ...], target[i, ...])
#         return dice / input.shape[0]


# def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all classes
#     assert input.size() == target.size()
#     dice = 0
#     for channel in range(input.shape[1]):
#         dice += dice_coeff(input[:, channel, ...], target[:,
#                            channel, ...], reduce_batch_first, epsilon)

#     return dice / input.shape[1]


# def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     assert input.size() == target.size()
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)
class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / (union+self.eps)
        #print("intersection", intersection, union, dice_score)
        return 1 - dice_score