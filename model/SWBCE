import torch.nn as nn
import torch
import torch.nn.functional as F
from metric_tool import SegEvaluator

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(WeightedBCEWithLogitsLoss,self).__init__()

    def forward(self, input, target):


        evaluator = SegEvaluator(1)
        evaluator.reset()
        pred = torch.where(torch.sigmoid(input) > 0.5, 1, 0)
        evaluator.add_batch(gt_image=target.cpu().numpy(), pre_image=pred.cpu().numpy())
        w_00,w_11 = evaluator.loss_weight()
        weight1 = torch.zeros_like(target)
        weight1 = torch.fill_(weight1, w_00)
        weight1[target > 0] = w_11
        loss = F.binary_cross_entropy_with_logits(input, target,weight=weight1,reduction="mean")

        return loss
