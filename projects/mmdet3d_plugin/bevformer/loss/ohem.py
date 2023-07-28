import torch
import torch.nn as nn
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.utils import reduce_loss
from mmseg.models import LOSSES


@LOSSES.register_module()
class CrossEntropyOHEMLoss(nn.Module):
    """
    Cross Entropy Loss with additional OHEM.

    Args:
        top_ratio (float): top ratio to be mined. Default: 0.3.
        top_weight (float): scaling weight given to top hard examples mined. Default: 1.0.
        weight_per_cls (list, tuple, Tensor, np.ndarray): a manual rescaling weight given to each class.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
        ce_loss_weight (float): weight to reweigh CE loss output. Default: 1.0.
        reduction (str): mean, sum and none are supported.
    """
    def __init__(self, 
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 top_ratio=0.3,
                 use_ohem=True,
                 top_weight=1.0, 
                 reduction='mean'):
        super(CrossEntropyOHEMLoss, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.use_ohem = use_ohem

        self.ce_loss = CrossEntropyLoss(use_sigmoid, use_mask, 
            reduction='none', class_weight=class_weight, 
            ignore_index=ignore_index, loss_weight=loss_weight)
    
    def forward(self, 
                input, 
                target, 
                weight=None,
                avg_factor=None):
        loss = self.ce_loss(input, target, weight=weight, avg_factor=avg_factor)
        size = loss.size()
        loss: torch.Tensor = loss.reshape(-1)
        if not self.use_ohem: return loss

        k = max(int(self.top_ratio * loss.shape[0]), 1)
        loss_topk, topk_idx = torch.topk(loss, k, largest=True, sorted=False)

        if self.reduction != 'none':
            loss = reduce_loss(loss, self.reduction)
            loss_topk = reduce_loss(loss_topk, self.reduction)
            return loss + self.top_weight * loss_topk
        else:
            loss[topk_idx] += self.top_weight * loss_topk
            return loss.reshape(size)


@LOSSES.register_module()
class FocalOHEMLoss(nn.Module):
    """
    Cross Entropy Loss with additional OHEM.

    Args:
        use_sigmoid (bool): Only support sigmoid in FocalLoss. Must be True.
        gamma (float): gamma of FocalLoss. Default: 2.0.
        alpha (float): alpha of FocalLoss. Default: 0.5.
        focal_loss_weight (float): rescaling weight given to FocalLoss. Default: 1.0.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
        top_ratio (float): top ratio to be mined. Default: 0.3.
        top_weight (float): scaling weight given to top hard examples mined. Default: 1.0.
        reduction (str): mean, sum and none are supported.
    """
    def __init__(self, 
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 loss_weight=1.0,
                 activated=False, 
                 top_ratio=0.3, 
                 top_weight=1.0, 
                 reduction='mean'):
        super(FocalOHEMLoss, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

        self.focal_loss = FocalLoss(use_sigmoid, gamma, alpha, 
            reduction='none', loss_weight=loss_weight, activated=activated)
    
    def forward(self, 
                input, 
                target, 
                weight=None,
                avg_factor=None):
        loss = self.focal_loss(input, target, weight=weight, avg_factor=avg_factor)
        size = loss.size()
        loss: torch.Tensor = loss.reshape(-1)

        k = max(int(self.top_ratio * loss.shape[0]), 1)
        loss_topk, topk_idx = torch.topk(loss, k, largest=True, sorted=False)

        if self.reduction != 'none':
            loss = reduce_loss(loss, self.reduction)
            loss_topk = reduce_loss(loss_topk, self.reduction)
            return loss + self.top_weight * loss_topk
        else:
            loss[topk_idx] += self.top_weight * loss_topk
            return loss.reshape(size)