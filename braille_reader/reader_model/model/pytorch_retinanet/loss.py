from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, class_loss_scale=1.0):
        '''
        :param num_classes: can be int or list of ints (for several class groups)
        :param class_loss_scale: scale that class part of loss is multiplyed to before being added to location part
        '''
        super(FocalLoss, self).__init__()
        self.num_classes = [num_classes] if isinstance(num_classes, int) else num_classes
        self.class_loss_scale = class_loss_scale
        self.loss_dict = {'loc':0, 'cls':0}

    def focal_loss(self, x, y, num_classes):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D]. (for current class group for several class groups)
          y: (tensor) sized [N,]. (for current class group for several class groups)
          num_classes: same as self.num_classes for single class group or num_class for current group (for several groups)

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data, 1+num_classes)  # [N,21]
        t = t[:, 1:]  # exclude background
        t = Variable(t)  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w.data, size_average=False)

    def focal_loss_alt(self, x, y, num_classes):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D]. (for current class group for several class groups)
          y: (tensor) sized [N,]. (for current class group for several class groups)
          num_classes: same as self.num_classes for single class group or num_class for current group (for several groups)

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data, 1+num_classes)
        t = t[:,1:]
        t = Variable(t)

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def focal_loss_3(self, inputs, targets):
        '''Focal loss alternative 3

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        gamma = 2
        targets = targets.unsqueeze(1).float()
        #GVNC 1) only for 1 class 2) alpha is missed
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = (1 - pt) ** gamma * bce_loss
        return f_loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets, loc_calc_mask=None, cls_calc_mask=None):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors] or [batch_size, #anchors, #class_groups] for several class groups.
                cls_targets == 0 for empty class, 1..N for real object class  (iou > iuo_fit_thr),
                -1 for objects to be excluded from loss, i.e. iuo_nofit_thr < iou < iuo_fit_thr
          loc_calc_mask: (tensor) if defined, mask[batch_size] defining what samples use in loc_loss
          cls_calc_mask: (tensor) if defined, mask[batch_size] defining what samples use in cls_loss
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        if len(self.num_classes) == 1:
            pos = cls_targets > 0  # [N,#anchors]
        else:
            pos = cls_targets.max(2)[0] > 0  # [N,#anchors]
        if loc_calc_mask is not None:
            pos = pos & loc_calc_mask.unsqueeze(1)
        num_pos = max(1, pos.data.long().sum())

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        del pos
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        del mask
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)/num_pos
        del masked_loc_preds
        del masked_loc_targets

            ################################################################
        # cls_loss = FocalLoss(cls_preds, cls_targets)
        ################################################################
        if len(self.num_classes) == 1:
            cls_targets.unsqueeze(2)
        cls_loss = 0.
        class_pred_idx = 0
        for class_group_idx, num_classes_i in enumerate(self.num_classes):
            cls_targets_i = cls_targets if len(self.num_classes) == 1 else cls_targets[:, :, class_group_idx]
            cls_preds_i = cls_preds if len(self.num_classes) == 1 else cls_preds[:, :, class_pred_idx:class_pred_idx + num_classes_i]
            class_pred_idx += num_classes_i
            pos_neg = cls_targets_i > -1  # exclude ignored anchors
            if cls_calc_mask is not None:
                pos_neg = pos_neg & cls_calc_mask.unsqueeze(1)
            mask = pos_neg.unsqueeze(2).expand_as(cls_preds_i)
            num_neg = max(1, pos_neg.data.long().sum() * cls_preds_i.shape[-1])
            masked_cls_preds = cls_preds_i[mask].view(-1, num_classes_i)
            del mask
            masked_cls_targets = cls_targets_i[pos_neg]
            del pos_neg
            cls_loss += self.focal_loss(masked_cls_preds, masked_cls_targets, num_classes_i)/num_neg*self.class_loss_scale
            del masked_cls_preds
            del masked_cls_targets

        #print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss, cls_loss), end=' | ')
        loss = loc_loss+cls_loss
        self.loss_dict = {'loss':loss, 'loc':loc_loss, 'cls':cls_loss}
        return loss
