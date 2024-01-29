'''Encode object boxes and labels.'''
import math
from typing import List, Tuple, Union
from torch import Tensor
import torch

from .utils import meshgrid, box_iou, box_nms, change_box_order


class DataEncoder(torch.nn.Module):
    def __init__(self,
                 anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.],
                 aspect_ratios = [1 / 2., 1 / 1., 2 / 1.], # width/height
                 scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)],
                 iuo_fit_thr = 0.5, # if iou > iuo_fit_thr => rect fits anchor
                 iuo_nofit_thr = 0.4, # if iou < iuo_nofit_thr => anchor has no fit rect
                 ):
        # type: (List[float], List[float], List[float], float, float) -> None
        super(DataEncoder, self).__init__()
        self.anchor_areas = anchor_areas  # p3 -> p7
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.anchor_wh = self._get_anchor_wh()
        self.iuo_fit_thr = iuo_fit_thr
        self.iuo_nofit_thr = iuo_nofit_thr
        self.input_size = torch.tensor(0)

    def forward(self):
        pass

    def num_layers(self):
        '''
        :return: num_layers to be passed to RetinaNet
        '''
        return len(self.anchor_areas)

    def num_anchors(self):
        '''
        :return: num_anchors to be passed to RetinaNet
        '''
        return len(self.aspect_ratios)*len(self.scale_ratios)

    def _get_anchor_wh(self):
        # type: ()->Tensor
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh: List[List[float]] = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        r = torch.tensor(anchor_wh, dtype=torch.float32).view(num_fms, -1, 2)
        return r

    def _get_anchor_boxes(self, input_size, device):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        if not torch.equal(input_size, self.input_size):
            self.input_size = input_size
            num_fms = len(self.anchor_areas)
            fm_sizes = [(input_size/math.pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes
            num_anchors = self.num_anchors()

            boxes: List[Tensor] = []
            for i in range(num_fms):
                fm_size = fm_sizes[i]
                grid_size = input_size / fm_size
                fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
                xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
                xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,num_anchors,2).to(self.anchor_wh.device)
                wh = self.anchor_wh[i].view(1,1,num_anchors,2).expand(fm_h,fm_w,num_anchors,2)
                box = torch.cat([xy,wh], 3)  # [x,y,w,h]
                boxes.append(box.view(-1,4))
            self.anchor_boxes = torch.cat(boxes, 0).to(device)
        return self.anchor_boxes

    @torch.jit.unused
    def encode(self, boxes, labels, input_size):
        # type: (Tensor, Tensor, Tuple[int, int])->Tuple[Tensor, Tensor, Tensor]
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,] or [#obj, class_groups].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        assert not isinstance(input_size, int)
        input_size = torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size, boxes.device)
        if boxes.shape[0] != 0:
            boxes = change_box_order(boxes, 'xyxy2xywh')

            ious = box_iou(anchor_boxes, boxes, order='xywh')
            max_ious, max_ids = ious.max(1)
            boxes = boxes[max_ids]

            loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
            loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
            loc_targets = torch.cat([loc_xy,loc_wh], 1)
            cls_targets = 1 + labels[max_ids]

            idxs: Tensor = max_ious<=self.iuo_fit_thr
            cls_targets[idxs] = torch.tensor(0).to(cls_targets.device)
            if self.iuo_nofit_thr < self.iuo_fit_thr:
                ignore = (max_ious>self.iuo_nofit_thr) & (idxs)  # ignore ious between [0.4,0.5]
                cls_targets[ignore] = torch.tensor(-1).to(cls_targets.device)  # for now just mark ignored to -1
        else:
            loc_targets = torch.zeros(anchor_boxes.shape[0], 4, dtype = torch.float32, device=anchor_boxes.device)
            if len(labels.shape) > 1:
                assert len(labels.shape) == 2
                cls_targets = torch.zeros(anchor_boxes.shape[0], labels.shape[1], dtype=torch.long, device=anchor_boxes.device)
            else:
                cls_targets = torch.zeros(anchor_boxes.shape[0], dtype=torch.long, device=anchor_boxes.device)
            max_ious = torch.zeros(anchor_boxes.shape[0], dtype=torch.float32)
        return loc_targets, cls_targets, max_ious

    def decode(self, loc_preds, cls_preds, input_size, cls_thresh = 0.5, nms_thresh = 0.5, num_classes: List[int] = []):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        assert loc_preds.device == cls_preds.device
        assert not isinstance(input_size, int)
        input_size = torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size, loc_preds.device)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        if len(num_classes) <= 1:
            score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        else:
            score, _ = cls_preds.sigmoid().max(1)  # [#anchors,]
            labels = torch.zeros((len(cls_preds), len(num_classes)), dtype=torch.int, device=cls_preds.device)
            pos = 0
            for i, n in enumerate(num_classes):
                score_i, labels[:, i] = cls_preds[:, pos:pos+n].sigmoid().max(1)
                labels[:, i][score_i <= cls_thresh] = -1
                pos += n

        ids = score > cls_thresh
        ids = ids.nonzero().squeeze()             # [#obj,]
        if len(ids.shape):
            keep = box_nms(boxes[ids], score[ids], threshold=nms_thresh)
            return boxes[ids][keep], labels[ids][keep], score[ids][keep]
        else:
            return boxes[ids].unsqueeze(0), labels[ids].unsqueeze(0), score[ids].unsqueeze(0)

def DataEncoderScripted():
    return torch.jit.script(DataEncoder)
