from pathlib import Path
import torch
from .model import create_model_retinanet
from .models.orientation_attemps import OrientationAttempts
from .model import pytorch_retinanet
from torch import Tensor
from .utils import label_tools as lt


class BraileInferenceImpl(torch.nn.Module):
    def __init__(
        self,
        params,
        model: Path | str | torch.nn.Module,
        device,
        label_is_valid,
        verbose=1,
        cls_thresh=0.3,
        nms_thresh=0.02,
    ):
        super(BraileInferenceImpl, self).__init__()

        self.verbose = verbose
        self.device = device

        if isinstance(model, torch.nn.Module):
            self.model_weights_fn = ""
            self.model = model

        else:
            self.model_weights_fn = model
            self.model, _, _ = create_model_retinanet(
                params, device=device
            )
            self.model = self.model.to(device)
            self.model.load_state_dict(
                torch.load(self.model_weights_fn, map_location="cpu")
            )

        self.model.eval()

        self.encoder = pytorch_retinanet.encoder.DataEncoder(
            **params.model_params.encoder_params
        )
        self.valid_mask = torch.tensor(label_is_valid).long()
        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = [] if not params.data.get("class_as_6pt", False) else [1] * 6

    def calc_letter_statistics(self, cls_preds, cls_thresh, orientation_attempts):
        device = cls_preds[min(orientation_attempts)].device
        stats = [
            torch.zeros(
                (
                    1,
                    64,
                ),
                device=device,
            )
        ] * 8
        for i, cls_pred in enumerate(cls_preds):
            if i in orientation_attempts:
                scores = cls_pred.sigmoid()
                scores[scores < cls_thresh] = torch.tensor(0.0).to(scores.device)
                stat = scores.sum(1)
                assert list(stat.shape) == [1, 64]
                stats[i] = stat
        stat = torch.cat(stats, dim=0)
        valid_mask = self.valid_mask.to(stat.device)
        sum_valid = (stat * valid_mask).sum(1)
        sum_invalid = (stat * (1 - valid_mask)).sum(1)
        err_score = (sum_invalid + 1) / (sum_valid + 1)
        best_idx = torch.argmin(err_score / (sum_valid + 1))
        return best_idx.item(), (err_score, sum_valid, sum_invalid)

    def forward(self, input_tensor, input_tensor_rotated, find_orientation):
        orientation_attempts = [OrientationAttempts.NONE]
        if find_orientation:
            orientation_attempts += [
                OrientationAttempts.ROT180,
                OrientationAttempts.ROT90,
                OrientationAttempts.ROT270,
            ]

        if len(self.num_classes) > 1:
            assert not find_orientation

        input_data = [None] * 8
        input_data[OrientationAttempts.NONE] = input_tensor.unsqueeze(0)

        if find_orientation:
            input_data[OrientationAttempts.ROT180] = torch.flip(
                input_data[OrientationAttempts.NONE], [2, 3]
            )
            input_data[OrientationAttempts.ROT90] = input_tensor_rotated.unsqueeze(0)
            input_data[OrientationAttempts.ROT270] = torch.flip(
                input_data[OrientationAttempts.ROT90], [2, 3]
            )

        loc_preds: list[Tensor] = [torch.tensor(0)] * 8
        cls_preds: list[Tensor] = [torch.tensor(0)] * 8

        for i, input_data_i in enumerate(input_data):
            if i in orientation_attempts:
                loc_pred, cls_pred = self.model(input_data_i)
                loc_preds[i] = loc_pred
                cls_preds[i] = cls_pred

        if find_orientation:
            best_idx, err_score = self.calc_letter_statistics(
                cls_preds, self.cls_thresh, orientation_attempts
            )

        else:
            best_idx, err_score = OrientationAttempts.NONE, (
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
            )

        if best_idx in [
            OrientationAttempts.INV,
            OrientationAttempts.INV_ROT180,
            OrientationAttempts.INV_ROT90,
            OrientationAttempts.INV_ROT270,
        ]:
            best_idx -= 2

        h, w = input_data[best_idx].shape[2:]
        boxes, labels, scores = self.encoder.decode(
            loc_preds[best_idx][0].cpu().data,
            cls_preds[best_idx][0].cpu().data,
            (w, h),
            cls_thresh=self.cls_thresh,
            nms_thresh=self.nms_thresh,
            num_classes=self.num_classes,
        )

        if len(self.num_classes) > 1:
            labels = torch.tensor(
                [
                    lt.label010_to_int([str(s.item() + 1) for s in lbl101])
                    for lbl101 in labels
                ]
            )

        return boxes, labels, scores, best_idx, err_score
