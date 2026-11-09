"""Precision-only wrapper: published loss formulas stay unchanged and run in FP32."""
import torch
from accident_anticipation.models.top import AnticipationHead


def install():
    original = AnticipationHead.loss_by_feat
    if getattr(original, '_top_fp32_loss', False):
        return

    def fp32_loss(self, cls_scores, data_samples):
        with torch.cuda.amp.autocast(enabled=False):
            return original(self, cls_scores.float(), data_samples)

    fp32_loss._top_fp32_loss = True
    AnticipationHead.loss_by_feat = fp32_loss


install()
