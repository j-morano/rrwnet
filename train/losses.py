import torch.nn as nn
import torch
import torchvision.utils as vutils



class BCE3Loss(nn.Module):
    """ BCE3 loss for the simultaneous segmentation of arteries [A], veins [V]
    and vessel tree [VT] (AV3).
    indices:
        artery: 0
        vein: 1
        vessel_tree (artery+vein): 2
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels, vessels, mask):
        mask = mask[:, 0, :, :]
        mask = torch.round(mask)

        pred_a = pred_vessels[:, 0, :, :]
        pred_v = pred_vessels[:, 1, :, :]
        pred_vt = pred_vessels[:, 2, :, :]

        gt_a = vessels[:, 0, :, :]
        gt_v = vessels[:, 1, :, :]
        gt_vt = vessels[:, 2, :, :]

        uncertain = gt_vt - gt_v - gt_a
        uncertain[uncertain < 0] = 0

        mask_unknown = mask - uncertain
        mask_unknown[mask_unknown < 0] = 0

        loss = self.loss(pred_a[mask_unknown > 0.5], gt_a[mask_unknown > 0.5])
        loss += self.loss(pred_v[mask_unknown > 0.5], gt_v[mask_unknown > 0.5])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])

        return loss

    def save_predicted(self, prediction, fname):
        prediction_processed = self.process_predicted(prediction)
        vutils.save_image(prediction_processed, fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())



class RRLoss(nn.Module):
    """Recursive refinement loss.
    """
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, predictions, gt, mask):
        loss_1 = self.base_criterion(predictions[0], gt, mask)
        if len(predictions) == 1:
            return loss_1

        # Second loss (refinement) inspired by Mosinska:CVPR:2018.
        loss_2 = 1 * self.base_criterion(predictions[1], gt, mask)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, prediction in enumerate(predictions[2:], 2):
            loss_2 += i * self.base_criterion(prediction, gt, mask)

        K = len(predictions[1:])
        Z = (1/2) * K * (K + 1)

        loss_2 *= 1/Z

        loss = loss_1 + loss_2

        return loss

    def save_predicted(self, predictions, fname):
        self.base_criterion.save_predicted(predictions[-1], fname)

    def process_predicted(self, predictions):
        new_predictions = []
        for prediction in predictions:
            new_predictions.append(self.base_criterion.process_predicted(prediction))
        return new_predictions
