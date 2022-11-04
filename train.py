
import torch
import yaml

class Network():
    def __init__(self, Torch_Model, Model_Name, Log_Path):
        super(Network, self).__init__()


        self.BinaryCrossEntropy = torch.nn.BCELoss()

    def dice_loss(self, pred_mask, true_mask):
        loss = 1 - self.dice(pred_mask, true_mask)

        return loss


    def calc_loss(self,pred, target, bce_weight=0.2):
        bce = self.BinaryCrossEntropy(pred, target)
        dice = self.dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss


    def dice(self, pred, target):
        intersection = (abs(target - pred) < 0.05).sum()
        cardinality = (target >= 0).sum() + (pred >= 0).sum()

        return 2.0 * intersection / cardinality