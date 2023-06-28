import torch
from torch import nn


class LambdaRankLoss(nn.Module):
    def __init__(self):
        super(LambdaRankLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the LambdaRank loss.
        Args:
        y_pred: Predicted scores for each example.
        y_true: True labels for each example.
        Returns:
        A single tensor, the LambdaRank loss.
        """

        # Convert true labels to ranks
        _, y_true_rank = y_true.sort(descending=True)

        # Compute the rank difference matrix
        rank_diff_matrix = y_true_rank[:, None] - y_true_rank[None, :]

        # Compute the score difference matrix
        score_diff_matrix = y_pred[:, None] - y_pred[None, :]

        # Compute the rank_diff/score_diff ratio matrix
        lambda_ij = rank_diff_matrix / (torch.abs(score_diff_matrix) + 1e-10)

        # Compute the Sij matrix (where Sij = 1 if true_i > true_j and -1 otherwise)
        Sij = torch.sign(rank_diff_matrix)

        # Compute the losses matrix
        losses = Sij * lambda_ij

        # Return the mean loss
        return torch.mean(losses)


class ConcordantPairsLoss(nn.Module):
    def __init__(self):
        super(ConcordantPairsLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the Concordant Pairs loss.
        Args:
        y_pred: Predicted scores for each example.
        y_true: True labels for each example.
        Returns:
        A single tensor, the Concordant Pairs loss.
        """
        # Get the number of examples
        num_examples = y_pred.shape[0]

        # Compute the pairwise difference for predictions and labels
        pred_diffs = y_pred[:, None] - y_pred[None, :]
        true_diffs = y_true[:, None] - y_true[None, :]

        # Compute the pairwise product
        prod = pred_diffs * true_diffs

        # Compute the signs
        signs = torch.sign(prod)

        # Concordant pairs contribute a loss of 0, discordant pairs contribute a loss of 1
        losses = torch.clamp(-signs, min=0)

        ties = (pred_diffs == 0) | (true_diffs == 0)

        # Apply the mask to ignore pairs with ties
        losses = losses.masked_fill(ties, 0)

        # Return the mean loss
        return torch.mean(losses)


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm, ym = x - mx, y - my
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
        r = r_num / r_den
        return 1 - r ** 2


import torch
import torch.nn as nn
from scipy.stats import spearmanr
import numpy as np


class SpearmanCorrelationLoss(nn.Module):
    def __init__(self):
        super(SpearmanCorrelationLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        if len(y_true.shape) == 1:
            y_true = y_true[:, None]
            y_pred = y_pred[:, None]

        spearsum = 0
        cnt = 0
        for col in range(y_pred.shape[1]):
            v = spearmanr(y_pred[:, col], y_true[:, col]).correlation
            if np.isnan(v):
                continue
            spearsum += v
            cnt += 1
        res = spearsum / cnt if cnt > 0 else 0
        return 1 - res


import torch


class RBOLoss(nn.Module):
    def __init__(self, p=0.9, reduction='mean') -> None:
        super(RBOLoss, self).__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, predictions, targets):
        # Compute rankings
        gt_ranking = torch.argsort(targets, descending=True)
        pred_ranking = torch.argsort(predictions, descending=True)

        # Compute RBO
        p = self.p
        n = gt_ranking.size(0)
        weights = (1 - p) * torch.pow(p, torch.arange(n, dtype=torch.float))
        rbo = torch.sum(weights * (gt_ranking == pred_ranking).float())

        # Compute loss
        loss = 1 - rbo
        return loss