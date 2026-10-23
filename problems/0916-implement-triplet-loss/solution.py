import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # TODO: mean triplet loss with squared L2 distance
    pos_anchor_dist = torch.sum((anchor - positive) ** 2, dim=-1)
    neg_anchor_dist = torch.sum((anchor - negative) ** 2, dim=-1)
    triplet_calc =  pos_anchor_dist - neg_anchor_dist + margin
    loss = F.relu(triplet_calc)
    loss = torch.mean(loss, dim=0)

    return loss