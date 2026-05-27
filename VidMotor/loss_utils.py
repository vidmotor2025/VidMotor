import torch
import torch.nn.functional as F
import torch.nn as nn


def cross_individual_contrastive_loss(feats1, labels1, subs1=None, feats2=None, labels2=None, subs2=None, temperature=0.5, type='subject'):
    """Implementation of cross-individual contrastive learning."""
    if type == 'sample':  # intra-batch
        loss = intra_batch_supcon_loss(feats1, labels1, subs1, temperature, type)
    elif type == 'subject' and len(set(subs1)) > 1:  # intra-batch
        loss = intra_batch_supcon_loss(feats1, labels1, subs1, temperature, type)
    elif type == 'subject' and len(set(subs1)) == 1:  # cross-batch
        feats = torch.cat([feats1, feats2], dim=0)
        labels = torch.cat([labels1, labels2], dim=0)
        subs = subs1 + subs2
        assert len(labels) == len(subs)
        loss = intra_batch_supcon_loss(feats, labels, subs, temperature, type)
    else:
        raise ValueError(f"Unsupported combination of arguments: type={type}, num_unique_subs={len(set(subs1))}")
    return loss


def intra_batch_supcon_loss(feats, labels, subs, temperature=0.1, type='subject'):
    """
    Construct positive and negative pairs for contrastive learning:
    • Positive pairs: samples from different subjects sharing the same label.
    • Negative pairs: samples from the same subject but having different labels.
    The temperature parameter corresponds to the temperature used in supervised contrastive learning, controlling the smoothness or scaling of similarity scores.
    """
    feats = F.normalize(feats, p=2, dim=1)
    B = feats.size(0)
    labels = labels.to(feats.device)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # Class mask: [B, B]
    logits = torch.matmul(feats, feats.T) / temperature  # Feature similarity matrix: [B, B]
    mask_self = ~torch.eye(B, dtype=torch.bool, device=feats.device)

    if type == 'subject':
        unique_subs = sorted(set(subs))  # Fixed order of subject IDs
        sub2id = {s: i for i, s in enumerate(unique_subs)}  # Build a mapping from subject IDs to integer indices
        subs_encoded = torch.tensor([sub2id[s] for s in subs], device=feats.device)  # Map subs to an integer tensor
        subs_eq = subs_encoded.unsqueeze(0) == subs_encoded.unsqueeze(1)  # Subject mask: [B, B]
        pos_mask = labels_eq & (~subs_eq) & mask_self  # Different subjects sharing the same label
        neg_mask = (~labels_eq) & subs_eq & mask_self  # Same subject but having different labels
    elif type == 'sample':
        pos_mask = labels_eq & mask_self
        neg_mask = (~labels_eq) & mask_self
    else:
        raise ValueError(f"Unsupported type: {type}. Must be 'subject' or 'sample'")

    loss = 0.0
    count_anchor = 0  # Some anchors have no positive or negative samples and are excluded from the calculation
    for i in range(B):
        # Extract similarity scores for the current anchor
        pos_logits = logits[i][pos_mask[i]]  # Similarities with positive samples
        neg_logits = logits[i][neg_mask[i]]  # Similarities with negative samples
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue
        pos_exp = pos_logits.exp()
        neg_exp = neg_logits.exp()
        denom = torch.cat([pos_exp, neg_exp]).sum()
        loss += -(pos_exp / denom).log().mean()
        count_anchor += 1
    if count_anchor > 0:
        loss = loss / count_anchor
    else:
        print("Length of anchors is 0. No valid anchor found in batch.")
        loss = torch.zeros(1, device=feats.device, requires_grad=True).squeeze()
    return loss


def counterfactual_loss(logits, target, margin=0.1):
    """
    Apply a stronger penalty when the predicted probability of the ground-truth class exceeds the maximum probability of any other class.
    In other words, encourages the model to predict incorrectly.
    Parameters:
        logits: Tensor of shape (N, num_classes), model outputs
        target: LongTensor of shape (N,), ground-truth labels
        margin: float, margin threshold
    """
    probs = F.softmax(logits, dim=1)  # (N, num_classes)
    batch_size = logits.size(0)
    device = logits.device
    range_batch = torch.arange(batch_size, device=device)  # 0,...,N-1
    p_y = probs[range_batch, target]
    probs_clone = probs.clone()
    probs_clone[range_batch, target] = -1.0
    max_p_other = probs_clone.max(dim=1)[0]
    loss = F.relu(p_y - max_p_other + margin).mean()
    return loss


def non_causal_loss(x_pred_c, x_pred_nc):
    """Maximize the distributional divergence between the causal and non-causal predictions."""
    p_c = F.softmax(x_pred_c, dim=1)
    p_nc = F.softmax(x_pred_nc, dim=1)
    return 2 - ((p_c - p_nc) ** 2).sum(dim=1).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()