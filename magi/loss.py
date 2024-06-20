import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, scale_by_weight=False):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight

    def forward(self, out, mask):
        device = (torch.device('cuda') if out.is_cuda else torch.device('cpu'))

        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # compute logits
        dot = torch.matmul(out, out.T)
        dot = torch.div(dot, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(dot) * logits_mask
        log_probs = dot - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        labels = row.view(row.shape[0], 1)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        log_probs = log_probs[row, col]

        log_probs = log_probs.view(-1, 1)
        loss = torch.zeros_like(unique_labels, dtype=torch.float).to(device)
        loss.scatter_add_(0, labels, log_probs)
        loss = -1 * loss / labels_count.float().unsqueeze(1)

        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

