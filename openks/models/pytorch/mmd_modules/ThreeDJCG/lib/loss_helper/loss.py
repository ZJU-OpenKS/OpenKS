import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=1)

        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss


class SoftmaxRankingLoss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape

        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=1)

        # reduction
        loss = -torch.sum(torch.log(1 - probs + 1e-8) * (1 - targets), dim=1).mean()

        return loss

class SoftmaxRankingLoss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape

        # compute the probabilities
        sigmoid = nn.Sigmoid()
        probs = sigmoid(inputs + 1e-8)
        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss

class SigmoidRankingFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.autograd.Variable(torch.ones(2))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = torch.autograd.Variable(alpha)
        super().__init__()

    def forward(self, inputs, targets, mask=None):
        # input check
        assert inputs.shape == targets.shape
        # compute the probabilities
        sigmoid = nn.Sigmoid()
        probs = sigmoid(inputs)
        # reduction
        loss_positive = -self.alpha[1]*((1-probs)**self.gamma) * torch.log(probs+1e-8) * targets
        loss_negative = -self.alpha[0]*(probs**self.gamma) * torch.log(1-probs+1e-8) * (1-targets)

        if mask is None:
            loss = (loss_positive + loss_negative).mean()
        else:
            loss = ((loss_positive + loss_negative) * mask).sum() / (mask.sum() + 1e-8)
        return loss

class SoftmaxRankingFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, class_num=10000):
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.autograd.Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = torch.autograd.Variable(alpha)
        super().__init__()

    def forward(self, inputs, targets, mask=None):
        # input check
        assert inputs.shape == targets.shape
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=-1)

        #print(class_mask)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = (targets * self.alpha[None, :targets.shape[-1]]).sum(dim=-1)

        # reduction
        probs = torch.sum(probs * targets, dim=-1)
        log_p = torch.log(probs + 1e-8)
        loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

