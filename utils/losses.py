import torch.nn.functional as F
import torch
from torch import nn
class CELoss:
    
    def __call__(self, logits, targets, reduction='none'):
        
        if logits.shape == targets.shape:
            preds = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets * preds, dim=1)
            if reduction == 'none':
                return nll_loss
            return nll_loss.mean()
        else:
            preds = F.log_softmax(logits, dim=-1)
            targets = targets.type(torch.LongTensor).cuda()
            return F.nll_loss(preds, targets, reduction=reduction)


class ConsistencyLoss:
    
    def __call__(self, logits, targets, mask=None, consistency_weight=0.1, reg_weight=0.01):
        preds = F.softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        
        if mask is not None:
            # Enhanced Mask Usage:
            # Use the softmax probabilities as weights for the loss to handle uncertain predictions more gracefully
            # softmax_weights = preds.max(dim=1)[0]  # Max probability as confidence
            weighted_loss = (loss * mask.float()).mean()

            # Consistency Loss between sequential elements to ensure smooth predictions
            # consistency_loss = ((preds[:-1] - preds[1:])**2).mean() * consistency_weight
            
            # Regularization: Adding an L2 penalty on logits to prevent overfitting
            l2_penalty = (logits**2).mean() * reg_weight
            
            # return weighted_loss + consistency_loss + l2_penalty
            return weighted_loss + l2_penalty

        # If no mask is provided, calculate the basic loss without enhancements
        basic_loss = loss.mean()
        
        # Regularization can still be applied even if no mask is provided
        l2_penalty = (logits**2).mean() * reg_weight
        
        return basic_loss + l2_penalty

class SelfAdaptiveFairnessLoss:
    
    def __init__(self, eps=1e-8, entropy_weight=0.1, kl_div_weight=0.05):
        # Initialize parameters
        self.eps = eps  # Small epsilon for numerical stability
        self.entropy_weight = entropy_weight  # Initial weight for entropy loss
        self.kl_div_weight = kl_div_weight  # Weight for KL divergence loss

    def update_weights(self, current_epoch, max_epochs):
        # Dynamic update of weights based on the current epoch
        # Gradually decrease entropy weight to focus more on the primary loss components
        self.entropy_weight = max(0.01, self.entropy_weight * (1 - current_epoch / max_epochs))

    def __call__(self, mask, logits_ulb_s, p_t, label_hist):
        # Apply mask to select high confidence samples
        logits_ulb_s = logits_ulb_s[mask.bool()]
        probs_ulb_s = torch.softmax(logits_ulb_s, dim=-1)
        max_idx_s = torch.argmax(probs_ulb_s, dim=-1)
        
        # Calculate the histogram of predicted classes and normalize it
        histogram = torch.bincount(max_idx_s, minlength=probs_ulb_s.shape[1])
        histogram = histogram.float().cuda()
        histogram = histogram / (histogram.sum() + self.eps)  # Add eps for numerical stability

        # Normalize p_t and label_hist using eps to prevent division by zero
        p_t_normalized = p_t / (p_t.sum() + self.eps)
        label_hist_normalized = label_hist / (label_hist.sum() + self.eps)

        # Calculate entropy of the predicted probabilities to encourage diversity
        entropy_loss = -torch.sum(probs_ulb_s * torch.log(torch.clamp(probs_ulb_s, min=self.eps)), dim=1).mean()

        # Calculate cross entropy between the true class distribution and predicted class distribution
        cross_entropy_loss = torch.sum(p_t_normalized * torch.log(label_hist_normalized + self.eps))

        # Calculate KL divergence for additional regularization
        kl_div_loss = F.kl_div(torch.log(label_hist_normalized + self.eps), p_t_normalized, reduction='batchmean')

        # Combine all loss components
        total_loss = -cross_entropy_loss + self.entropy_weight * entropy_loss + self.kl_div_weight * kl_div_loss

        return total_loss, histogram.mean()

# # Example of initializing and updating the loss function
# loss_func = SelfAdaptiveFairnessLoss()
# loss_func.update_weights(current_epoch=10, max_epochs=100)

class SelfAdaptiveThresholdLoss:
    
    def __init__(self, sat_ema, temperature=1.0, diversity_weight=0.01):
        self.sat_ema = sat_ema
        self.temperature = temperature  # Temperature for softmax scaling
        self.diversity_weight = diversity_weight  # Weight for diversity loss
        self.criterion = ConsistencyLoss()
        
    @torch.no_grad()
    def __update__params__(self, logits_ulb_w, tau_t, p_t, label_hist):
        # Softmax with temperature scaling
        probs_ulb_w = torch.softmax(logits_ulb_w / self.temperature, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)

        # Updating tau_t using a weighted moving average for adaptiveness
        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()
        
        # Updating p_t with the mean of the softmax probabilities
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * probs_ulb_w.mean(dim=0)
        
        # Updating label_hist with bincount adjusted with the sat_ema
        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * (histogram / histogram.sum())

        return tau_t, p_t, label_hist
   
    def __call__(self, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):
        tau_t, p_t, label_hist = self.__update__params__(logits_ulb_w, tau_t, p_t, label_hist)
        
        logits_ulb_w = logits_ulb_w.detach()
        probs_ulb_w = torch.softmax(logits_ulb_w / self.temperature, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
        mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).float()

        # Apply mask and calculate loss
        loss = self.criterion(logits_ulb_s, max_idx_w, mask=mask)

        # Calculate diversity loss to encourage model to explore different features
        diversity_loss = -torch.sum(F.softmax(logits_ulb_s, dim=1) * F.log_softmax(logits_ulb_s, dim=1), dim=1).mean()
        
        # Total loss combines consistency loss and diversity loss
        total_loss = loss + self.diversity_weight * diversity_loss

        return total_loss, mask, tau_t, p_t, label_hist

