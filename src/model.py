import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SotaConfig(PatchTSTConfig):
    def __init__(self, mse_weight=0.5, stride=None, **kwargs):
        # [Jeff Dean Fix] Explicitly handle 'stride' to ensure it propagates to parent config.
        # Some transformer versions might default 'stride' if not passed explicitly in args.
        if stride is not None:
            kwargs['stride'] = stride

        super().__init__(**kwargs)
        self.mse_weight = mse_weight

        # Double enforcement: Ensure self.stride matches what we expect
        # yotta fix this model key parameter for stride is patch_stride
        if stride is not None:
            self.patch_stride = stride


class HybridLoss(nn.Module):
    """
    【工业级 SOTA Loss】混合损失函数
    Formula: Loss = (1 - Pearson_IC) + mse_weight * MSE
    """

    def __init__(self, mse_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()

        if pred.numel() < 2 or pred.std() == 0 or target.std() == 0:
            return self.mse_loss(pred, target)

        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)
        ic_loss = 1 - correlation

        mse = self.mse_loss(pred, target)
        total_loss = ic_loss + self.mse_weight * mse
        return total_loss


class PatchTSTForStock(PatchTSTPreTrainedModel):
    config_class = SotaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        # [Verification] Ensure config.stride is respected in logic
        # If config.stride=1, patch count increases.
        # If config.stride > 1 (e.g., 2), patch count decreases.
        num_patches = (config.context_length - config.patch_length) // config.stride + 1
        linear_dim = config.d_model
        # Linear Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.num_input_channels * num_patches, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(linear_dim, 1)
        )

        weight = getattr(config, 'mse_weight', 0.5)
        self.loss_fct = HybridLoss(mse_weight=weight)

        self.post_init()

    def forward(self, past_values, labels=None, **kwargs):
        outputs = self.model(past_values=past_values)
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )