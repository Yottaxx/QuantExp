import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SotaConfig(PatchTSTConfig):
    def __init__(self, mse_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.mse_weight = mse_weight


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

        # 异常保护：防止单样本或常数输出导致 NaN
        if pred.numel() < 2 or pred.std() == 0 or target.std() == 0:
            return self.mse_loss(pred, target)

        # IC Loss (Pearson Correlation)
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)
        ic_loss = 1 - correlation

        # MSE Loss
        mse = self.mse_loss(pred, target)

        # 动态加权融合
        total_loss = ic_loss + self.mse_weight * mse

        return total_loss


class PatchTSTForStock(PatchTSTPreTrainedModel):
    config_class = SotaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        # 计算 Patch 数量
        num_patches = (config.context_length - config.patch_length) // config.stride + 1

        # 增强 Head：增加一层 MLP 以提升非线性表达能力
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.num_input_channels * num_patches, 512),
            nn.BatchNorm1d(512),  # BN 加速收敛
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 1)
        )

        # 兼容性读取
        weight = getattr(config, 'mse_weight', 0.5)
        self.loss_fct = HybridLoss(mse_weight=weight)

        self.post_init()

    def forward(self, past_values, labels=None, **kwargs):
        outputs = self.model(past_values=past_values)

        # 确保 Hidden State 维度正确 [Batch, Patches, D_Model]
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )