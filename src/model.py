import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SotaConfig(PatchTSTConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ICLoss(nn.Module):
    """
    【工业级 Loss】IC (Information Coefficient) Loss
    目标：最大化预测值与真实值的相关性 (Pearson Correlation)
    公式：Loss = 1 - Correlation(Pred, Target)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target shape: [Batch, 1] -> flatten -> [Batch]
        pred = pred.flatten()
        target = target.flatten()

        # 避免全0导致的NaN
        if pred.std() == 0 or target.std() == 0:
            return torch.tensor(0.0, requires_grad=True).to(pred.device)

        # 计算 Pearson 相关系数
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)

        # 我们希望 Correlation 越大越好 (接近1)，所以 Loss = 1 - Corr
        return 1 - cost


class PatchTSTForStock(PatchTSTPreTrainedModel):
    config_class = SotaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        # 计算 Patch 数量
        num_patches = (config.context_length - config.patch_length) // config.stride + 1

        # 回归头
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.num_input_channels * num_patches, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )

        # 替换默认的 MSE，使用 IC Loss
        self.loss_fct = ICLoss()

        self.post_init()

    def forward(self, past_values, labels=None, **kwargs):
        outputs = self.model(past_values=past_values)
        logits = self.head(outputs.last_hidden_state)  # [Batch, 1]

        loss = None
        if labels is not None:
            # 使用 IC Loss 替代 MSE
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )