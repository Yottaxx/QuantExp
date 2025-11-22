import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SotaConfig(PatchTSTConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class HybridLoss(nn.Module):
    """
    【工业级 SOTA Loss】混合损失函数
    结合 IC Loss (优化排名) 和 MSE Loss (数值稳定性)

    Formula: Loss = (1 - Pearson_IC) + lambda * MSE
    """

    def __init__(self, mse_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # 1. 展平处理
        pred = pred.flatten()
        target = target.flatten()

        # 2. 异常保护 (防止 Batch 内只有一个样本或全 0 导致 NaN)
        if pred.numel() < 2 or pred.std() == 0 or target.std() == 0:
            # 降级为纯 MSE，保证梯度回传不断流
            return self.mse_loss(pred, target)

        # 3. 计算 IC Loss (Pearson Correlation)
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)

        # Cosine Similarity
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)

        # 我们希望 Correlation 越大越好，所以 Loss 项为 1 - Corr
        ic_loss = 1 - correlation

        # 4. 计算 MSE Loss (用于约束数值范围，防止漂移)
        mse = self.mse_loss(pred, target)

        # 5. 融合
        # IC Loss 范围通常在 [0, 2]，MSE 在 [0, 1] (因为 target 是 rank 0~1)
        # 0.5 的权重通常比较平衡
        total_loss = ic_loss + self.mse_weight * mse

        return total_loss


class PatchTSTForStock(PatchTSTPreTrainedModel):
    config_class = SotaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        # 计算 Patch 数量
        # 确保除法取整逻辑正确
        num_patches = (config.context_length - config.patch_length) // config.stride + 1

        # 预测头 (Prediction Head)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.num_input_channels * num_patches, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )

        # 【核心升级】使用混合损失函数
        # mse_weight=1.0 表示同等重视排名和数值分布
        self.loss_fct = HybridLoss(mse_weight=1.0)

        self.post_init()

    def forward(self, past_values, labels=None, **kwargs):
        # Backbone Forward
        outputs = self.model(past_values=past_values)

        # Head Forward
        # outputs.last_hidden_state shape: [Batch, Num_Input_Channels, Num_Patches, D_Model]
        logits = self.head(outputs.last_hidden_state)  # [Batch, 1]

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )