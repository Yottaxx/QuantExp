import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union


class SotaConfig(PatchTSTConfig):
    def __init__(self, mse_weight=0.5, rank_weight=0.5, stride=None, **kwargs):
        """
        [Config] 增加了 rank_weight 用于平衡 MSE 和 排序损失
        """
        # [Jeff Dean Fix] Explicitly handle 'stride' to ensure it propagates to parent config.
        if stride is not None:
            kwargs['stride'] = stride

        super().__init__(**kwargs)
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight

        # Double enforcement
        if stride is not None:
            self.patch_stride = stride


class QuantLoss(nn.Module):
    """
    【SOTA Quant Loss v2.1 - Ranking Enhanced】
    集成 MSE + IC(Cosine) + Pairwise RankNet
    目标：直接优化横截面排序能力 (IC/RankIC)，而非点预测精度。
    """

    def __init__(self, mse_weight=1.0, rank_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param pred:  [Batch, 1] Logits
        :param target: [Batch, 1] Returns / Labels
        """
        pred = pred.flatten()
        target = target.flatten()

        # 1. MSE Loss (Anchor)
        # 作用：防止预测值漂移到无穷大，保持数值分布合理
        mse = self.mse_loss(pred, target)

        # 2. IC Loss (Correlation Proxy)
        # 作用：优化整体线性相关性
        ic_loss = torch.tensor(0.0, device=pred.device)
        if pred.numel() > 1:
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            # 使用 Cosine Similarity 替代 Pearson，数值更稳定
            cosine_sim = F.cosine_similarity(
                pred_centered.unsqueeze(0),
                target_centered.unsqueeze(0),
                dim=1,
                eps=1e-8
            )
            ic_loss = 1 - cosine_sim.mean()

        # 3. Pairwise Ranking Loss (RankNet / ListMLE)
        # 作用：核心 Alpha 能力，强制模型学习 "A > B" 的关系
        rank_loss = torch.tensor(0.0, device=pred.device)

        # 仅当 Batch 内有多个样本且 rank_weight > 0 时计算
        if self.rank_weight > 0 and pred.numel() > 1:
            # A. 构建差值矩阵 [Batch, Batch]
            # pred_diff[i][j] = pred[i] - pred[j]
            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
            target_diff = target.unsqueeze(1) - target.unsqueeze(0)

            # B. 指示矩阵 S_ij
            # S_ij = 1 (如果 target_i > target_j), -1 (如果 target_i < target_j), 0 (平局)
            S_ij = torch.sign(target_diff)

            # C. RankNet Loss 计算
            # Loss = log(1 + exp(-sigma * (si - sj)))
            # 简化形式: softplus(-S_ij * pred_diff)
            # 这种写法利用了 log-sum-exp trick 保证数值稳定性
            pairwise_losses = F.softplus(-S_ij * pred_diff)

            # D. Masking
            # 1. 过滤掉平局 (target_diff == 0) 的样本对，它们不提供排序梯度
            # 2. 过滤掉自比较 (i==j)
            mask = (target_diff.abs() > 1e-6).float()

            # E. 聚合
            valid_pairs = mask.sum()
            if valid_pairs > 0:
                rank_loss = (pairwise_losses * mask).sum() / valid_pairs

        # 4. 最终加权
        total_loss = (self.mse_weight * mse) + \
                     (self.rank_weight * ic_loss) + \
                     (self.rank_weight * rank_loss)

        return total_loss

class PatchTSTForStock(PatchTSTPreTrainedModel):
    config_class = SotaConfig

    def __init__(self, config: SotaConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        # [Logic Fix] 根据 stride 动态计算 patch 数量
        # Formula: N = (L - P) / S + 1
        num_patches = (config.context_length - config.patch_length) // config.stride + 1

        # Linear Head (Projection)
        # 将 Transformer 输出的 latent representation 映射为 scalar score
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.num_input_channels * num_patches, config.d_model),
            nn.BatchNorm1d(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1)
        )

        # 初始化自定义 Loss
        self.loss_fct = QuantLoss(
            mse_weight=getattr(config, 'mse_weight', 0.5),
            rank_weight=getattr(config, 'rank_weight', 1.0)  # 提高排序权重的默认值
        )

        self.post_init()

    def forward(
            self,
            past_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            **kwargs
    ) -> SequenceClassifierOutput:

        # PatchTST Encoder
        outputs = self.model(past_values=past_values)

        # Pooling / Projection
        # outputs.last_hidden_state shape: [Batch, n_vars, n_patches, d_model]
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # 确保 labels 和 logits 维度对齐
            if labels.shape != logits.shape:
                labels = labels.view_as(logits)
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )