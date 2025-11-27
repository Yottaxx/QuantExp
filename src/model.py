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
    【SOTA Quant Loss v2.0】
    融合了 IC (Information Coefficient) 优化与 Pairwise Ranking (RankNet) 思想。
    """

    def __init__(self, mse_weight=1.0, rank_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param pred:  [Batch, 1] 模型预测的分数 (Logits)
        :param target: [Batch, 1] 真实的收益率或 Rank Label
        """
        pred = pred.flatten()
        target = target.flatten()

        # 1. 基础 MSE 损失 (Anchor)
        # 用于约束预测值的分布范围，防止梯度爆炸
        mse = self.mse_loss(pred, target)

        # 2. IC 损失 (Pearson Correlation Proxy)
        # 使用 Cosine Similarity 替代手动计算，利用 PyTorch 底层优化，数值更稳定
        # Cosine(x-mean, y-mean) == Pearson(x, y)
        if pred.numel() > 1:
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()

            # 添加 epsilon 防止除零错误
            cosine_sim = F.cosine_similarity(
                pred_centered.unsqueeze(0),
                target_centered.unsqueeze(0),
                dim=1,
                eps=1e-8
            )
            # 我们希望相关性越大越好，所以 Loss = 1 - IC
            ic_loss = 1 - cosine_sim.mean()
        else:
            ic_loss = torch.tensor(0.0, device=pred.device)

        # 3. (可选) Pairwise Ranking Loss 思想
        # 如果这是一个强排序任务，可以引入 Pairwise Margin Loss
        # 这里为了保持训练速度，暂时只用 IC Loss 代表排序能力
        # rank_loss = torch.tensor(0.0, device=pred.device)
        #
        # if self.rank_weight > 0 and pred.numel() > 1:
        #     # A. 构建 Pairwise 差值矩阵 [Batch, Batch]
        #     # pred_diff[i][j] = pred[i] - pred[j]
        #     pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        #
        #     # target_diff[i][j] = target[i] - target[j]
        #     target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        #
        #     # B. 生成“正序对”掩码 (Mask)
        #     # 我们只关心那些 target_i > target_j 的配对 (即 target_diff > 0)
        #     # 这样避免重复计算 (i,j) 和 (j,i)，也忽略了 target_i == target_j 的噪声对
        #     s_ij = (target_diff > 0).float()
        #
        #     # C. 计算 RankNet 损失
        #     # RankNet Loss L = log(1 + exp(-(s_i - s_j))) 当真实标签为 i > j 时
        #     # 使用 logaddexp 函数保证数值稳定性，避免 exp 溢出
        #     # log(1 + e^x) 等价于 softplus(x) 或 logaddexp(0, x)
        #     pairwise_loss = torch.logaddexp(torch.zeros_like(pred_diff), -pred_diff)
        #
        #     # D. 只计算有效配对的 Loss
        #     # 有效配对数
        #     num_valid_pairs = s_ij.sum()
        #
        #     if num_valid_pairs > 0:
        #         # 引入权重：可以根据 target 差异的大小加权 (差异越大，排序越重要)
        #         # 这里暂时使用简单的平均，你也可以改为: weighted_loss = pairwise_loss * s_ij * target_diff.abs()
        #         rank_loss = (pairwise_loss * s_ij).sum() / num_valid_pairs
        # total_loss = (self.mse_weight * mse) + (self.rank_weight * ic_loss) + (self.rank_weight * rank_loss)

        # 最终 Loss 组合
        total_loss = (self.mse_weight * mse) + (self.rank_weight * ic_loss)
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