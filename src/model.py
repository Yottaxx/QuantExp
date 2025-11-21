import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SotaConfig(PatchTSTConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PatchTSTForStock(PatchTSTPreTrainedModel):
    config_class = SotaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        # 计算 Patch 数量用于 Flatten
        num_patches = (config.context_length - config.patch_length) // config.stride + 1

        # 自定义回归头
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.num_input_channels * num_patches, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )
        self.post_init()

    def forward(self, past_values, labels=None, **kwargs):
        outputs = self.model(past_values=past_values)
        logits = self.head(outputs.last_hidden_state)  # [Batch, 1]

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )