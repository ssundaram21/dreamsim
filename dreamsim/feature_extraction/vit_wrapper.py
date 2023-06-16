from transformers import PretrainedConfig
from transformers import PreTrainedModel


class ViTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ViTModel(PreTrainedModel):
    config_class = ViTConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.blocks = model.blocks

    def forward(self, x):
        return self.model(x)
