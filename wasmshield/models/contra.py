import torch
from lightly.models.modules import heads

class SimCLR(torch.nn.Module):
    def __init__(self, backbone, in_size=1024, out_size=128, hidden_size=512):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=in_size,
            hidden_dim=hidden_size,
            output_dim=out_size,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z
