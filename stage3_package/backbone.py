
import torch
import torch.nn as nn
import torchvision.models as tvm

class FeatureBackbone(nn.Module):
    """
    Wrap a CNN to output (logits, features) where features are penultimate embeddings.
    Supports 'resnet18' and 'convnext_tiny'.
    """
    def __init__(self, name, num_classes):
        super().__init__()
        name = name.lower()
        self.name = name
        if name == "resnet18":
            m = tvm.resnet18(weights=None)
            conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv1.weight.copy_(m.conv1.weight.mean(dim=1, keepdim=True))
            m.conv1 = conv1
            in_f = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Linear(in_f, num_classes)
            self.feat_dim = in_f
        elif name == "convnext_tiny":
            core = tvm.convnext_tiny(weights=None)
            in_f = core.classifier[2].in_features
            core.classifier[2] = nn.Identity()
            self.core = core
            self.head = nn.Linear(in_f, num_classes)
            self.feat_dim = in_f
        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x, return_feat=True):
        if self.name == "resnet18":
            f = self.backbone(x)          # N x feat_dim
            logits = self.head(f)
            return (logits, f) if return_feat else logits
        else:
            f = self.core(x.repeat(1,3,1,1))
            logits = self.head(f)
            return (logits, f) if return_feat else logits
