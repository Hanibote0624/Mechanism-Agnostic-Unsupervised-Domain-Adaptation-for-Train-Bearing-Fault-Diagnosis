
import torch
import torch.nn as nn
import torchvision.models as tvm

def make_backbone(name, num_classes):
    name = name.lower()
    if name == "resnet18":
        m = tvm.resnet18(weights=None)
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(m.conv1.weight.mean(dim=1, keepdim=True))
        m.conv1 = conv1
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "convnext_tiny":
        core = tvm.convnext_tiny(weights=None)
        class Wrap(nn.Module):
            def __init__(self, core, ncls):
                super().__init__()
                self.core = core
                in_f = core.classifier[2].in_features
                core.classifier[2] = nn.Identity()
                self.head = nn.Linear(in_f, ncls)
            def forward(self, x):
                x = x.repeat(1,3,1,1)
                f = self.core(x)
                return self.head(f)
        return Wrap(core, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")
