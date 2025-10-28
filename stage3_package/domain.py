
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2)
        )
    def forward(self, f):
        return self.net(f)

def mmd_rbf(x, y, sigma=None):
    """
    RBF-kernel MMD with optional sigma (float). If None, median heuristic.
    """
    def pdist(a, b):
        a2 = (a*a).sum(dim=1, keepdim=True)
        b2 = (b*b).sum(dim=1, keepdim=True)
        dist = a2 - 2*a@b.t() + b2.t()
        return torch.clamp(dist, min=0.0)

    if sigma is None:
        with torch.no_grad():
            z = torch.cat([x, y], dim=0)
            dists = pdist(z, z)
            vals = dists[dists>0].detach().view(-1)
            sigma = (torch.median(vals).sqrt().item() + 1e-8) if vals.numel()>0 else 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    Kxx = torch.exp(-gamma * pdist(x,x))
    Kyy = torch.exp(-gamma * pdist(y,y))
    Kxy = torch.exp(-gamma * pdist(x,y))

    m = x.size(0); n = y.size(0)
    mmd = Kxx.sum() - Kxx.trace()
    mmd /= (m*(m-1)+1e-8)
    mmd += (Kyy.sum() - Kyy.trace()) / (n*(n-1)+1e-8)
    mmd -= 2.0 * Kxy.mean()
    return mmd
