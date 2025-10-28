
import torch
import torch.nn as nn
import random

class RandomTimeMask(nn.Module):
    def __init__(self, p=0.3, max_ratio=0.1):
        super().__init__()
        self.p = p
        self.max_ratio = max_ratio
    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B, C, H, W = x.shape
        w = max(1, int(W * random.uniform(0, self.max_ratio)))
        start = random.randint(0, max(0, W - w))
        x[..., :, start:start+w] = 0.0
        return x

class RandomFreqMask(nn.Module):
    def __init__(self, p=0.3, max_ratio=0.1):
        super().__init__()
        self.p = p
        self.max_ratio = max_ratio
    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B, C, H, W = x.shape
        h = max(1, int(H * random.uniform(0, self.max_ratio)))
        start = random.randint(0, max(0, H - h))
        x[..., start:start+h, :] = 0.0
        return x

class AdditiveNoise(nn.Module):
    def __init__(self, p=0.2, snr_db=25.0):
        super().__init__()
        self.p = p
        self.snr_db = snr_db
    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        # approximate SNR by per-sample variance
        power = x.var(dim=[2,3], keepdim=True) + 1e-8
        snr = 10 ** (self.snr_db / 10.0)
        noise_power = power / snr
        noise = torch.randn_like(x) * noise_power.sqrt()
        y = x + noise
        return y.clamp(0.0, 1.0)

class ComposeAug(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tm = RandomTimeMask(cfg.get("time_mask_prob",0.3), cfg.get("time_mask_max_ratio",0.1))
        self.fm = RandomFreqMask(cfg.get("freq_mask_prob",0.3), cfg.get("freq_mask_max_ratio",0.1))
        self.noise = AdditiveNoise(cfg.get("noise_prob",0.2), cfg.get("noise_snr_db",25.0))
    def forward(self, x):
        x = self.tm(x)
        x = self.fm(x)
        x = self.noise(x)
        return x
