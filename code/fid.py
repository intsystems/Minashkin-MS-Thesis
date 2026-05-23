"""FID: Inception-v3 pool features (MNIST [-1,1] → RGB 299², ImageNet norm)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights

_net = None


def _to_inception(x):
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    m = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    s = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - m) / s


def _inception(device):
    global _net
    if _net is None:
        _net = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
            aux_logits=True,
        )
        _net.fc = nn.Identity()
        _net.eval().to(device)
    return _net


@torch.no_grad()
def _embed(x, device, bs=32):
    net = _inception(device)
    outs = []
    x = x.to(device)
    for i in range(0, len(x), bs):
        outs.append(net(_to_inception(x[i : i + bs])).cpu())
    return torch.cat(outs, dim=0).numpy()


def reference_statistics(real, device):
    """real: [n,1,28,28] CPU; μ и Σ по Inception-признакам."""
    r = _embed(real, device)
    return r.mean(axis=0), np.cov(r, rowvar=False)


def fid_from_statistics(mu_real, sigma_real, mu_fake, sigma_fake):
    eps = 1e-6 * np.eye(sigma_real.shape[0])
    cm = linalg.sqrtm((sigma_real + eps) @ (sigma_fake + eps))
    if np.iscomplexobj(cm):
        cm = cm.real
    d = mu_real - mu_fake
    return float(d @ d + np.trace(sigma_real + sigma_fake - 2 * cm))


def fid_score_from_ref_stats(mu_real, sigma_real, fake, device):
    """fake: [n,1,28,28]; считает μ, Σ по fake и FID к заданному референсу."""
    f = _embed(fake, device)
    return fid_from_statistics(mu_real, sigma_real, f.mean(0), np.cov(f, rowvar=False))


def fid_score(real, fake, device):
    """real, fake: [N,1,28,28], same N; MNIST normalize [-1,1]."""
    mu_r, sig_r = reference_statistics(real, device)
    return fid_score_from_ref_stats(mu_r, sig_r, fake, device)


def reference_from_test_loader(trainer, n):
    xs = []
    k = 0
    for x, _ in trainer.test_loader:
        xs.append(x)
        k += x.shape[0]
        if k >= n:
            break
    return torch.cat(xs)[:n]
