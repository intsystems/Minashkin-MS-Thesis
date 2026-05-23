import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()

        self.enc1 = self._block(in_channels, hidden_dim)
        self.enc2 = self._block(hidden_dim, hidden_dim * 2)
        self.enc3 = self._block(hidden_dim * 2, hidden_dim * 4)

        self.bottleneck = self._block(hidden_dim * 4, hidden_dim * 4)

        self.upconv3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2,
                                         kernel_size=2, stride=2)
        self.dec3 = self._block(hidden_dim * 4, hidden_dim * 2)

        self.upconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim,
                                         kernel_size=2, stride=2)
        self.dec2 = self._block(hidden_dim * 2, hidden_dim)

        self.dec1 = nn.Conv2d(hidden_dim, in_channels, 3, padding=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleneck(e3)
        up3 = self.upconv3(b)
        if up3.size() != e2.size():
            up3 = F.interpolate(up3, size=e2.shape[2:], mode='nearest')
        d3 = torch.cat([up3, e2], dim=1)
        d3 = self.dec3(d3)

        up2 = self.upconv2(d3)
        if up2.size() != e1.size():
            up2 = F.interpolate(up2, size=e1.shape[2:], mode='nearest')
        d2 = torch.cat([up2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.dec1(d2)

    def extract_features(self, x):
        """Global-pooled embedding before final conv (B, hidden_dim)."""
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleneck(e3)
        up3 = self.upconv3(b)
        if up3.size() != e2.size():
            up3 = F.interpolate(up3, size=e2.shape[2:], mode="nearest")
        d3 = torch.cat([up3, e2], dim=1)
        d3 = self.dec3(d3)

        up2 = self.upconv2(d3)
        if up2.size() != e1.size():
            up2 = F.interpolate(up2, size=e1.shape[2:], mode="nearest")
        d2 = torch.cat([up2, e1], dim=1)
        d2 = self.dec2(d2)
        return F.adaptive_avg_pool2d(d2, 1).flatten(1)

class MLPNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        grad_flat = self.net(x_flat)
        return grad_flat.view_as(x)

    def extract_features(self, x):
        batch_size = x.size(0)
        h = x.view(batch_size, -1)
        for layer in list(self.net.children())[:-1]:
            h = layer(h)
        return h
