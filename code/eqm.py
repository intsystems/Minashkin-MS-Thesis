import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
from typing import Tuple
import os

from models import UNet, MLPNet


def get_c_function(c_type: str, a: float = 0.8, b: float = 2.0):
    if c_type == "linear":

        def c(gamma):
            return 1 - gamma

    elif c_type == "truncated":

        def c(gamma):
            return torch.where(
                gamma <= a, torch.ones_like(gamma), (1 - gamma) / (1 - a)
            )

    elif c_type == "piecewise":

        def c(gamma):
            return torch.where(
                gamma <= a, b - (b - 1) / a * gamma, (1 - gamma) / (1 - a)
            )

    else:
        raise ValueError(f"Unknown c_type: {c_type}")

    return c

def get_gamma_scheduler(schedule_type: str):
    """Возвращает функцию для сэмплирования gamma."""
    if schedule_type == "uniform":
        def sampler(batch_size, device):
            return torch.rand(batch_size, device=device)

    elif schedule_type == "linear_decay":
        # Вероятность пропорциональна (1 - gamma) -> чаще видим зашумленные (gamma -> 0)
        def sampler(batch_size, device):
            u = torch.rand(batch_size, device=device)
            return 1.0 - torch.sqrt(1.0 - u)

    elif schedule_type == "linear_grow":
        # Вероятность пропорциональна gamma -> чаще видим чистые данные (gamma -> 1)
        def sampler(batch_size, device):
            u = torch.rand(batch_size, device=device)
            return torch.sqrt(u)

    elif schedule_type == "beta":
        # Гладкое смещение в сторону чистых данных (gamma -> 1)
        def sampler(batch_size, device):
            m = torch.distributions.Beta(1.0, 0.25)
            return m.sample((batch_size,)).to(device)

    else:
        raise ValueError(f"Unknown gamma_schedule: {schedule_type}")

    return sampler

class EqMTrainer:
    def __init__(self, config):
        self.config = config

        self.train_loader, self.test_loader = self._get_data()

        if config.model_type == "unet":
            self.model = UNet(hidden_dim=config.hidden_dim).to(config.device)
        else:
            self.model = MLPNet(hidden_dim=config.hidden_dim).to(config.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        self.c_func = get_c_function(config.c_type, config.c_a, config.c_b)

        self.gamma_sampler = get_gamma_scheduler(getattr(config, 'gamma_schedule', 'uniform'))

        self.train_losses = []

    def _get_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, test_loader

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.config.device)
            batch_size = data.size(0)

            noise = torch.randn_like(data)

            # gamma = torch.rand(batch_size, device=self.config.device)
            gamma = self.gamma_sampler(batch_size, self.config.device)

            x_gamma = (
                gamma.view(-1, 1, 1, 1) * data + (1 - gamma.view(-1, 1, 1, 1)) * noise
            )

            c_gamma = self.c_func(gamma) * self.config.grad_multiplier
            target = (noise - data) * c_gamma.view(-1, 1, 1, 1)

            pred_gradient = self.model(x_gamma)

            loss = F.mse_loss(pred_gradient, target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_grad_norm = 0

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.config.device)

                grad = self.model(data)
                grad_norm = grad.norm(dim=1).mean().item()
                total_grad_norm += grad_norm

        avg_grad_norm = total_grad_norm / len(self.test_loader)
        return avg_grad_norm

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.config.device)
            )
            print("Checkpoint loaded successfully")
        else:
            print(
                f"Warning: Checkpoint file {checkpoint_path} not found. Starting from scratch."
            )


class EqMSampler:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def sample_gd(
        self,
        init_samples: torch.Tensor,
        steps: int = None,
        step_size: float = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.config.sample_steps
        if step_size is None:
            step_size = self.config.step_size

        x = init_samples.clone()
        trajectory = [x.cpu()] if return_trajectory else None

        self.model.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(steps), desc="GD Sampling"):
                grad = self.model(x)
                x = x - step_size * grad

                if return_trajectory:
                    trajectory.append(x.cpu())

        if return_trajectory:
            return x, trajectory
        return x

    def sample_nag(
        self,
        init_samples: torch.Tensor,
        steps: int = None,
        step_size: float = None,
        mu: float = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.config.sample_steps
        if step_size is None:
            step_size = self.config.step_size
        if mu is None:
            mu = self.config.nesterov_mu

        x = init_samples.clone()
        x_last = x.clone()
        trajectory = [x.cpu()] if return_trajectory else None

        self.model.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(steps), desc="NAG-GD Sampling"):
                look_ahead = x + mu * (x - x_last)

                grad = self.model(look_ahead)

                x_last = x
                x = x - step_size * grad

                if return_trajectory:
                    trajectory.append(x.cpu())

        if return_trajectory:
            return x, trajectory
        return x

    def sample_muon(
        self,
        init_samples: torch.Tensor,
        steps: int = None,
        step_size: float = None,
        momentum: float = 0.95,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.config.sample_steps
        if step_size is None:
            step_size = self.config.step_size

        x = init_samples.clone()
        momentum_buffer = torch.zeros_like(x)
        trajectory = [x.cpu()] if return_trajectory else None

        self.model.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(steps), desc="Muon Sampling"):
                grad = self.model(x)

                if i > 0 and momentum_buffer.norm() > 1e-8:
                    projection = (grad * momentum_buffer).sum() / (
                        momentum_buffer.norm() ** 2
                    )
                    grad = grad - projection * momentum_buffer

                momentum_buffer = momentum * momentum_buffer + grad

                x = x - step_size * momentum_buffer

                if return_trajectory:
                    trajectory.append(x.cpu())

        if return_trajectory:
            return x, trajectory
        return x

    def sample_adaptive(
        self,
        init_samples: torch.Tensor,
        step_size: float = None,
        threshold: float = None,
        max_steps: int = 100,
    ) -> Tuple[torch.Tensor, int]:
        if step_size is None:
            step_size = self.config.step_size
        if threshold is None:
            threshold = self.config.adaptive_threshold

        x = init_samples.clone()
        self.model.eval()

        with torch.no_grad():
            for i in range(max_steps):
                grad = self.model(x)
                grad_norm = grad.norm(dim=1).mean().item()

                if grad_norm < threshold:
                    break

                x = x - step_size * grad

        return x, i + 1
