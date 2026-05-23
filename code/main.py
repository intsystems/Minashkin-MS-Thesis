import torch
import os

from torch.jit import save

from eqm import EqMTrainer, EqMSampler
from fid import reference_from_test_loader
from visualizations import (
    plot_training_curve,
    visualize_samples,
    visualize_sampling_trajectory_animation,
    compare_step_sizes_animation,
    compare_sampling_algorithms_animation,
    test_partial_denoising,
)


class Config:
    batch_size = 32
    num_workers = 1

    model_type = "unet"  # "unet" or "mlp"
    hidden_dim = 128

    epochs = 4 # 20
    lr = 1e-4
    grad_clip = 1.0

    c_type = "linear"  # "linear", "truncated", "piecewise"
    c_a = 0.8
    c_b = 2.0
    grad_multiplier = 4.0

    gamma_schedule = "beta" # "uniform", "linear_decay", "linear_grow", "beta"

    sample_steps = 50
    step_size = 0.01
    nesterov_mu = 0.35
    adaptive_threshold = 0.1

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    n_samples = 64

    # Per-frame FID on GIFs (many Inception passes — slow); PNG titles still use FID when real_ref is passed.
    fid_on_gif = False

    log_interval = 5
    vis_interval = 100
    save_dir = "../figures"

    load_checkpoint_path = None  # code/eqm.pth
    # load_checkpoint_path = "eqm_mnist_12epochs.pth"
    save_checkpoint_path = "eqm_mnist_beta_4epoch.pth"

config = Config()
os.makedirs(config.save_dir, exist_ok=True)


def main():
    print(f"Using device: {config.device}")

    config_dict = {
        "batch_size": config.batch_size,
        "model_type": config.model_type,
        "hidden_dim": config.hidden_dim,
        "epochs": config.epochs,
        "lr": config.lr,
        "grad_clip": config.grad_clip,
        "c_type": config.c_type,
        "c_a": config.c_a,
        "c_b": config.c_b,
        "gamma_schedule": getattr(config, 'gamma_schedule', 'uniform'), # Добавить сюда
        "grad_multiplier": config.grad_multiplier,
        "sample_steps": config.sample_steps,
        "step_size": config.step_size,
        "nesterov_mu": config.nesterov_mu,
        "adaptive_threshold": config.adaptive_threshold,
        "n_samples": config.n_samples,
        "fid_on_gif": config.fid_on_gif,
        "save_dir": config.save_dir,
        "load_checkpoint_path": config.load_checkpoint_path,
        "save_checkpoint_path": config.save_checkpoint_path,
    }
    print(f"Configuration: {config_dict}")

    trainer = EqMTrainer(config)

    if config.load_checkpoint_path:
        trainer.load_checkpoint(config.load_checkpoint_path)
    else:
        grad_norms = []

        print("Starting training...")
        for epoch in range(config.epochs):
            avg_loss = trainer.train_epoch(epoch)

            if epoch % config.vis_interval == 0:
                avg_grad_norm = trainer.validate()
                grad_norms.append(avg_grad_norm)
                print(
                    f"Epoch {epoch}: Loss={avg_loss:.6f}, Grad Norm at Real Data={avg_grad_norm:.6f}"
                )

        torch.save(trainer.model.state_dict(), config.save_checkpoint_path)

        plot_training_curve(trainer, config)

    print("\nGenerating samples...")
    sampler = EqMSampler(trainer.model, config)

    init_samples = torch.randn(config.n_samples, 1, 28, 28).to(config.device)
    real_ref = reference_from_test_loader(trainer, config.n_samples)

    print("\nSampling with Vanilla Gradient Descent...")
    samples_gd, trajectory_gd = sampler.sample_gd(
        init_samples.clone(), steps=50, return_trajectory=True
    )
    visualize_samples(
        samples_gd,
        "Samples - Vanilla GD",
        "samples_gd.png",
        config,
        real_ref=real_ref,
        device=config.device,
    )

    print("\nCreating sampling trajectory animation...")
    visualize_sampling_trajectory_animation(
        trajectory_gd,
        "sampling_trajectory.gif",
        config,
        real_ref=real_ref,
        device=config.device,
    )

    print("\nSampling with NAG-GD...")
    samples_nag, trajectory_nag = sampler.sample_nag(
        init_samples.clone(), steps=50, return_trajectory=True
    )
    visualize_samples(
        samples_nag,
        "Samples - NAG-GD",
        "samples_nag.png",
        config,
        real_ref=real_ref,
        device=config.device,
    )

    print("\nCreating GD vs NAG comparison animation...")
    compare_sampling_algorithms_animation(
        trainer.model,
        config,
        init_samples.clone(),
        real_ref=real_ref,
        device=config.device,
    )

    print("\nSampling with Adaptive Compute...")
    samples_adaptive, steps_used = sampler.sample_adaptive(init_samples.clone())
    visualize_samples(
        samples_adaptive,
        f"Samples - Adaptive (avg {steps_used:.1f} steps)",
        "samples_adaptive.png",
        config,
        real_ref=real_ref,
        device=config.device,
    )
    print(f"Average steps used: {steps_used:.1f}")

    print("\nCreating step size comparison animation...")
    compare_step_sizes_animation(
        trainer.model,
        config,
        init_samples.clone(),
        real_ref=real_ref,
        device=config.device,
    )

    print("\nTesting partial denoising...")
    test_partial_denoising(
        trainer.model, config, real_ref=real_ref, device=config.device
    )

    print(f"\nAll results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
