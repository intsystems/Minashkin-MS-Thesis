import torch
import os

from eqm import EqMTrainer, EqMSampler
from visualizations import (
    plot_training_curve,
    visualize_samples,
    visualize_sampling_trajectory_animation,
    compare_step_sizes_animation,
    compare_sampling_algorithms_animation,
    test_partial_denoising
)


class Config:
    batch_size = 32
    num_workers = 1

    model_type = "unet"  # "unet" or "mlp"
    hidden_dim = 128

    epochs = 20
    lr = 1e-4
    grad_clip = 1.0

    c_type = "truncated"  # "linear", "truncated", "piecewise"
    c_a = 0.8
    c_b = 2.0
    grad_multiplier = 4.0

    sample_steps = 50
    step_size = 0.01
    nesterov_mu = 0.35
    adaptive_threshold = 0.1

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    n_samples = 64

    log_interval = 5
    vis_interval = 100
    save_dir = "../figures"

config = Config()
os.makedirs(config.save_dir, exist_ok=True)


def main():
    print(f"Using device: {config.device}")
    print(f"Configuration: {vars(config)}")

    trainer = EqMTrainer(config)

    grad_norms = []

    print("Starting training...")
    for epoch in range(config.epochs):
        avg_loss = trainer.train_epoch(epoch)

        if epoch % config.vis_interval == 0:
            avg_grad_norm = trainer.validate()
            grad_norms.append(avg_grad_norm)
            print(f"Epoch {epoch}: Loss={avg_loss:.6f}, Grad Norm at Real Data={avg_grad_norm:.6f}")

    torch.save(trainer.model.state_dict(), "eqm.pth")

    plot_training_curve(trainer, config)

    print("\nGenerating samples...")
    sampler = EqMSampler(trainer.model, config)

    init_samples = torch.randn(config.n_samples, 1, 28, 28).to(config.device)

    print("\nSampling with Vanilla Gradient Descent...")
    samples_gd, trajectory_gd = sampler.sample_gd(
        init_samples.clone(), steps=50, return_trajectory=True
    )
    visualize_samples(samples_gd, "Samples - Vanilla GD", "samples_gd.png", config)

    print("\nCreating sampling trajectory animation...")
    visualize_sampling_trajectory_animation(trajectory_gd, "sampling_trajectory.gif", config)

    print("\nSampling with NAG-GD...")
    samples_nag, trajectory_nag = sampler.sample_nag(init_samples.clone(), steps=50, return_trajectory=True)
    visualize_samples(samples_nag, "Samples - NAG-GD", "samples_nag.png", config)

    print("\nCreating GD vs NAG comparison animation...")
    compare_sampling_algorithms_animation(trainer.model, config, init_samples.clone())

    print("\nSampling with Adaptive Compute...")
    samples_adaptive, steps_used = sampler.sample_adaptive(init_samples.clone())
    visualize_samples(samples_adaptive, f"Samples - Adaptive (avg {steps_used:.1f} steps)",
                        "samples_adaptive.png", config)
    print(f"Average steps used: {steps_used:.1f}")

    print("\nCreating step size comparison animation...")
    compare_step_sizes_animation(trainer.model, config, init_samples.clone())

    print("\nTesting partial denoising...")
    test_partial_denoising(trainer.model, config)

    print(f"\nAll results saved to: {config.save_dir}")

if __name__ == "__main__":
    main()
