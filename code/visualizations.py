import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from eqm import EqMSampler


def plot_training_curve(trainer, config):
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"{config.save_dir}/training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

def visualize_samples(samples, title, filename, config, nrow=8):
    samples = (samples + 1) / 2  # Denormalize
    samples = samples.clamp(0, 1)

    grid = torchvision.utils.make_grid(samples, nrow=nrow, normalize=False)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{config.save_dir}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def visualize_sampling_trajectory_animation(trajectory, filename, config, n_frames=50):

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.close(fig)

    total_steps = len(trajectory)
    step_indices = np.linspace(0, total_steps-1, min(n_frames, total_steps), dtype=int)

    def update(frame_idx):
        ax.clear()
        step = step_indices[frame_idx]
        samples = trajectory[step][:25]
        samples = (samples + 1) / 2  # Denormalize
        samples = samples.clamp(0, 1)

        grid = torchvision.utils.make_grid(samples, nrow=5, normalize=False)
        ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        ax.set_title(f"Step {step}/{total_steps-1}")
        ax.axis('off')

    anim = animation.FuncAnimation(fig, update, frames=len(step_indices),
                                   interval=200, blit=False)

    try:
        anim.save(f"{config.save_dir}/{filename}", writer='pillow', fps=5)
        print(f"Saved animation: {filename}")
    except Exception as e:
        print(f"Warning: Could not save animation {filename}: {e}")
        visualize_samples(trajectory[0][:25], "Sampling Start", "sampling_start.png", config, nrow=5)
        visualize_samples(trajectory[-1][:25], "Sampling End", "sampling_end.png", config, nrow=5)
    plt.close()

def compare_step_sizes_animation(model, config, init_samples):

    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    sampler = EqMSampler(model, config)
    steps = 50

    print("Generating trajectories for different step sizes...")
    trajectories = []
    for step_size in step_sizes:
        print(f"  step_size={step_size}")
        _, trajectory = sampler.sample_gd(init_samples.clone(), steps=steps,
                                         step_size=step_size, return_trajectory=True)
        trajectories.append(trajectory)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    plt.close(fig)

    def update(frame):
        for idx, (ax, step_size, trajectory) in enumerate(zip(axes, step_sizes, trajectories)):
            ax.clear()
            samples = trajectory[frame][:9]  # 3x3 grid
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            grid = torchvision.utils.make_grid(samples, nrow=3, normalize=False)
            ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
            ax.set_title(f"Step Size: {step_size}", fontsize=10)
            ax.axis('off')

        fig.suptitle(f"Generation Step {frame}", fontsize=16)
        return axes

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=200, blit=False)
    try:
        anim.save(f"{config.save_dir}/step_size_comparison.gif", writer='pillow', fps=5)
        print("Saved step size comparison animation")
    except Exception as e:
        print(f"Warning: Could not save step size animation: {e}")
        compare_step_sizes(model, init_samples, config)
    plt.close()

def compare_sampling_algorithms_animation(model, config, init_samples):
    sampler = EqMSampler(model, config)
    steps = 50

    print("Generating trajectories for GD vs NAG vs Muon comparison...")
    _, trajectory_gd = sampler.sample_gd(init_samples.clone(), steps=steps, return_trajectory=True)
    _, trajectory_nag = sampler.sample_nag(init_samples.clone(), steps=steps, return_trajectory=True)
    _, trajectory_muon = sampler.sample_muon(init_samples.clone(), steps=steps, return_trajectory=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_gd, ax_nag, ax_muon = axes
    plt.close(fig)

    def update(frame):
        # GD
        ax_gd.clear()
        samples_gd = trajectory_gd[frame][:16]
        samples_gd = (samples_gd + 1) / 2
        samples_gd = samples_gd.clamp(0, 1)
        grid_gd = torchvision.utils.make_grid(samples_gd, nrow=4, normalize=False)
        ax_gd.imshow(grid_gd.permute(1, 2, 0).numpy(), cmap='gray')
        ax_gd.set_title(f"Vanilla GD\nStep {frame}", fontsize=11)
        ax_gd.axis('off')

        # NAG
        ax_nag.clear()
        samples_nag = trajectory_nag[frame][:16]
        samples_nag = (samples_nag + 1) / 2
        samples_nag = samples_nag.clamp(0, 1)
        grid_nag = torchvision.utils.make_grid(samples_nag, nrow=4, normalize=False)
        ax_nag.imshow(grid_nag.permute(1, 2, 0).numpy(), cmap='gray')
        ax_nag.set_title(f"NAG-GD\nStep {frame}", fontsize=11)
        ax_nag.axis('off')

        # Muon
        ax_muon.clear()
        samples_muon = trajectory_muon[frame][:16]
        samples_muon = (samples_muon + 1) / 2
        samples_muon = samples_muon.clamp(0, 1)
        grid_muon = torchvision.utils.make_grid(samples_muon, nrow=4, normalize=False)
        ax_muon.imshow(grid_muon.permute(1, 2, 0).numpy(), cmap='gray')
        ax_muon.set_title(f"Muon\nStep {frame}", fontsize=11)
        ax_muon.axis('off')

        fig.suptitle(f"EqM Sampling Algorithm Comparison\n{frame}/{steps-1} steps",
                    fontsize=14, fontweight='bold')
        return axes

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=200, blit=False)
    try:
        anim.save(f"{config.save_dir}/sampling_comparison.gif", writer='pillow', fps=5)
        print("Saved GD vs NAG vs Muon comparison animation")
    except Exception as e:
        print(f"Warning: Could not save sampling comparison animation: {e}")
    plt.close()

def compare_step_sizes(model, init_samples, config):
    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    sampler = EqMSampler(model, config)

    for idx, step_size in enumerate(step_sizes):
        print(f"Sampling with step_size={step_size}")
        samples = sampler.sample_gd(init_samples[:25],
                                   step_size=step_size,
                                   steps=50)

        vis_samples = (samples + 1) / 2
        vis_samples = vis_samples.clamp(0, 1)

        grid = torchvision.utils.make_grid(vis_samples, nrow=5, normalize=False)
        axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[idx].set_title(f"Step Size: {step_size}")
        axes[idx].axis('off')

    plt.suptitle("Effect of Different Step Sizes on Generation")
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/step_size_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def test_partial_denoising(model, config):
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    real_images = torch.stack([dataset[i][0] for i in range(25)]).to(config.device)

    noise = torch.randn_like(real_images)
    gamma = 0.5
    partial_noised = gamma * real_images + (1 - gamma) * noise

    with torch.no_grad():
        current = partial_noised.clone()
        trajectory = []
        step_labels = []
        target_steps = [0, 1, 5, 10, 20, 45]

        for i in range(50):
            if i in target_steps:
                trajectory.append(current.cpu())
                step_labels.append(i)

            grad = model(current)
            current = current - 0.01 * grad

        if len(trajectory) < 5:
            trajectory.append(current.cpu())
            step_labels.append(49)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes_flat = axes.flatten()

    images_and_titles = [
        (real_images, "Original"),
        (partial_noised, f"Partial Noise (γ={gamma})")
    ] + [(trajectory[i], f"Step {step_labels[i]}") for i in range(len(trajectory))]

    for idx, (ax, (img, title)) in enumerate(zip(axes_flat, images_and_titles)):
        if idx >= len(axes_flat):
            break

        img_denorm = (img + 1) / 2  # From [-1,1] to [0,1]
        img_denorm = img_denorm.clamp(0, 1)

        grid_img = torchvision.utils.make_grid(img_denorm, nrow=5, normalize=False)
        ax.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    for idx in range(len(images_and_titles), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.suptitle("Partial Denoising with EqM", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/partial_denoising.png", dpi=150, bbox_inches='tight')
    plt.close()
