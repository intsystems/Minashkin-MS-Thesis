import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from eqm import EqMSampler
from fid import _embed, fid_from_statistics, fid_score_from_ref_stats, reference_statistics


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


def plot_loss_list(losses, title, ylabel, filename, config, y_log=False):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    if y_log:
        plt.yscale("log")
    plt.grid(True)
    plt.savefig(f"{config.save_dir}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()


def _project_features(features, proj_dim, seed=0):
    if proj_dim is None or proj_dim <= 0 or features.shape[1] <= proj_dim:
        return features, "FID"

    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(features.shape[1], proj_dim)).astype(np.float32)
    projection /= np.sqrt(proj_dim)
    return features @ projection, f"FID-{proj_dim}"


def _frame_fids_projected(trajectory, real_ref, device, n_fid, proj_dim):
    frames = [frame[:n_fid].detach().cpu() for frame in trajectory]
    fake = torch.cat(frames, dim=0)

    real_features = _embed(real_ref[:n_fid], device)
    fake_features = _embed(fake, device).reshape(len(frames), n_fid, -1)

    all_features = np.concatenate(
        [real_features[None, :, :], fake_features],
        axis=0,
    ).reshape(-1, real_features.shape[1])
    projected, fid_label = _project_features(all_features, proj_dim)

    real_features = projected[:n_fid]
    fake_features = projected[n_fid:].reshape(len(frames), n_fid, -1)

    mu_real = real_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    fids = [
        fid_from_statistics(
            mu_real,
            sigma_real,
            frame_features.mean(axis=0),
            np.cov(frame_features, rowvar=False),
        )
        for frame_features in fake_features
    ]
    return fids, fid_label

def visualize_samples(
    samples, title, filename, config, nrow=8, real_ref=None, device=None
):
    if real_ref is not None and device is not None:
        n = min(samples.shape[0], real_ref.shape[0])
        mu, sig = reference_statistics(real_ref[:n], device)
        title = f"{title}\nFID={fid_score_from_ref_stats(mu, sig, samples[:n].detach().cpu(), device):.2f}"

    samples = (samples + 1) / 2  # Denormalize
    samples = samples.clamp(0, 1)

    grid = torchvision.utils.make_grid(samples, nrow=nrow, normalize=False)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{config.save_dir}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def visualize_sampling_trajectory_animation(
    trajectory,
    filename,
    config,
    real_ref=None,
    device=None,
    n_frames=50,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    total_steps = len(trajectory)
    step_indices = np.linspace(
        0, total_steps - 1, min(n_frames, total_steps), dtype=int
    )

    n_vis = 25
    fids = None
    if (
        real_ref is not None
        and device is not None
        and getattr(config, "fid_on_gif", False)
    ):
        n_fid = min(n_vis, real_ref.shape[0])
        mu, sig = reference_statistics(real_ref[:n_fid], device)
        fids = [
            fid_score_from_ref_stats(
                mu, sig, trajectory[int(step)][:n_fid].detach().cpu(), device
            )
            for step in step_indices
        ]

    def update(frame_idx):
        ax.clear()
        step = int(step_indices[frame_idx])
        samples = trajectory[step][:n_vis]
        samples = (samples + 1) / 2  # Denormalize
        samples = samples.clamp(0, 1)

        grid = torchvision.utils.make_grid(samples, nrow=5, normalize=False)
        ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        ttl = f"Step {step}/{total_steps - 1}"
        if fids is not None:
            ttl += f"  |  FID={fids[frame_idx]:.2f}"
        ax.set_title(ttl)
        ax.axis('off')

    anim = animation.FuncAnimation(
        fig, update, frames=len(step_indices), interval=200, blit=False
    )

    try:
        anim.save(f"{config.save_dir}/{filename}", writer='pillow', fps=5)
        print(f"Saved animation: {filename}")
    except Exception as e:
        print(f"Warning: Could not save animation {filename}: {e}")
        visualize_samples(
            trajectory[0][:n_vis],
            "Sampling Start",
            "sampling_start.png",
            config,
            nrow=5,
            real_ref=real_ref,
            device=device,
        )
        visualize_samples(
            trajectory[-1][:n_vis],
            "Sampling End",
            "sampling_end.png",
            config,
            nrow=5,
            real_ref=real_ref,
            device=device,
        )
    plt.close(fig)

def compare_step_sizes_animation(
    model, config, init_samples, real_ref=None, device=None
):

    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    sampler = EqMSampler(model, config)
    steps = 50

    print("Generating trajectories for different step sizes...")
    trajectories = []
    for step_size in step_sizes:
        print(f"  step_size={step_size}")
        _, trajectory = sampler.sample_gd(
            init_samples.clone(),
            steps=steps,
            step_size=step_size,
            return_trajectory=True,
        )
        trajectories.append(trajectory)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    n_vis = 9
    frame_fids = None
    if (
        real_ref is not None
        and device is not None
        and getattr(config, "fid_on_gif", False)
    ):
        n_fid = min(n_vis, real_ref.shape[0])
        mu, sig = reference_statistics(real_ref[:n_fid], device)
        frame_fids = []
        for frame in range(steps):
            frame_fids.append(
                [
                    fid_score_from_ref_stats(
                        mu, sig, traj[frame][:n_fid].detach().cpu(), device
                    )
                    for traj in trajectories
                ]
            )

    def update(frame):
        for idx, (ax, step_size, trajectory) in enumerate(
            zip(axes, step_sizes, trajectories)
        ):
            ax.clear()
            samples = trajectory[frame][:n_vis]  # 3x3 grid
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            grid = torchvision.utils.make_grid(samples, nrow=3, normalize=False)
            ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
            ttl = f"Step Size: {step_size}"
            if frame_fids is not None:
                ttl += f"\nFID={frame_fids[frame][idx]:.2f}"
            ax.set_title(ttl, fontsize=10)
            ax.axis('off')

        fig.suptitle(f"Generation Step {frame}", fontsize=16)
        return axes

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=200, blit=False)
    try:
        anim.save(f"{config.save_dir}/step_size_comparison.gif", writer='pillow', fps=5)
        print("Saved step size comparison animation")
    except Exception as e:
        print(f"Warning: Could not save step size animation: {e}")
        compare_step_sizes(model, init_samples, config, real_ref, device)
    plt.close(fig)

def compare_sampling_algorithms_animation(
    model, config, init_samples, real_ref=None, device=None
):
    sampler = EqMSampler(model, config)
    steps = 50

    print("Generating trajectories for GD vs NAG vs Muon comparison...")
    _, trajectory_gd = sampler.sample_gd(
        init_samples.clone(), steps=steps, return_trajectory=True
    )
    _, trajectory_nag = sampler.sample_nag(
        init_samples.clone(), steps=steps, return_trajectory=True
    )
    _, trajectory_muon = sampler.sample_muon(
        init_samples.clone(), steps=steps, return_trajectory=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_gd, ax_nag, ax_muon = axes

    n_vis = 16
    fids_gd = fids_nag = fids_muon = None
    if (
        real_ref is not None
        and device is not None
        and getattr(config, "fid_on_gif", False)
    ):
        n_fid = min(n_vis, real_ref.shape[0])
        mu, sig = reference_statistics(real_ref[:n_fid], device)
        fids_gd = [
            fid_score_from_ref_stats(
                mu, sig, trajectory_gd[f][:n_fid].detach().cpu(), device
            )
            for f in range(steps)
        ]
        fids_nag = [
            fid_score_from_ref_stats(
                mu, sig, trajectory_nag[f][:n_fid].detach().cpu(), device
            )
            for f in range(steps)
        ]
        fids_muon = [
            fid_score_from_ref_stats(
                mu, sig, trajectory_muon[f][:n_fid].detach().cpu(), device
            )
            for f in range(steps)
        ]

    def update(frame):
        # GD
        ax_gd.clear()
        samples_gd = trajectory_gd[frame][:n_vis]
        samples_gd = (samples_gd + 1) / 2
        samples_gd = samples_gd.clamp(0, 1)
        grid_gd = torchvision.utils.make_grid(samples_gd, nrow=4, normalize=False)
        ax_gd.imshow(grid_gd.permute(1, 2, 0).numpy(), cmap='gray')
        ttl_gd = f"Vanilla GD\nStep {frame}"
        if fids_gd is not None:
            ttl_gd += f"\nFID={fids_gd[frame]:.2f}"
        ax_gd.set_title(ttl_gd, fontsize=11)
        ax_gd.axis('off')

        # NAG
        ax_nag.clear()
        samples_nag = trajectory_nag[frame][:n_vis]
        samples_nag = (samples_nag + 1) / 2
        samples_nag = samples_nag.clamp(0, 1)
        grid_nag = torchvision.utils.make_grid(samples_nag, nrow=4, normalize=False)
        ax_nag.imshow(grid_nag.permute(1, 2, 0).numpy(), cmap='gray')
        ttl_nag = f"NAG-GD\nStep {frame}"
        if fids_nag is not None:
            ttl_nag += f"\nFID={fids_nag[frame]:.2f}"
        ax_nag.set_title(ttl_nag, fontsize=11)
        ax_nag.axis('off')

        # Muon
        ax_muon.clear()
        samples_muon = trajectory_muon[frame][:n_vis]
        samples_muon = (samples_muon + 1) / 2
        samples_muon = samples_muon.clamp(0, 1)
        grid_muon = torchvision.utils.make_grid(samples_muon, nrow=4, normalize=False)
        ax_muon.imshow(grid_muon.permute(1, 2, 0).numpy(), cmap='gray')
        ttl_muon = f"Muon\nStep {frame}"
        if fids_muon is not None:
            ttl_muon += f"\nFID={fids_muon[frame]:.2f}"
        ax_muon.set_title(ttl_muon, fontsize=11)
        ax_muon.axis('off')

        fig.suptitle(
            f"EqM Sampling Algorithm Comparison\n{frame}/{steps - 1} steps",
            fontsize=14,
            fontweight='bold',
        )
        return axes

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=200, blit=False)
    try:
        anim.save(f"{config.save_dir}/sampling_comparison.gif", writer='pillow', fps=5)
        print("Saved GD vs NAG vs Muon comparison animation")
    except Exception as e:
        print(f"Warning: Could not save sampling comparison animation: {e}")
    plt.close(fig)


def compare_two_sampling_algorithms_animation(
    trajectory_a,
    trajectory_b,
    name_a,
    name_b,
    filename,
    config,
    real_ref=None,
    device=None,
):
    steps = min(len(trajectory_a), len(trajectory_b)) - 1
    trajectory_a = trajectory_a[: steps + 1]
    trajectory_b = trajectory_b[: steps + 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_a, ax_b = axes

    n_vis = 16
    fids_a = fids_b = None
    fid_label = "FID"
    if (
        real_ref is not None
        and device is not None
        and getattr(config, "fid_on_gif", False)
    ):
        n_fid = min(n_vis, real_ref.shape[0])
        proj_dim = getattr(config, "gif_fid_projection_dim", 64)
        fids_a, fid_label = _frame_fids_projected(
            trajectory_a,
            real_ref,
            device,
            n_fid,
            proj_dim,
        )
        fids_b, fid_label = _frame_fids_projected(
            trajectory_b,
            real_ref,
            device,
            n_fid,
            proj_dim,
        )

    def update(frame):
        ax_a.clear()
        samples_a = trajectory_a[frame][:n_vis]
        samples_a = (samples_a + 1) / 2
        samples_a = samples_a.clamp(0, 1)
        grid_a = torchvision.utils.make_grid(samples_a, nrow=4, normalize=False)
        ax_a.imshow(grid_a.permute(1, 2, 0).numpy(), cmap="gray")
        ttl_a = f"{name_a}\nStep {frame}"
        if fids_a is not None:
            ttl_a += f"\n{fid_label}={fids_a[frame]:.2f}"
        ax_a.set_title(ttl_a, fontsize=11)
        ax_a.axis("off")

        ax_b.clear()
        samples_b = trajectory_b[frame][:n_vis]
        samples_b = (samples_b + 1) / 2
        samples_b = samples_b.clamp(0, 1)
        grid_b = torchvision.utils.make_grid(samples_b, nrow=4, normalize=False)
        ax_b.imshow(grid_b.permute(1, 2, 0).numpy(), cmap="gray")
        ttl_b = f"{name_b}\nStep {frame}"
        if fids_b is not None:
            ttl_b += f"\n{fid_label}={fids_b[frame]:.2f}"
        ax_b.set_title(ttl_b, fontsize=11)
        ax_b.axis("off")

        fig.suptitle(
            f"{name_a} vs {name_b}\n{frame}/{steps} steps",
            fontsize=14,
            fontweight="bold",
        )
        return axes

    anim = animation.FuncAnimation(
        fig, update, frames=steps + 1, interval=200, blit=False
    )
    try:
        anim.save(f"{config.save_dir}/{filename}", writer="pillow", fps=5)
        print(f"Saved comparison animation: {filename}")
    except Exception as e:
        print(f"Warning: Could not save comparison animation {filename}: {e}")
    plt.close(fig)


def plot_two_sampling_fid_curve(
    trajectory_a,
    trajectory_b,
    name_a,
    name_b,
    filename,
    config,
    real_ref,
    device,
):
    steps = min(len(trajectory_a), len(trajectory_b)) - 1
    trajectory_a = trajectory_a[: steps + 1]
    trajectory_b = trajectory_b[: steps + 1]

    n_fid = min(getattr(config, "gif_fid_n_samples", 16), real_ref.shape[0])
    proj_dim = getattr(config, "gif_fid_projection_dim", 64)
    fids_a, fid_label = _frame_fids_projected(
        trajectory_a,
        real_ref,
        device,
        n_fid,
        proj_dim,
    )
    fids_b, _ = _frame_fids_projected(
        trajectory_b,
        real_ref,
        device,
        n_fid,
        proj_dim,
    )

    x = np.arange(steps + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, fids_a, label=name_a, linewidth=2)
    plt.plot(x, fids_b, label=name_b, linewidth=2)
    plt.xlabel("Sampling step")
    plt.ylabel(fid_label)
    plt.title(f"{fid_label} by sampling step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{config.save_dir}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved FID curve: {filename}")


def compare_step_sizes(model, init_samples, config, real_ref=None, device=None):
    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    sampler = EqMSampler(model, config)

    for idx, step_size in enumerate(step_sizes):
        print(f"Sampling with step_size={step_size}")
        samples = sampler.sample_gd(
            init_samples[:25], step_size=step_size, steps=50
        )

        vis_samples = (samples + 1) / 2
        vis_samples = vis_samples.clamp(0, 1)

        grid = torchvision.utils.make_grid(vis_samples, nrow=5, normalize=False)
        axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ttl = f"Step Size: {step_size}"
        if real_ref is not None and device is not None:
            n_fid = min(samples.shape[0], real_ref.shape[0])
            mu, sig = reference_statistics(real_ref[:n_fid], device)
            ttl += f"\nFID={fid_score_from_ref_stats(mu, sig, samples[:n_fid].detach().cpu(), device):.2f}"
        axes[idx].set_title(ttl)
        axes[idx].axis('off')

    plt.suptitle("Effect of Different Step Sizes on Generation")
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/step_size_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def test_partial_denoising(model, config, real_ref=None, device=None):
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
        (partial_noised, f"Partial Noise (γ={gamma})"),
    ] + [(trajectory[i], f"Step {step_labels[i]}") for i in range(len(trajectory))]

    n_fid = 25
    mu_sig = None
    if real_ref is not None and device is not None:
        mu_sig = reference_statistics(real_ref[:n_fid], device)

    for idx, (ax, (img, title)) in enumerate(zip(axes_flat, images_and_titles)):
        if idx >= len(axes_flat):
            break

        if mu_sig is not None:
            mu, sig = mu_sig
            fid_val = fid_score_from_ref_stats(mu, sig, img[:n_fid].detach().cpu(), device)
            title = f"{title}\nFID={fid_val:.2f}"

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
