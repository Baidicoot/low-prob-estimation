import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import os

import tqdm

from dataclasses import dataclass

import glob
from PIL import Image

from typing import Optional

def animate_plots(output_dir, num_frames, duration = 1000):
    # Create the GIF from saved images
    images = []
    filenames = [f"{output_dir}/xs_{i}.png" for i in range(num_frames)]
    for filename in filenames:
        images.append(Image.open(filename))
    
    duration_ms = np.ceil(duration / num_frames)

    # Save the GIF
    images[0].save(
        f"{output_dir}/animation.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,  # Duration for each frame in milliseconds
        loop=0
    )

@dataclass
class AnimateConfig:
    output_dir: str = None

    xlims: Optional[tuple[float, float]] = (-10, 10)
    ylims: Optional[tuple[float, float]] = (-10, 10)
    
    hist: bool = True
    size: int = 100
    d_lims: Optional[tuple[float, float]] = None

    duration: int = 1000
    capture_every: int = 100

@dataclass
class SampleConfig:
    steps: int = 1_000
    start_lr: float = 0.1
    end_lr: float = 0.0001
    start_beta: float = 10
    end_beta: float = 100

    animate: Optional[AnimateConfig] = None
    progress: bool = False

def sample_langevin(particles, log_density, cost=None, threshold=None, config = SampleConfig()):
    if cost is not None and threshold is None:
        raise ValueError("Threshold must be provided if cost is provided")
    
    def sample_step(xs, beta, step_size):
        xs.requires_grad_(True)
        scaled_cost = -torch.relu(beta * (cost(xs) - threshold)) if cost is not None else 0
        drift = torch.autograd.grad((log_density(xs) + scaled_cost).sum(), xs)[0]
        diffusion = torch.randn_like(xs)
        xs_new = xs.detach() + step_size * drift + np.sqrt(2 * step_size) * diffusion
        return xs_new

    xs = particles.clone()

    if config.animate is not None:
        os.makedirs(config.animate.output_dir, exist_ok = True)
        frame_idx = 0

    idxs = range(config.steps)
    if config.progress:
        idxs = tqdm.tqdm(idxs)

    for i in idxs:
        progress = i / config.steps
        beta = config.start_beta * (config.end_beta / config.start_beta) ** progress
        lr = config.start_lr * (config.end_lr / config.start_lr) ** progress
        xs = sample_step(xs, beta, lr)

        if config.animate is not None and i % config.animate.capture_every == 0:
            plt.figure(figsize = (10, 8))
            if config.animate.hist:
                plt.hist2d(
                    xs[:, 0],
                    xs[:, 1],
                    bins = config.animate.size,
                    range = [config.animate.xlims, config.animate.ylims],
                    density = True,
                    cmap = "inferno",
                    vmin = None if config.animate.d_lims is None else config.animate.d_lims[0],
                    vmax = None if config.animate.d_lims is None else config.animate.d_lims[1]
                )
                plt.colorbar()
            else:
                plt.scatter(
                    xs[:, 0],
                    xs[:, 1],
                    s = 0.1,
                    alpha = 0.5,
                    c = "red",
                    marker = "."
                )
                plt.xlim(config.animate.xlims)
                plt.ylim(config.animate.ylims)
            plt.title(f"Step {i}")
            plt.tight_layout()
            plt.savefig(f"{config.animate.output_dir}/xs_{frame_idx}.png")
            plt.close()
            frame_idx += 1

    if config.animate is not None:
        animate_plots(config.animate.output_dir, frame_idx, config.animate.duration)

    return xs

def integrate_over_plane(xlim, ylim, function, size = 100):
    xs, ys = torch.meshgrid(torch.linspace(xlim[0], xlim[1], size), torch.linspace(ylim[0], ylim[1], size))
    xs = xs.flatten()
    ys = ys.flatten()
    samples = torch.stack([xs, ys], dim = 1)
    return function(samples).mean() * (xlim[1] - xlim[0]) * (ylim[1] - ylim[0]) / size ** 2