import math
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance


@torch.no_grad()
def generate_samples(model, vae, labels, device):
    # Temporarily switch to eval for sampling
    was_training = model.training
    model.eval()

    # Generate latent samples
    latents = model.p_sample_loop(labels.to(device))

    # Decode latents -> images in [-1, 1]
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images.clamp(-1, 1) + 1) / 2.0  # -> [0,1]

    if was_training:
        model.train()

    return images


def compute_fid(dataset, model, vae, num_samples, accelerator, batch_size=128):
    device = accelerator.device
    was_training = model.training
    model.eval()

    # Initialize FID metric (feature=2048 is standard)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Calculate samples per rank
    samples_per_rank = math.ceil(num_samples / accelerator.num_processes)
    num_batches = math.ceil(samples_per_rank / batch_size)

    # Update with real images from dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process:
        print(
            f"Computing FID on {num_samples} samples ({samples_per_rank} per rank)..."
        )

    for real_images, real_labels in islice(dataloader, num_batches):
        # Update real
        real_images_uint8 = ((real_images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fid.update(real_images_uint8, real=True)

        # Generate fake images
        with torch.no_grad():
            latents = model.p_sample_loop(real_labels)

            # Decode latents -> images in [-1, 1]
            latents = latents / vae.config.scaling_factor
            fake_images = vae.decode(latents).sample

            # Convert to uint8 [0, 255]
            fake_images_uint8 = ((fake_images.clamp(-1, 1) + 1) / 2 * 255).to(
                torch.uint8
            )

        fid.update(fake_images_uint8, real=False)

    if was_training:
        model.train()

    return fid.compute()
