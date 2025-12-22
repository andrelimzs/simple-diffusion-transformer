import torch


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
