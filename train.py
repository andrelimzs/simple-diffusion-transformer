import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
import argparse
from tqdm import tqdm
import os
from torchvision.utils import save_image, make_grid
import wandb

from model import SimpleDiT
from eval import generate_samples


def get_args():
    parser = argparse.ArgumentParser(description="Train SimpleDiT")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--log", action="store_true")
    parser.add_argument(
        "--sample_every",
        type=int,
        default=500,
        help="Save samples every",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Ensure directories exist
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./samples", exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="wandb" if args.log else None,
        project_dir="./logs",
    )

    set_seed(args.seed)

    # Download dataset
    if accelerator.is_main_process:
        datasets.MNIST(root=args.data_dir, train=True, download=True)
        datasets.MNIST(root=args.data_dir, train=False, download=True)

    accelerator.wait_for_everyone()

    # Load pretrained VAE and freeze weights
    vae = AutoencoderKL.from_pretrained(args.vae_model)
    vae.requires_grad_(False)
    vae.eval()

    # Create model (operates in latent space)
    model = SimpleDiT(
        in_channels=args.latent_channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        num_classes=10,
    )
    model.initialize_weights()

    # Create dataset and dataloader (VAE expects RGB images)
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert MNIST to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )
    dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    vae = vae.to(accelerator.device)

    # Initialize tracking
    if accelerator.is_main_process:
        accelerator.init_trackers("simple-dit-training")

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )

        for batch in progress_bar:
            # MNIST returns (images, labels)
            images, labels = batch

            with accelerator.accumulate(model):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample random timesteps
                timesteps = torch.randint(
                    0, args.num_timesteps, (latents.shape[0],), device=latents.device
                )

                # Sample noise
                noise = torch.randn_like(latents)

                # Add noise to latents (simple linear schedule)
                t_normalized = timesteps.float() / args.num_timesteps
                t_normalized = t_normalized[:, None, None, None]
                noisy_latents = (1 - t_normalized) * latents + t_normalized * noise

                # Predict noise
                noise_pred = model(noisy_latents, timesteps, labels)

                # MSE loss
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=loss.item())
                accelerator.log({"loss": loss.item()}, step=global_step)

            # Generate samples for evaluation
            if args.sample_every > 0 and (global_step % args.sample_every == 0):
                # Generate model samples
                labels = torch.arange(9)
                images = generate_samples(
                    accelerator.unwrap_model(model), vae, labels, accelerator.device
                )

                if accelerator.is_main_process:
                    # Save images
                    save_path = os.path.join("./samples", f"step_{global_step:08d}.png")
                    save_image(images, save_path, nrow=3)

                    # Log if configured
                    if args.log:
                        image_to_log = wandb.Image(
                            make_grid(images, nrow=3), caption=f"Step {global_step}"
                        )
                        accelerator.log(
                            {"samples": image_to_log},
                            step=global_step,
                        )

            global_step += 1

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            accelerator.save_state(f"./checkpoints/epoch_{epoch + 1}")
            accelerator.print(f"Saved checkpoint at epoch {epoch + 1}")

    # End tracking
    accelerator.end_training()


if __name__ == "__main__":
    main()
