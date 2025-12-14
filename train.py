import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse
from tqdm import tqdm
import os

from model import SimpleDiT


def get_args():
    parser = argparse.ArgumentParser(description="Train SimpleDiT")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()

    # Ensure directories exist
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="wandb" if args.log else None,
        project_dir="./logs",
    )

    set_seed(args.seed)

    # Create model
    model = SimpleDiT(
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
    )

    # Create dataset and dataloader
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
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
            # MNIST returns (images, labels), we only need images
            images, _ = batch

            with accelerator.accumulate(model):
                # Sample random timesteps
                timesteps = torch.randint(
                    0, args.num_timesteps, (images.shape[0],), device=images.device
                )

                # Sample noise
                noise = torch.randn_like(images)

                # Add noise to batch (simple linear schedule)
                t_normalized = timesteps.float() / args.num_timesteps
                t_normalized = t_normalized[:, None, None, None]
                noisy_batch = (1 - t_normalized) * images + t_normalized * noise

                # Predict noise
                noise_pred = model(noisy_batch, timesteps)

                # MSE loss
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=loss.item())
                accelerator.log({"loss": loss.item()}, step=global_step)

            global_step += 1

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            accelerator.save_state(f"./checkpoints/epoch_{epoch + 1}")
            accelerator.print(f"Saved checkpoint at epoch {epoch + 1}")

    # End tracking
    accelerator.end_training()


if __name__ == "__main__":
    main()
