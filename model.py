import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        t: tensor of timesteps (M, )
        dim: output_dim
        out: (M, D) if M divisible by 2 else (M+1, D)
        """
        # Create exponentially spaced frequencies
        freqs = (1 / max_period ** torch.linspace(start=0, end=1, steps=dim // 2)).to(
            device=t.device
        )

        # Compute grid as outer product of time and frequency
        grid = t.reshape(-1, 1).float() @ freqs.reshape(1, -1)

        embedding = torch.cat([torch.cos(grid), torch.sin(grid)], dim=-1)  # (T, D)

        # Pad if necessary
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout):
        super().__init__()
        use_cfg = dropout > 0
        self.embedding = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, labels, is_train):
        if is_train and self.dropout > 0:
            ids_to_drop = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout
            )
            labels = torch.where(ids_to_drop, -1, labels)
        labels = torch.where(labels == -1, self.num_classes, labels)
        return self.embedding(labels)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-attention with modulation
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with modulation
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


def compute_2d_sincos_embed(embed_dim, grid_size):
    """
    embed_dim: output_dim, must be divisible by 2
    grid_size: size of grid, must be int
    out: (grid_size**2, embed_dim)
    """
    # Create grid of index positions
    grid_h, grid_w = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="xy"
    )

    embedding = torch.concat(
        [
            compute_1d_sincos_embed(embed_dim // 2, grid_h.flatten()),
            compute_1d_sincos_embed(embed_dim // 2, grid_w.flatten()),
        ],
        dim=1,
    )

    return embedding


def compute_1d_sincos_embed(embed_dim, pos):
    """
    embed_dim: output_dim, must be divisible by 2
    pos: tensor of positions (M, )
    out: (M, D)
    """
    assert embed_dim % 2 == 0

    # Compute exponentially spaced frequencies
    exp = torch.arange(embed_dim // 2) / (embed_dim / 2)
    freqs = 1 / 1e4**exp

    # Compute grid as outer product of position and frequency
    pos_freq_grid = pos.float().reshape(-1, 1) @ freqs.reshape(1, -1)

    # Embedding is concatenation of sin and cos components
    embedding = torch.concat(
        [torch.sin(pos_freq_grid), torch.cos(pos_freq_grid)], dim=1
    )

    return embedding


class SimpleDiT(nn.Module):
    def __init__(
        self,
        input_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        num_classes=10,
        class_dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )

        # Time embedding
        self.time_embed = TimestepEmbedding(hidden_size)

        # Label embedding
        self.label_embed = LabelEmbedding(num_classes, hidden_size, class_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        # Final layer
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.final_linear = nn.Linear(
            hidden_size, patch_size * patch_size * in_channels
        )

    def initialize_weights(self):
        with torch.no_grad():
            # Initialize pos_embed with fixed 2d sin-cos embedding
            _, num_patches, hidden_size = self.pos_embed.shape
            sincos_embed = compute_2d_sincos_embed(hidden_size, int(num_patches**0.5))
            self.pos_embed.copy_(sincos_embed.unsqueeze(0))

    def patchify(self, x):
        """
        x: (B, C, H, W)
        out: (B, N, D)
        """
        x = self.patch_embed(x)
        return x.flatten(2).transpose(1, 2)

    def unpatchify(self, x):
        """
        x: (B, N, patch_size**2 * C)
        out: (B, C, H, W)
        """
        b = x.shape[0]
        h = w = int(x.shape[1] ** 0.5)
        c = self.in_channels
        x = x.reshape(b, h, w, self.patch_size, self.patch_size, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(
            b, c, h * self.patch_size, w * self.patch_size
        )

    def forward(self, x, t, y):
        """
        x: (B, C, H, W)
        t: (B, )
        y: (B, )
        out: (B, C, H, W)
        """
        # Patchify
        x = self.patchify(x) + self.pos_embed  # (B, N, D)

        # Time conditioning
        t_emb = self.time_embed(t.float())  # (B, D)

        # Label conditioning
        y_emb = self.label_embed(y, self.training)  # (B, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb + y_emb)

        # Final layer
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)  # (B, N, patch_size * patch_size * in_channels)

        # Unpatchify
        x = self.unpatchify(x)  # (B, in_channels, H, W)

        return x

    def cfg_forward(self, x, t, y, scale=5.0):
        """
        Batched classifier-free guidance forward pass
        """
        # Compute both conditionl and unconditional scores in one forward
        y_batched = torch.cat([y, -1 * torch.ones_like(y)])
        batched_out = self.forward(x.repeat(2, 1, 1, 1), t.repeat(2), y_batched)

        cond_score, uncond_score = batched_out.chunk(2)

        cfg_score = uncond_score + scale * (cond_score - uncond_score)
        return cfg_score

    @torch.no_grad()
    def p_sample_loop(
        self, labels, scale=5.0, train_timesteps=1000, inference_steps=None
    ):
        num_samples = labels.shape[0]
        device = next(self.parameters()).device

        assert labels.device == device

        steps = inference_steps if inference_steps is not None else train_timesteps

        # Infer spatial size from pos_embed
        h = w = int(self.num_patches**0.5) * self.patch_size

        latents = torch.randn(num_samples, self.in_channels, h, w, device=device)

        for t in reversed(range(steps)):
            t = int(t / steps * train_timesteps)  # Map inference steps to train steps
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_norm = timesteps.float() / train_timesteps
            t_norm = t_norm[:, None, None, None]

            noise_pred = self.cfg_forward(latents, timesteps, labels, scale)

            # Invert the training forward mixture:
            # noisy = (1-t)*x0 + t*eps  => x0_hat = (noisy - t*eps_hat)/(1-t)
            denom = (1.0 - t_norm).clamp(min=1e-5)
            x0_hat = (latents - t_norm * noise_pred) / denom

            # Use x0_hat as next latent; add a bit of noise except at final step
            if t > 0:
                latents = (1.0 - t_norm) * x0_hat + t_norm * torch.randn_like(latents)
            else:
                latents = x0_hat

        return latents
