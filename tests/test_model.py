import torch
from model import SimpleDiT


def test_simple_dit_initialization():
    model = SimpleDiT(
        input_size=8,
        in_channels=4,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        patch_size=2,
        num_classes=10,
    )
    assert isinstance(model, SimpleDiT)
    # input_size=8, patch_size=2 -> 4x4 patches = 16 patches
    assert model.pos_embed.shape == (1, 16, 32)


def test_simple_dit_forward():
    model = SimpleDiT(
        input_size=8,
        in_channels=4,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        patch_size=2,
        num_classes=10,
    )

    # Create dummy input
    # Batch size 2, 4 channels, 8x8 latent size
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))

    output = model(x, t, y)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_simple_dit_shapes():
    # Test with different image sizes
    model = SimpleDiT(
        input_size=16,
        in_channels=4,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        patch_size=2,
        num_classes=10,
    )

    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([500])
    y = torch.tensor([1])

    output = model(x, t, y)
    assert output.shape == (1, 4, 16, 16)
