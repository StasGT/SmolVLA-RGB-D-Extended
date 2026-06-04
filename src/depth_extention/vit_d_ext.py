import torch
import torch.nn as nn


class DepthCrossAttention(nn.Module):
    def __init__(self, config=None, embed_dim=960, num_heads=8):
        super().__init__()
        self.scale_factor = getattr(config, 'scale_factor', 4)

        # 512x512 --> 32x32 = 1024
        self.depth_init_dim = 512
        self.depth_patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=self.depth_init_dim,
            kernel_size=16,
            stride=16
        )

        self.num_patches = 1024  # (512 / 16) * (512 / 16)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.depth_init_dim) * 0.02)

        # pixel_shuffle --> 960
        # scale_factor=4, 4^2 = (512 * 16 = 8192)
        shuffled_dim = self.depth_init_dim * (self.scale_factor ** 2)
        self.depth_projection = nn.Sequential(
            nn.Linear(shuffled_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 3. Q = RGB (64, 960), K, V = Depth (64, 960)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def pixel_shuffle(self, x, scale_factor=4):
        # SmolVLMConnector
        bsz, seq, embed_dim = x.size()
        height = width = int(seq ** 0.5)  # 1024 --> 32x32
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor ** 2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor ** 2)), embed_dim * (scale_factor ** 2))
        return x

    def forward(self, rgb_emb, depth_img):
        # Conv --> [8, 512, 32, 32]
        x = self.depth_patch_embed(depth_img)

        # --> [8, 1024, 512]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x + self.pos_embedding.to(device=x.device, dtype=x.dtype)

        # Pixel Shuffle -> [8, 64, 8192]
        x = self.pixel_shuffle(x, self.scale_factor)

        # --> [8, 64, 960]
        depth_emb = self.depth_projection(x)

        # RGB top -->, depth --> K, V
        attn_output, _ = self.cross_attn(
            query=rgb_emb,
            key=depth_emb,
            value=depth_emb
        )

        return self.layer_norm(rgb_emb + attn_output)
